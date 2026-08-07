"""
app.py
------
Single Streamlit interface that runs the full Capstone Group 4 pipeline:
 
    User Input
        -> ProfileEncoder              (predicted occupation, profile fit score)
        -> Employment XGBoost          (employment probability)
        -> Income XGBoost              (predicted annual income)
        -> Adzuna API                  (available jobs)
        -> Recommendation Engine       (ranked jobs, match scores, explanations)
        -> Dashboard + Folium map
 
Run with:
    streamlit run app.py
"""
 
from __future__ import annotations
 
import os
 
import folium
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st
from streamlit_folium import st_folium
 
from pipeline import config
from pipeline.profile_encoder import encode_profile
from pipeline.xgb_models import get_models
from pipeline.adzuna_client import get_jobs_multi_page, build_search_keyword
from pipeline.job_source import get_jobs_from_dataset, dataset_info
from pipeline.recommendation_engine import recommend_jobs
 
st.set_page_config(
    page_title="Newcomer Career Navigator — Ontario",
    page_icon="🍁",
    layout="wide",
)
 
# ---------------------------------------------------------------------------
# Cached loaders
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_label_encoders():
    return joblib.load(config.LABEL_ENCODERS_PATH)
 
 
@st.cache_resource(show_spinner=False)
def load_xgb_models():
    return get_models()
 
 
@st.cache_data(show_spinner=False)
def load_housing_data():
    if os.path.exists(config.HOUSING_DATA_PATH):
        return pd.read_csv(config.HOUSING_DATA_PATH)
    return None
 
 
@st.cache_data(show_spinner=False, ttl=3600)
def cached_get_jobs_live(keyword: str, city: str):
    return get_jobs_multi_page(keyword, city)
 
 
@st.cache_data(show_spinner=False)
def cached_get_jobs_local(keyword: str, city: str):
    return get_jobs_from_dataset(keyword, city)
 
 
def _with_extra_options(base_options: list[str], extra_options: list[str]) -> list[str]:
    """Return base_options with any extra_options appended, skipping duplicates.
 
    Used to widen dropdowns (e.g. adding "North America" / "South America" to the
    region list, or "Other" to occupation lists) without disturbing the original,
    model-trained option ordering.
    """
    seen = {opt.strip().lower() for opt in base_options}
    widened = list(base_options)
    for opt in extra_options:
        if opt.strip().lower() not in seen:
            widened.append(opt)
            seen.add(opt.strip().lower())
    return widened
 
 
def _safe_model_category(field_label: str, selected_value: str, known_classes: list[str]) -> str:
    """Map a selected dropdown value back onto a class the trained models understand.
 
    The dropdowns show a few extra, more complete options (like "Other" or new
    regions) that aren't part of the original trained label encoders. If the user
    picks one of those, fall back to the closest known category so the XGBoost /
    ProfileEncoder models still receive a value they were trained on, and let the
    user know a substitution was made.
    """
    if selected_value in known_classes:
        return selected_value
    fallback = "Other" if "Other" in known_classes else known_classes[0]
    st.caption(
        f"ℹ️ \"{selected_value}\" isn't part of the trained {field_label} categories yet, "
        f"so predictions for this run use the closest available category (\"{fallback}\")."
    )
    return fallback
 
 
def artifact_status() -> list[dict]:
    checks = [
        ("ProfileEncoder network", config.PROFILE_ENCODER_MODEL_PATH),
        ("ProfileEncoder multitask model", config.MULTITASK_MODEL_PATH),
        ("ProfileEncoder preprocessor", config.PREPROCESSOR_PATH),
        ("Employment XGBoost classifier", config.CLASSIFIER_PATH),
        ("Income XGBoost regressor", config.REGRESSOR_PATH),
        ("Local Adzuna dataset (offline mode)", config.LOCAL_JOBS_DATASET_PATH),
        ("Housing / Folium dataset", config.HOUSING_DATA_PATH),
    ]
    return [{"Component": name, "Found": os.path.exists(path), "Path": path} for name, path in checks]
 
 
# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------
st.sidebar.title("🍁 Career Navigator")
page = st.sidebar.radio("Navigation", ["Get My Recommendations", "How This Works"])
 
st.sidebar.divider()
st.sidebar.subheader("Job data source")
job_source_choice = st.sidebar.radio(
    "Where should job listings come from?",
    ["Local dataset (offline, no API key needed)", "Live Adzuna API"],
    index=0 if config.JOB_SOURCE_DEFAULT == "local" else 1,
)
use_local_jobs = job_source_choice.startswith("Local")
 
if use_local_jobs:
    _info = dataset_info()
    st.sidebar.caption(
        f"📦 {_info['total_jobs']} jobs already collected from Adzuna "
        f"({_info['unique_companies']} companies, mostly Toronto area, "
        f"{str(_info['date_min'])[:10]} to {str(_info['date_max'])[:10]}). "
        "No live API call, no key needed."
    )
else:
    st.sidebar.caption("🌐 Calling the live Adzuna API — needs a valid key (see README).")
 
_REQUIRED_COMPONENTS = ("Employment XGBoost classifier", "Income XGBoost regressor")
missing_required = [
    c["Component"] for c in artifact_status()
    if c["Component"] in _REQUIRED_COMPONENTS and not c["Found"]
]
if missing_required:
    st.sidebar.error("Something's missing behind the scenes — see How This Works.")
 
# ---------------------------------------------------------------------------
# Page: How This Works (formerly "Pipeline Status" — technical details live
# here instead of on the main recommendations page)
# ---------------------------------------------------------------------------
if page == "How This Works":
    st.title("How this works")
    st.write(
        "You fill out a short profile once. Behind the scenes, the app runs your "
        "answers through a series of prediction models to work out your likely "
        "occupation, employment odds, and income, then uses those to find and rank "
        "job listings and check housing affordability for your preferred city. "
        "The technical details behind each step are below."
    )
 
    with st.expander("Step-by-step pipeline"):
        st.write(
            "1. **Your profile** is read and matched to a predicted occupation and a "
            "profile fit score.\n"
            "2. **Employment odds** are estimated for that profile.\n"
            "3. **Expected income** is estimated for that profile.\n"
            "4. **Job listings** are pulled from the selected data source and city.\n"
            "5. **Recommendations** are ranked by how well each job matches your "
            "profile and preferences."
        )
        st.code(
            "User Input\n"
            "   -> ProfileEncoder            (predicted occupation, profile fit score)\n"
            "   -> Employment XGBoost        (employment probability)\n"
            "   -> Income XGBoost            (predicted annual income)\n"
            "   -> Adzuna API                (available jobs)\n"
            "   -> Recommendation Engine     (ranked jobs, match scores, explanations)\n"
            "   -> Streamlit Dashboard + Folium interactive map",
            language="text",
        )
 
    with st.expander("System health check (for troubleshooting)"):
        st.caption("Confirms each stage's trained artifacts are present on disk.")
        status_df = pd.DataFrame(artifact_status())
        st.dataframe(status_df, use_container_width=True, hide_index=True)
 
        if os.path.exists(config.LOCAL_JOBS_DATASET_PATH):
            st.divider()
            st.subheader("Local Adzuna dataset (offline mode)")
            _info = dataset_info()
            c1, c2, c3 = st.columns(3)
            c1.metric("Jobs", _info["total_jobs"])
            c2.metric("Companies", _info["unique_companies"])
            c3.metric("Locations", _info["unique_locations"])
            st.caption(
                f"Collected {str(_info['date_min'])[:10]} to {str(_info['date_max'])[:10]}. "
                f"Categories: {', '.join(_info['categories'])}."
            )
 
        if not status_df.loc[status_df["Component"].str.contains("ProfileEncoder"), "Found"].all():
            st.warning(
                "ProfileEncoder trained artifacts (`.keras` / `.joblib` files) were not found in "
                "`artifacts/profile_encoder/`. The app currently falls back to a keyword-based "
                "heuristic for predicted occupation and profile fit score. Re-run "
                "`06_ProfileEncoder_revised.ipynb` and copy its five output files into that folder "
                "to switch to the real neural network — no code changes needed."
            )
    st.stop()
 
# ---------------------------------------------------------------------------
# Page: Get My Recommendations
# ---------------------------------------------------------------------------
st.title("🍁 Find jobs that fit your profile")
st.caption(
    "Fill out your background once. Based on your answers, you'll get: an estimated "
    "employment and income outlook, a list of recommended jobs matched to your profile, "
    "and a housing-affordability check and map for your preferred city."
)
 
try:
    label_encoders = load_label_encoders()
except FileNotFoundError:
    st.error(
        "Could not find the XGBoost label encoders at "
        f"`{config.LABEL_ENCODERS_PATH}`. Make sure the artifacts/xgboost/ folder is populated."
    )
    st.stop()
 
housing_df = load_housing_data()
 
with st.form("profile_form"):
    st.subheader("1. Your background")
    col1, col2, col3 = st.columns(3)
 
    with col1:
        age = st.number_input("Age", min_value=18, max_value=65, value=30, step=1)
        sex = st.selectbox("Sex", list(label_encoders["sex"].classes_))
        admission_category = st.selectbox(
            "Immigration admission category",
            list(label_encoders["admission_category"].classes_),
            help="The immigration stream you were, or expect to be, admitted to Canada under "
            "(e.g. Express Entry, Family Sponsorship, Refugee).",
        )
        world_region = st.selectbox(
            "Region of origin",
            _with_extra_options(
                list(label_encoders["world_region"].classes_),
                ["North America", "South America"],
            ),
        )
        family_size = st.slider(
            "Family size",
            1,
            8,
            2,
            help="The total number of people in your household, including yourself.",
        )
 
    with col2:
        education_level = st.selectbox(
            "Education level", list(label_encoders["education_level"].classes_)
        )
        field_of_study = st.selectbox(
            "Field of study", list(label_encoders["field_of_study"].classes_)
        )
        previous_occupation = st.selectbox(
            "Previous occupation",
            list(label_encoders["previous_occupation"].classes_),
            help="The specific job title or role you held before coming to Canada.",
        )
        occupation_category = st.selectbox(
            "Previous occupation category",
            _with_extra_options(
                list(label_encoders["occupation_category"].classes_), ["Other"]
            ),
            help="The broader field your previous occupation belongs to (e.g. Health, "
            "Trades, Business). Choose \"Other\" if none of the listed categories fit.",
        )
        years_of_experience = st.slider("Years of experience", 0, 40, 5)
 
    with col3:
        teer_category = st.selectbox(
            "TEER category",
            list(label_encoders["teer_category"].classes_),
            help="Canada's Training, Education, Experience and Responsibilities (TEER) "
            "scale, which groups jobs from 0 (management) to 5 (on-the-job training only) "
            "by the skill level they typically require.",
        )
        credential_recognition_status = st.selectbox(
            "Credential recognition status",
            list(label_encoders["credential_recognition_status"].classes_),
            help="Whether your foreign education or professional credentials have been "
            "formally assessed and recognized as equivalent to a Canadian credential.",
        )
        regulated_profession = st.radio(
            "Is your profession regulated in Canada?", ["Yes", "No"], horizontal=True
        )
        speaks_official_language = st.radio(
            "Speak English or French fluently?", ["Yes", "No"], horizontal=True
        )
 
    st.subheader("2. Job search preferences")
    pref_col1, pref_col2, pref_col3 = st.columns(3)
    with pref_col1:
        preferred_city = st.text_input("Preferred city", "Toronto")
    with pref_col2:
        preferred_contract_type = st.selectbox(
            "Preferred contract type", ["permanent", "contract", "temporary"]
        )
    with pref_col3:
        preferred_work_arrangement = st.selectbox(
            "Preferred work arrangement", ["onsite", "hybrid", "remote"]
        )
 
    _COMMON_SKILLS = [
        "Python", "SQL", "Excel", "Communication", "Project Management", "Customer Service",
        "JavaScript", "Java", "Data Analysis", "Accounting", "Sales", "Leadership",
        "Nursing", "Teaching", "Welding", "Carpentry", "Electrical", "Marketing",
        "Bilingual (English/French)", "Machine Learning", "Cloud Computing (AWS/Azure)",
        "Graphic Design", "Writing", "Bookkeeping",
    ]
    selected_skills = st.multiselect(
        "Your skills",
        options=_COMMON_SKILLS,
        default=["Python", "SQL", "Excel", "Communication"],
        help="Pick as many as apply. Don't see a skill you have? Add it below.",
    )
    other_skills_input = st.text_input(
        "Other skills not listed above (comma-separated, optional)"
    )
    keyword_override = st.text_input(
        "Job search keyword (optional — leave blank to auto-fill from your predicted occupation)"
    )
 
    submitted = st.form_submit_button(
        "Find My Matches", type="primary", use_container_width=True
    )
 
#if not submitted:
    #st.info("Fill out the form above and click **Run pipeline** to get recommendations.")
    #st.stop()
 
_all_skills = list(selected_skills) + [
    s.strip() for s in other_skills_input.split(",") if s.strip()
]
 
user_profile = {
    "age": age,
    "sex": sex,
    "admission_category": admission_category,
    "world_region": _safe_model_category(
        "region of origin", world_region, list(label_encoders["world_region"].classes_)
    ),
    "speaks_official_language": 1 if speaks_official_language == "Yes" else 0,
    "education_level": education_level,
    "family_size": family_size,
    "field_of_study": field_of_study,
    "previous_occupation": previous_occupation,
    "occupation_category": _safe_model_category(
        "occupation category",
        occupation_category,
        list(label_encoders["occupation_category"].classes_),
    ),
    "years_of_experience": years_of_experience,
    "teer_category": teer_category,
    "credential_recognition_status": credential_recognition_status,
    "regulated_profession": 1 if regulated_profession == "Yes" else 0,
    # preferences carried alongside, used by the recommendation engine, not the models
    "preferred_city": preferred_city,
    "preferred_contract_type": preferred_contract_type,
    "preferred_work_arrangement": preferred_work_arrangement,
    "skills": _all_skills,
}
 
# ---------------------------------------------------------------------------
# Run the pipeline, stage by stage
# ---------------------------------------------------------------------------
if submitted:
    with st.status("Finding your matches...", expanded=True) as status:
        try:
            status.update(label="Step 1/5 — Reading your profile")
            profile_result = encode_profile(user_profile)
            if profile_result.used_fallback:
                st.warning(
                    "We couldn't reach our full prediction model just now, so this run used a "
                    "simpler backup method for your predicted occupation and profile fit score. "
                    "See **How This Works** in the sidebar for details."
                )
            st.write(
                f"Predicted occupation: **{profile_result.predicted_occupation}** "
                f"(profile fit score: **{profile_result.profile_fit_score:.0%}**)"
            )
 
            status.update(label="Step 2/5 — Estimating your employment odds")
            models = load_xgb_models()
            ei_result = models.run(user_profile)
            st.write(f"Employment probability: **{ei_result.employment_probability:.0%}**")
 
            status.update(label="Step 3/5 — Estimating your income")
            if ei_result.predicted_income is not None:
                st.write(f"Predicted annual income: **${ei_result.predicted_income:,.0f}**")
            else:
                st.write(ei_result.income_skipped_reason)
 
            status.update(label="Step 4/5 — Finding job listings")
            search_keyword = build_search_keyword(
                profile_result.predicted_occupation, previous_occupation, keyword_override
            )
            if use_local_jobs:
                jobs_df = cached_get_jobs_local(search_keyword, preferred_city)
                source_label = "local dataset (offline)"
            else:
                jobs_df = cached_get_jobs_live(search_keyword, preferred_city)
                source_label = "live Adzuna API"
            st.write(
                f"Search: `{search_keyword}` in `{preferred_city}` → **{len(jobs_df)}** jobs "
                f"from the {source_label}."
            )
            if use_local_jobs:
                st.caption(
                    "ℹ️ This dataset was collected from one Adzuna search (mostly Toronto-area, "
                    "IT/data/analyst-leaning roles) and doesn't refresh — results for very different "
                    "occupations or other cities fall back to the closest available matches rather "
                    "than nothing. Switch to **Live Adzuna API** in the sidebar for a real-time, "
                    "location-specific search."
                )
 
            status.update(label="Step 5/5 — Ranking your recommendations")
            predictions = {
                "employment_probability": ei_result.employment_probability,
                "predicted_income": ei_result.predicted_income,
                "predicted_occupation": profile_result.predicted_occupation,
                "profile_fit_score": profile_result.profile_fit_score,
            }
            if jobs_df.empty:
                ranked_jobs = pd.DataFrame()
                status.update(label="No jobs found for this search", state="error")
            else:
                ranked_jobs = recommend_jobs(user_profile, predictions, jobs_df, top_n=10)
                status.update(label="Matches ready!", state="complete")
 
            # Save everything the results section needs
            st.session_state["pipeline_results"] = {
                "profile_result": profile_result,
                "ei_result": ei_result,
                "ranked_jobs": ranked_jobs,
                "preferred_city": preferred_city,
                "housing_df": housing_df,
            }
        except Exception as e:
            status.update(label=f"Pipeline failed: {e}", state="error")
            st.exception(e)
            st.stop()
 
# ---------------------------------------------------------------------------
# Results dashboard — always render from session_state if present
# ---------------------------------------------------------------------------
if "pipeline_results" not in st.session_state:
    st.info("Fill out the form above and click **Find My Matches** to get recommendations.")
    st.stop()
 
results = st.session_state["pipeline_results"]
profile_result = results["profile_result"]
ei_result = results["ei_result"]
ranked_jobs = results["ranked_jobs"]
preferred_city = results["preferred_city"]
housing_df = results["housing_df"]
 
st.divider()
st.header("Your results")
 
# ---------------------------------------------------------------------------
# Key metric cards — predicted occupation & income first (what you get),
# followed by the supporting scores that explain how confident those are.
# ---------------------------------------------------------------------------
st.subheader("Your career snapshot")
 
headline1, headline2 = st.columns(2)
headline1.metric("Predicted occupation", profile_result.predicted_occupation)
if ei_result.predicted_income is not None:
    headline2.metric("Expected annual income", f"${ei_result.predicted_income:,.0f}")
else:
    headline2.metric("Expected annual income", "N/A")
 
support1, support2 = st.columns(2)
support1.metric(
    "Profile fit score",
    f"{profile_result.profile_fit_score:.0%}",
    help="How closely your background matches the training data for your predicted "
    "occupation. Higher means the model is more confident in that match.",
)
support2.metric(
    "Employment probability",
    f"{ei_result.employment_probability:.0%}",
    help="Our model's estimate of the likelihood you'll be employed in Canada within "
    "the first year, based on profiles with a similar background.",
)
 
st.divider()
 
# ---------------------------------------------------------------------------
# Ranked jobs table
# ---------------------------------------------------------------------------
st.subheader("Recommended jobs for you")
 
if ranked_jobs.empty:
    st.info("No job recommendations to display for this search.")
else:
    def first_existing(df, candidates):
        for c in candidates:
            if c in df.columns:
                return c
        return None
 
    title_col = first_existing(ranked_jobs, ["title", "job_title"])
    company_col = first_existing(ranked_jobs, ["company", "company_name"])
    salary_col = first_existing(ranked_jobs, ["salary", "salary_avg", "predicted_salary"])
    salary_min_col = first_existing(ranked_jobs, ["salary_min"])
    salary_max_col = first_existing(ranked_jobs, ["salary_max"])
    link_col = first_existing(ranked_jobs, ["redirect_url", "url", "link", "job_url"])
 
    table = pd.DataFrame()
    table["Title"] = ranked_jobs[title_col] if title_col else ""
    table["Company"] = ranked_jobs[company_col] if company_col else ""
    table["Match score"] = ranked_jobs["match_score"]
 
    def format_salary_range(row):
        smin = row.get(salary_min_col) if salary_min_col else None
        smax = row.get(salary_max_col) if salary_max_col else None
        smin = None if pd.isna(smin) or smin in (0, "0") else float(smin)
        smax = None if pd.isna(smax) or smax in (0, "0") else float(smax)
        if smin is None and smax is None:
            return "Salary not available"
        if smin is not None and smax is not None:
            return f"${smin:,.0f} – ${smax:,.0f}"
        return f"${(smin or smax):,.0f}"
 
    if salary_col:
        table["Salary"] = ranked_jobs[salary_col].apply(
            lambda v: "Salary not available" if pd.isna(v) or v in (0, "0") else f"${float(v):,.0f}"
        )
    elif salary_min_col and salary_max_col:
        table["Salary"] = ranked_jobs.apply(format_salary_range, axis=1)
    else:
        table["Salary"] = "Salary not available"
 
    table["Link"] = ranked_jobs[link_col] if link_col else None
 
    table = table.sort_values("Match score", ascending=False)
 
    st.dataframe(
        table,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match score": st.column_config.ProgressColumn(
                "Match score",
                min_value=0,
                max_value=100,
                format="%.0f%%",
                help="How well this job posting fits your profile and preferences, "
                "combining your predicted occupation, skills, and job search preferences.",
            ),
            "Link": st.column_config.LinkColumn("Job posting", display_text="View job ↗"),
        },
    )
 
    # ---------------------------------------------------------------------------
    # Match score chart
    # ---------------------------------------------------------------------------
    st.subheader("Match score comparison")
    chart_df = table.sort_values("Match score", ascending=True)
    fig = px.bar(
        chart_df,
        x="Match score",
        y="Title",
        orientation="h",
        color="Match score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        labels={"Match score": "Match score (%)", "Title": ""},
    )
    fig.update_layout(showlegend=False, height=max(300, 40 * len(chart_df)))
    st.plotly_chart(fig, use_container_width=True)
 
st.divider()
 
# ---------------------------------------------------------------------------
# Housing affordability
# ---------------------------------------------------------------------------
st.subheader("Can you afford to live there?")
 
if housing_df is None:
    st.info("Housing dataset not found — affordability can't be estimated for this run.")
else:
    city_housing_afford = housing_df[
        housing_df["city"].str.lower() == preferred_city.strip().lower()
    ]
    if city_housing_afford.empty:
        st.info(f"No housing data on file for **{preferred_city}** — affordability can't be estimated.")
    elif ei_result.predicted_income is None:
        st.info(
            "Predicted income wasn't available for this profile "
            f"({ei_result.income_skipped_reason}), so affordability can't be calculated."
        )
    else:
        avg_rent = city_housing_afford["monthly_rent"].dropna().astype(float).mean()
        monthly_income = ei_result.predicted_income / 12
        rent_to_income = (avg_rent / monthly_income) if monthly_income else None
 
        a1, a2, a3 = st.columns(3)
        a1.metric(f"Avg. monthly rent — {preferred_city.title()}", f"${avg_rent:,.0f}")
        a2.metric("Predicted monthly income", f"${monthly_income:,.0f}")
        if rent_to_income is not None:
            a3.metric("Rent-to-income ratio", f"{rent_to_income:.0%}")
 
        if rent_to_income is not None:
            if rent_to_income <= 0.30:
                st.success(
                    f"🟢 **Affordable** — rent is about {rent_to_income:.0%} of predicted monthly "
                    "income (the common affordability benchmark is 30% or less)."
                )
            elif rent_to_income <= 0.50:
                st.warning(
                    f"🟠 **Stretched** — rent is about {rent_to_income:.0%} of predicted monthly "
                    "income, above the 30% affordability benchmark."
                )
            else:
                st.error(
                    f"🔴 **Unaffordable at this income level** — rent is about {rent_to_income:.0%} "
                    "of predicted monthly income, well above the 30% affordability benchmark."
                )
 
        st.caption(
            f"Based on {len(city_housing_afford)} housing data point(s) for {preferred_city.title()} "
            "in `housing_geocoded.csv`. The 30% / 50% thresholds follow the standard "
            "shelter-cost-to-income affordability rule of thumb, and this is an estimate, not "
            "financial advice."
        )
 
st.divider()
 
# ---------------------------------------------------------------------------
# Folium interactive map
# ---------------------------------------------------------------------------
st.subheader("Job locations")
 
geo_jobs = ranked_jobs.dropna(subset=["latitude", "longitude"]) if "latitude" in ranked_jobs.columns else pd.DataFrame()
 
if geo_jobs.empty:
    st.info("None of the returned jobs had coordinates to plot on the map.")
else:
    center_lat = geo_jobs["latitude"].astype(float).mean()
    center_lon = geo_jobs["longitude"].astype(float).mean()
    job_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)
 
    for _, job in geo_jobs.iterrows():
        score = job["match_score"]
        color = "green" if score >= 70 else "orange" if score >= 40 else "red"
        folium.Marker(
            location=[float(job["latitude"]), float(job["longitude"])],
            popup=folium.Popup(
                f"<b>{job['title']}</b><br>{job.get('company', '')}<br>"
                f"Match score: {score:.1f}%",
                max_width=250,
            ),
            tooltip=job["title"],
            icon=folium.Icon(color=color),
        ).add_to(job_map)
 
    if housing_df is not None:
        city_housing = housing_df[
            housing_df["city"].str.lower() == preferred_city.strip().lower()
        ]
        for _, row in city_housing.dropna(subset=["latitude", "longitude"]).iterrows():
            folium.Marker(
                location=[float(row["latitude"]), float(row["longitude"])],
                popup=folium.Popup(
                    f"<b>{row.get('neighbourhood', 'Housing data point')}</b><br>"
                    f"Avg. monthly rent: ${row.get('monthly_rent', 'N/A')}",
                    max_width=250,
                ),
                tooltip="Housing data point",
                icon=folium.Icon(color="blue", icon="home", prefix="fa"),
            ).add_to(job_map)
 
    st_folium(job_map, use_container_width=True, height=500)
    st.caption(
        "🟢 Strong match (≥70%)   🟠 Moderate match (40–70%)   🔴 Weaker match (<40%)   "
        "🔵 Nearby housing / rent data point"
    )
