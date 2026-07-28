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
page = st.sidebar.radio("Navigation", ["Get My Recommendations", "Pipeline Status"])

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

st.sidebar.divider()
st.sidebar.caption(
    "User Input → ProfileEncoder → Employment XGBoost → Income XGBoost "
    "→ Adzuna API → Recommendation Engine → Dashboard + Map"
)

_REQUIRED_COMPONENTS = ("Employment XGBoost classifier", "Income XGBoost regressor")
missing_required = [
    c["Component"] for c in artifact_status()
    if c["Component"] in _REQUIRED_COMPONENTS and not c["Found"]
]
if missing_required:
    st.sidebar.error("Missing required XGBoost artifacts — see Pipeline Status.")

# ---------------------------------------------------------------------------
# Page: Pipeline Status
# ---------------------------------------------------------------------------
if page == "Pipeline Status":
    st.title("Pipeline status")
    st.caption("Health check for every stage's trained artifacts.")

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

    st.divider()
    st.subheader("Pipeline diagram")
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
    st.stop()

# ---------------------------------------------------------------------------
# Page: Get My Recommendations
# ---------------------------------------------------------------------------
st.title("Find jobs that fit your profile")
st.caption(
    "Fill out your background once — the app runs it through every model in the pipeline "
    "and returns ranked, explained job recommendations."
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
        age = st.slider("Age", 18, 65, 30)
        sex = st.selectbox("Sex", list(label_encoders["sex"].classes_))
        admission_category = st.selectbox(
            "Admission category", list(label_encoders["admission_category"].classes_)
        )
        world_region = st.selectbox(
            "Region of origin", list(label_encoders["world_region"].classes_)
        )
        family_size = st.slider("Family size", 1, 8, 2)

    with col2:
        education_level = st.selectbox(
            "Education level", list(label_encoders["education_level"].classes_)
        )
        field_of_study = st.selectbox(
            "Field of study", list(label_encoders["field_of_study"].classes_)
        )
        previous_occupation = st.selectbox(
            "Previous occupation", list(label_encoders["previous_occupation"].classes_)
        )
        occupation_category = st.selectbox(
            "Occupation category", list(label_encoders["occupation_category"].classes_)
        )
        years_of_experience = st.slider("Years of experience", 0, 40, 5)

    with col3:
        teer_category = st.selectbox(
            "TEER category", list(label_encoders["teer_category"].classes_)
        )
        credential_recognition_status = st.selectbox(
            "Credential recognition status",
            list(label_encoders["credential_recognition_status"].classes_),
        )
        regulated_profession = st.radio(
            "Is your profession regulated in Canada?", ["No", "Yes"], horizontal=True
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

    skills_input = st.text_input(
        "Your skills (comma-separated)", "Python, SQL, Excel, Communication"
    )
    keyword_override = st.text_input(
        "Job search keyword (optional — leave blank to auto-fill from your predicted occupation)"
    )

    submitted = st.form_submit_button("Run pipeline", type="primary", use_container_width=True)

if not submitted:
    st.info("Fill out the form above and click **Run pipeline** to get recommendations.")
    st.stop()

user_profile = {
    "age": age,
    "sex": sex,
    "admission_category": admission_category,
    "world_region": world_region,
    "speaks_official_language": 1 if speaks_official_language == "Yes" else 0,
    "education_level": education_level,
    "family_size": family_size,
    "field_of_study": field_of_study,
    "previous_occupation": previous_occupation,
    "occupation_category": occupation_category,
    "years_of_experience": years_of_experience,
    "teer_category": teer_category,
    "credential_recognition_status": credential_recognition_status,
    "regulated_profession": 1 if regulated_profession == "Yes" else 0,
    # preferences carried alongside, used by the recommendation engine, not the models
    "preferred_city": preferred_city,
    "preferred_contract_type": preferred_contract_type,
    "preferred_work_arrangement": preferred_work_arrangement,
    "skills": [s.strip() for s in skills_input.split(",") if s.strip()],
}

# ---------------------------------------------------------------------------
# Run the pipeline, stage by stage
# ---------------------------------------------------------------------------
if submitted:
    with st.status("Running pipeline...", expanded=True) as status:
        status.update(label="Stage 1/5 — ProfileEncoder")
        profile_result = encode_profile(user_profile)
        # ... (all your existing stage code, unchanged) ...

        status.update(label="Stage 5/5 — Recommendation Engine")
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
            status.update(label="Pipeline complete", state="complete")

    # Save everything the results section needs
    st.session_state["pipeline_results"] = {
        "profile_result": profile_result,
        "ei_result": ei_result,
        "ranked_jobs": ranked_jobs,
        "preferred_city": preferred_city,
        "housing_df": housing_df,
    }

# ---------------------------------------------------------------------------
# Results dashboard — always render from session_state if present
# ---------------------------------------------------------------------------
if "pipeline_results" not in st.session_state:
    st.info("Fill out the form above and click **Run pipeline** to get recommendations.")
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
            folium.CircleMarker(
                location=[float(row["latitude"]), float(row["longitude"])],
                radius=5,
                color="blue",
                fill=True,
                fill_opacity=0.4,
                popup=(
                    f"{row.get('neighbourhood', '')}<br>"
                    f"Avg. monthly rent: ${row.get('monthly_rent', 'N/A')}"
                ),
                tooltip="Housing data point",
            ).add_to(job_map)

    st_folium(job_map, use_container_width=True, height=500)
    st.caption(
        "🟢 Strong match (≥70%)   🟠 Moderate match (40–70%)   🔴 Weaker match (<40%)   "
        "🔵 Nearby housing / rent data point"
    )
