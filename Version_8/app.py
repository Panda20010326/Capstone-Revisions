```python
"""
app.py
------
Single Streamlit interface that runs the full Capstone Group 4 pipeline:

    User Input
        -> ProfileEncoder              (predicted occupation, profile fit score)
        -> Employment XGBoost          (employment probability)
        -> Income XGBoost              (predicted annual income)
        -> Expanded Adzuna dataset     (available jobs across Ontario)
        -> Recommendation Engine       (ranked jobs, match scores, explanations)
        -> Dashboard + Folium map

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import os
from pathlib import Path

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
from pipeline.job_source import dataset_info
from pipeline.karthika_recommendation import (
    rank_jobs as karthika_rank_jobs,
    build_housing_recommendations as karthika_build_housing_recommendations,
    prepare_housing_data,
)


# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Newcomer Career Navigator — Ontario",
    page_icon=str(config.LOGO_PATH) if config.LOGO_PATH.exists() else "🍁",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Expanded local Adzuna dataset
# ---------------------------------------------------------------------------
#
# The expanded dataset created from adzuna_more_ontario_cities.py contains
# additional Ontario cities such as:
#
#   Mississauga, Ottawa, Hamilton, Brampton, Kitchener,
#   London, Windsor, Waterloo, Markham
#
# We look for the expanded file first, then fall back to the configured
# dataset path.
#
# IMPORTANT:
# We do NOT run adzuna_more_ontario_cities.py inside Streamlit.
# That script is a data-collection script and should be run separately
# whenever the dataset needs to be refreshed.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent

EXPANDED_DATASET_CANDIDATES = [
    PROJECT_ROOT / "processed_adzuna_jobs_expanded.csv",
    PROJECT_ROOT / "processed_adzuna_jobs (2).csv",
    PROJECT_ROOT / "processed_adzuna_jobs.csv",
]

# Also support a configured path if the project uses one.
try:
    CONFIGURED_DATASET_PATH = Path(config.LOCAL_JOBS_DATASET_PATH)
except Exception:
    CONFIGURED_DATASET_PATH = None


def _find_local_jobs_dataset() -> Path | None:
    """Find the best available local Adzuna dataset."""

    candidates = list(EXPANDED_DATASET_CANDIDATES)

    if CONFIGURED_DATASET_PATH is not None:
        candidates.append(CONFIGURED_DATASET_PATH)

    # Remove duplicate paths while preserving order.
    seen = set()
    unique_candidates = []

    for path in candidates:
        path = Path(path)
        if path not in seen:
            unique_candidates.append(path)
            seen.add(path)

    for path in unique_candidates:
        if path.exists():
            return path

    return None


LOCAL_JOBS_DATASET = _find_local_jobs_dataset()


# ---------------------------------------------------------------------------
# City matching helpers
# ---------------------------------------------------------------------------

CITY_ALIASES = {
    "toronto": [
        "toronto",
        "etobicoke",
        "scarborough",
        "north york",
        "east york",
    ],
    "mississauga": [
        "mississauga",
    ],
    "brampton": [
        "brampton",
    ],
    "ottawa": [
        "ottawa",
        "billings bridge",
    ],
    "hamilton": [
        "hamilton",
        "stoney creek",
        "ancaster",
        "dundas",
        "hannon",
    ],
    "kitchener": [
        "kitchener",
    ],
    "waterloo": [
        "waterloo",
        "wellesley",
        "wilmot",
    ],
    "london": [
        "london",
    ],
    "windsor": [
        "windsor",
    ],
    "markham": [
        "markham",
    ],
    "cambridge": [
        "cambridge",
    ],
    "vaughan": [
        "vaughan",
    ],
    "burlington": [
        "burlington",
    ],
    "oakville": [
        "oakville",
    ],
    "kingston": [
        "kingston",
    ],
    "sudbury": [
        "sudbury",
        "greater sudbury",
    ],
}


def _normalize_text(value) -> str:
    """Normalize text for robust city matching."""

    if value is None:
        return ""

    return (
        str(value)
        .strip()
        .lower()
        .replace("'", "")
        .replace(",", " ")
        .replace("-", " ")
    )


def _city_terms(city: str) -> list[str]:
    """Return recognized search terms for a requested city."""

    normalized = _normalize_text(city)

    if normalized in CITY_ALIASES:
        return CITY_ALIASES[normalized]

    # Allow the user to enter "Hamilton, Ontario", etc.
    for canonical, aliases in CITY_ALIASES.items():
        if normalized.startswith(canonical) or canonical in normalized:
            return aliases

    # Generic fallback: use the first meaningful part of the city name.
    first_part = normalized.split()[0] if normalized else ""
    return [first_part] if first_part else []


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


@st.cache_data(show_spinner=False)
def load_expanded_local_jobs() -> pd.DataFrame:
    """
    Load the expanded local Adzuna dataset.

    The dataset is loaded once and cached by Streamlit.
    """

    dataset_path = _find_local_jobs_dataset()

    if dataset_path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(dataset_path)
    except Exception:
        return pd.DataFrame()

    # Normalize column names.
    df.columns = [str(c).strip() for c in df.columns]

    # Ensure important columns exist.
    for column in [
        "id",
        "title",
        "company",
        "location",
        "salary_min",
        "salary_max",
        "latitude",
        "longitude",
        "description",
        "category",
        "category_tag",
        "contract_type",
        "contract_time",
        "created",
        "url",
    ]:
        if column not in df.columns:
            df[column] = None

    return df


@st.cache_data(show_spinner=False)
def cached_get_jobs_live(keyword: str, city: str):
    return get_jobs_multi_page(keyword, city)


@st.cache_data(show_spinner=False)
def cached_get_jobs_local(keyword: str, city: str):
    """
    Search the expanded local dataset.

    Matching is intentionally flexible because Adzuna location values can
    look like:

        Hamilton, Hamilton region
        Hamilton region, Ontario
        Kitchener, Waterloo region
        Waterloo region, Ontario

    The user's selected city is therefore matched against both the city
    name and known regional aliases.
    """

    df = load_expanded_local_jobs()

    if df.empty:
        return pd.DataFrame()

    working = df.copy()

    location_series = working["location"].fillna("").astype(str).map(_normalize_text)

    terms = _city_terms(city)

    if not terms:
        return pd.DataFrame()

    city_mask = pd.Series(False, index=working.index)

    for term in terms:
        term_normalized = _normalize_text(term)

        if term_normalized:
            city_mask = city_mask | location_series.str.contains(
                term_normalized,
                regex=False,
                na=False,
            )

    city_jobs = working.loc[city_mask].copy()

    if city_jobs.empty:
        return pd.DataFrame()

    # -----------------------------------------------------------------------
    # Keyword matching
    # -----------------------------------------------------------------------
    #
    # We first try to identify jobs related to the requested keyword.
    # If there aren't enough keyword matches, return the city's jobs rather
    # than returning zero jobs.
    #
    # This gives the recommendation engine more opportunities to rank jobs.
    # -----------------------------------------------------------------------

    keyword = str(keyword or "").strip()

    if keyword:
        searchable_columns = [
            "title",
            "description",
            "category",
            "category_tag",
        ]

        text_parts = []

        for column in searchable_columns:
            if column in city_jobs.columns:
                text_parts.append(
                    city_jobs[column].fillna("").astype(str)
                )

        if text_parts:
            combined_text = text_parts[0]

            for part in text_parts[1:]:
                combined_text = combined_text + " " + part

            keyword_terms = [
                term.strip().lower()
                for term in keyword.replace("/", " ").split()
                if len(term.strip()) >= 3
            ]

            if keyword_terms:
                keyword_mask = pd.Series(False, index=city_jobs.index)

                for term in keyword_terms:
                    keyword_mask = keyword_mask | combined_text.str.lower().str.contains(
                        term,
                        regex=False,
                        na=False,
                    )

                keyword_jobs = city_jobs.loc[keyword_mask].copy()

                # Use keyword matches when available.
                if not keyword_jobs.empty:
                    city_jobs = keyword_jobs

    # Remove duplicate postings.
    if "id" in city_jobs.columns:
        city_jobs = city_jobs.drop_duplicates(subset="id")

    return city_jobs.reset_index(drop=True)


def local_dataset_summary(df: pd.DataFrame) -> dict:
    """Return summary statistics for the expanded local dataset."""

    if df.empty:
        return {
            "total_jobs": 0,
            "unique_companies": 0,
            "unique_locations": 0,
            "locations": [],
            "dataset_path": str(LOCAL_JOBS_DATASET) if LOCAL_JOBS_DATASET else "Not found",
        }

    companies = (
        df["company"].dropna().astype(str).nunique()
        if "company" in df.columns
        else 0
    )

    locations = (
        df["location"].dropna().astype(str).nunique()
        if "location" in df.columns
        else 0
    )

    return {
        "total_jobs": len(df),
        "unique_companies": companies,
        "unique_locations": locations,
        "locations": sorted(
            df["location"].dropna().astype(str).unique().tolist()
        )
        if "location" in df.columns
        else [],
        "dataset_path": str(LOCAL_JOBS_DATASET) if LOCAL_JOBS_DATASET else "Not found",
    }


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------

def _with_extra_options(
    base_options: list[str],
    extra_options: list[str],
) -> list[str]:
    """Return base_options with extra options appended without duplicates."""

    seen = {opt.strip().lower() for opt in base_options}
    widened = list(base_options)

    for opt in extra_options:
        if opt.strip().lower() not in seen:
            widened.append(opt)
            seen.add(opt.strip().lower())

    return widened


def _safe_model_category(
    field_label: str,
    selected_value: str,
    known_classes: list[str],
) -> str:
    """
    Map a selected dropdown value back onto a class the trained models
    understand.
    """

    if selected_value in known_classes:
        return selected_value

    fallback = "Other" if "Other" in known_classes else known_classes[0]

    st.caption(
        f'ℹ️ "{selected_value}" isn\'t part of the trained '
        f'{field_label} categories yet, so predictions for this run use '
        f'the closest available category ("{fallback}").'
    )

    return fallback


def artifact_status() -> list[dict]:
    checks = [
        ("ProfileEncoder network", config.PROFILE_ENCODER_MODEL_PATH),
        ("ProfileEncoder multitask model", config.MULTITASK_MODEL_PATH),
        ("ProfileEncoder preprocessor", config.PREPROCESSOR_PATH),
        ("Employment XGBoost classifier", config.CLASSIFIER_PATH),
        ("Income XGBoost regressor", config.REGRESSOR_PATH),
        ("Local Adzuna dataset", str(LOCAL_JOBS_DATASET) if LOCAL_JOBS_DATASET else ""),
        ("Housing / Folium dataset", config.HOUSING_DATA_PATH),
    ]

    return [
        {
            "Component": name,
            "Found": bool(path) and os.path.exists(path),
            "Path": path,
        }
        for name, path in checks
    ]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

if config.LOGO_PATH.exists():
    st.sidebar.image(str(config.LOGO_PATH), width=190)
else:
    st.sidebar.title("🍁 Career Navigator")

page = st.sidebar.radio(
    "Navigation",
    ["Get My Recommendations", "How This Works"],
)

st.sidebar.divider()

st.sidebar.subheader("Job data source")

job_source_choice = st.sidebar.radio(
    "Where should job listings come from?",
    [
        "Expanded local dataset (offline, no API key needed)",
        "Live Adzuna API",
    ],
    index=0 if config.JOB_SOURCE_DEFAULT == "local" else 1,
)

use_local_jobs = job_source_choice.startswith("Expanded local")


if use_local_jobs:
    local_jobs_df = load_expanded_local_jobs()
    local_info = local_dataset_summary(local_jobs_df)

    if local_jobs_df.empty:
        st.sidebar.error(
            "Expanded local job dataset was not found."
        )
        st.sidebar.caption(
            "Expected processed_adzuna_jobs_expanded.csv or "
            "processed_adzuna_jobs (2).csv."
        )
    else:
        st.sidebar.success(
            f"📦 {local_info['total_jobs']:,} jobs available"
        )

        st.sidebar.caption(
            f"{local_info['unique_companies']:,} companies • "
            f"{local_info['unique_locations']:,} locations"
        )

        st.sidebar.caption(
            "Includes expanded Ontario city coverage from the Adzuna "
            "collection script."
        )

else:
    st.sidebar.caption(
        "🌐 Calling the live Adzuna API — needs a valid key."
    )


_REQUIRED_COMPONENTS = (
    "Employment XGBoost classifier",
    "Income XGBoost regressor",
)

missing_required = [
    c["Component"]
    for c in artifact_status()
    if c["Component"] in _REQUIRED_COMPONENTS and not c["Found"]
]

if missing_required:
    st.sidebar.error(
        "Something's missing behind the scenes — see How This Works."
    )


# ---------------------------------------------------------------------------
# Page: How This Works
# ---------------------------------------------------------------------------

if page == "How This Works":

    st.title("How this works")

    st.write(
        "You fill out a short profile once. Behind the scenes, the app "
        "runs your answers through prediction models to work out your "
        "likely occupation, employment odds, and income, then uses the "
        "expanded Ontario job dataset to find and rank job listings and "
        "check housing affordability for your preferred city."
    )

    with st.expander("Step-by-step pipeline"):

        st.write(
            "1. **Your profile** is read and matched to a predicted "
            "occupation and a profile fit score.\n"
            "2. **Employment odds** are estimated for that profile.\n"
            "3. **Expected income** is estimated for that profile.\n"
            "4. **Job listings** are pulled from the expanded local Adzuna "
            "dataset or live Adzuna API.\n"
            "5. **Recommendations** are ranked by how well each job "
            "matches your profile and preferences.\n"
            "6. **Job and housing locations** are displayed on the "
            "interactive map."
        )

        st.code(
            "User Input\n"
            "   -> ProfileEncoder            (predicted occupation, profile fit score)\n"
            "   -> Employment XGBoost        (employment probability)\n"
            "   -> Income XGBoost            (predicted annual income)\n"
            "   -> Expanded Adzuna Dataset   (Ontario job listings)\n"
            "   -> Recommendation Engine     (ranked jobs, match scores)\n"
            "   -> Streamlit Dashboard + Folium interactive map",
            language="text",
        )

    with st.expander("Expanded Ontario job coverage"):

        st.write(
            "The expanded local dataset includes additional job postings "
            "for major Ontario cities such as:"
        )

        st.write(
            "Mississauga • Ottawa • Hamilton • Brampton • Kitchener • "
            "London • Windsor • Waterloo • Markham"
        )

        if not local_jobs_df.empty:
            st.metric(
                "Local job postings",
                f"{len(local_jobs_df):,}",
            )

            if "location" in local_jobs_df.columns:
                city_counts = (
                    local_jobs_df["location"]
                    .fillna("Unknown")
                    .astype(str)
                    .value_counts()
                    .head(20)
                    .rename_axis("Location")
                    .reset_index(name="Jobs")
                )

                st.dataframe(
                    city_counts,
                    use_container_width=True,
                    hide_index=True,
                )

    with st.expander("System health check"):

        st.caption(
            "Confirms each stage's trained artifacts and datasets are "
            "present on disk."
        )

        status_df = pd.DataFrame(artifact_status())

        st.dataframe(
            status_df,
            use_container_width=True,
            hide_index=True,
        )

        if not status_df.empty:

            profile_rows = status_df[
                status_df["Component"].str.contains(
                    "ProfileEncoder",
                    na=False,
                )
            ]

            if not profile_rows.empty and not profile_rows["Found"].all():

                st.warning(
                    "ProfileEncoder trained artifacts were not found. "
                    "The app can fall back to the simpler keyword-based "
                    "method for predicted occupation and profile fit."
                )

    st.stop()


# ---------------------------------------------------------------------------
# Page: Get My Recommendations
# ---------------------------------------------------------------------------

st.title("🍁 Find jobs that fit your profile")

st.caption(
    "Fill out your background once. Based on your answers, you'll get: "
    "an estimated employment and income outlook, a list of recommended "
    "jobs matched to your profile, and a housing-affordability check "
    "and map for your preferred city."
)


# ---------------------------------------------------------------------------
# Load model encoders
# ---------------------------------------------------------------------------

try:

    label_encoders = load_label_encoders()

except FileNotFoundError:

    st.error(
        "Could not find the XGBoost label encoders at "
        f"`{config.LABEL_ENCODERS_PATH}`. "
        "Make sure the artifacts/xgboost/ folder is populated."
    )

    st.stop()


housing_df = load_housing_data()


# ---------------------------------------------------------------------------
# Profile form
# ---------------------------------------------------------------------------

with st.form("profile_form"):

    st.subheader("1. Your background")

    col1, col2, col3 = st.columns(3)

    with col1:

        age = st.number_input(
            "Age",
            min_value=18,
            max_value=65,
            value=30,
            step=1,
        )

        sex = st.selectbox(
            "Sex",
            list(label_encoders["sex"].classes_),
        )

        admission_category = st.selectbox(
            "Immigration admission category",
            list(label_encoders["admission_category"].classes_),
            help=(
                "The immigration stream you were, or expect to be, "
                "admitted to Canada under."
            ),
        )

        world_region = st.selectbox(
            "Region of origin",
            _with_extra_options(
                list(label_encoders["world_region"].classes_),
                [
                    "North America",
                    "Central America",
                    "South America",
                    "Caribbean",
                    "Western Europe",
                    "Eastern Europe",
                    "Northern Europe",
                    "Southern Europe",
                    "North Africa",
                    "Sub-Saharan Africa",
                    "Middle East",
                    "Central Asia",
                    "South Asia",
                    "Southeast Asia",
                    "East Asia",
                    "Oceania",
                ],
            ),
            help=(
                "Pick the region that best matches where you're from."
            ),
        )

        family_size = st.slider(
            "Family size",
            1,
            8,
            2,
            help=(
                "The total number of people in your household, "
                "including yourself."
            ),
        )

    with col2:

        education_level = st.selectbox(
            "Education level",
            list(label_encoders["education_level"].classes_),
        )

        field_of_study = st.selectbox(
            "Field of study",
            list(label_encoders["field_of_study"].classes_),
        )

        previous_occupation = st.selectbox(
            "Previous occupation",
            list(label_encoders["previous_occupation"].classes_),
            help=(
                "The specific job title or role you held before "
                "coming to Canada."
            ),
        )

        occupation_category = st.selectbox(
            "Previous occupation category",
            _with_extra_options(
                list(label_encoders["occupation_category"].classes_),
                ["Other"],
            ),
            help=(
                "The broader field your previous occupation belongs to."
            ),
        )

        years_of_experience = st.slider(
            "Years of experience",
            0,
            40,
            5,
        )

    with col3:

        teer_category = st.selectbox(
            "TEER category",
            list(label_encoders["teer_category"].classes_),
            help=(
                "Canada's Training, Education, Experience and "
                "Responsibilities (TEER) scale."
            ),
        )

        credential_recognition_status = st.selectbox(
            "Credential recognition status",
            list(label_encoders["credential_recognition_status"].classes_),
            help=(
                "Whether your foreign education or professional "
                "credentials have been formally assessed and recognized."
            ),
        )

        regulated_profession = st.radio(
            "Is your profession regulated in Canada?",
            ["Yes", "No"],
            horizontal=True,
        )

        speaks_official_language = st.radio(
            "Speak English or French fluently?",
            ["Yes", "No"],
            horizontal=True,
        )

    st.subheader("2. Job search preferences")

    pref_col1, pref_col2, pref_col3 = st.columns(3)

    with pref_col1:

        preferred_city = st.text_input(
            "Preferred city",
            "Toronto",
            help=(
                "Examples: Toronto, Mississauga, Ottawa, Hamilton, "
                "Brampton, Kitchener, London, Windsor, Waterloo, Markham."
            ),
        )

    with pref_col2:

        preferred_contract_type = st.selectbox(
            "Preferred contract type",
            [
                "permanent",
                "contract",
                "temporary",
            ],
        )

    with pref_col3:

        preferred_work_arrangement = st.selectbox(
            "Preferred work arrangement",
            [
                "onsite",
                "hybrid",
                "remote",
            ],
        )

    _COMMON_SKILLS = [
        "Python",
        "SQL",
        "Excel",
        "Communication",
        "Project Management",
        "Customer Service",
        "JavaScript",
        "Java",
        "Data Analysis",
        "Accounting",
        "Sales",
        "Leadership",
        "Nursing",
        "Teaching",
        "Welding",
        "Carpentry",
        "Electrical",
        "Marketing",
        "Bilingual (English/French)",
        "Machine Learning",
        "Cloud Computing (AWS/Azure)",
        "Graphic Design",
        "Writing",
        "Bookkeeping",
    ]

    selected_skills = st.multiselect(
        "Your skills",
        options=_COMMON_SKILLS,
        default=[
            "Python",
            "SQL",
            "Excel",
            "Communication",
        ],
        help=(
            "Pick as many as apply. Don't see a skill you have? "
            "Add it below."
        ),
    )

    other_skills_input = st.text_input(
        "Other skills not listed above (comma-separated, optional)"
    )

    keyword_override = st.text_input(
        "Job search keyword (optional — leave blank to auto-fill "
        "from your predicted occupation)"
    )

    submitted = st.form_submit_button(
        "Find My Matches",
        type="primary",
        use_container_width=True,
    )


# ---------------------------------------------------------------------------
# Prepare user profile
# ---------------------------------------------------------------------------

_all_skills = list(selected_skills) + [
    s.strip()
    for s in other_skills_input.split(",")
    if s.strip()
]


user_profile = {
    "age": age,
    "sex": sex,
    "admission_category": admission_category,
    "world_region": _safe_model_category(
        "region of origin",
        world_region,
        list(label_encoders["world_region"].classes_),
    ),
    "speaks_official_language": (
        1 if speaks_official_language == "Yes" else 0
    ),
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
    "regulated_profession": (
        1 if regulated_profession == "Yes" else 0
    ),
    "preferred_city": preferred_city,
    "preferred_contract_type": preferred_contract_type,
    "preferred_work_arrangement": preferred_work_arrangement,
    "skills": _all_skills,
}


# ---------------------------------------------------------------------------
# Run pipeline
# ---------------------------------------------------------------------------

if submitted:

    with st.status(
        "Finding your matches...",
        expanded=True,
    ) as status:

        try:

            # ---------------------------------------------------------------
            # Step 1 — Profile
            # ---------------------------------------------------------------

            status.update(
                label="Step 1/5 — Reading your profile"
            )

            profile_result = encode_profile(user_profile)

            if profile_result.used_fallback:

                st.warning(
                    "We couldn't reach the full prediction model just now, "
                    "so this run used a simpler backup method for your "
                    "predicted occupation and profile fit score."
                )

            st.write(
                f"Predicted occupation: "
                f"**{profile_result.predicted_occupation}** "
                f"(profile fit score: "
                f"**{profile_result.profile_fit_score:.0%}**)"
            )


            # ---------------------------------------------------------------
            # Step 2 — Employment
            # ---------------------------------------------------------------

            status.update(
                label="Step 2/5 — Estimating your employment odds"
            )

            models = load_xgb_models()

            ei_result = models.run(user_profile)

            st.write(
                "Calibrated employment probability: "
                f"**{ei_result.employment_probability:.0%}**"
            )


            # ---------------------------------------------------------------
            # Step 3 — Income
            # ---------------------------------------------------------------

            status.update(
                label="Step 3/5 — Estimating your income"
            )

            if ei_result.predicted_income is not None:

                st.write(
                    "Predicted annual income: "
                    f"**${ei_result.predicted_income:,.0f}**"
                )

            else:

                st.write(
                    ei_result.income_skipped_reason
                )


            # ---------------------------------------------------------------
            # Step 4 — Jobs
            # ---------------------------------------------------------------

            status.update(
                label="Step 4/5 — Finding job listings"
            )

            search_keyword = build_search_keyword(
                profile_result.predicted_occupation,
                previous_occupation,
                keyword_override,
            )


            if use_local_jobs:

                jobs_df = cached_get_jobs_local(
                    search_keyword,
                    preferred_city,
                )

                source_label = "expanded local Adzuna dataset"


                # -----------------------------------------------------------
                # Live fallback
                # -----------------------------------------------------------

                if (
                    jobs_df.empty
                    and config.ADZUNA_APP_ID
                    and config.ADZUNA_APP_KEY
                ):

                    jobs_df = cached_get_jobs_live(
                        search_keyword,
                        preferred_city,
                    )

                    source_label = (
                        "live Adzuna API "
                        "(automatic city fallback)"
                    )

            else:

                jobs_df = cached_get_jobs_live(
                    search_keyword,
                    preferred_city,
                )

                source_label = "live Adzuna API"


            st.write(
                f"Search: `{search_keyword}` in `{preferred_city}` → "
                f"**{len(jobs_df)}** jobs from the {source_label}."
            )


            if use_local_jobs and not jobs_df.empty:

                st.caption(
                    "ℹ️ Jobs were matched against the expanded Ontario "
                    "dataset. City matching recognizes common Adzuna "
                    "regional location formats."
                )

            elif use_local_jobs and jobs_df.empty:

                if (
                    config.ADZUNA_APP_ID
                    and config.ADZUNA_APP_KEY
                ):

                    st.warning(
                        f"No local job postings were found for "
                        f"**{preferred_city}**. The app attempted a live "
                        "Adzuna search for that city."
                    )

                else:

                    st.warning(
                        f"No job postings were found for "
                        f"**{preferred_city}** in the local dataset. "
                        "Try another Ontario city or select "
                        "**Live Adzuna API**."
                    )


            # ---------------------------------------------------------------
            # Step 5 — Ranking
            # ---------------------------------------------------------------

            status.update(
                label="Step 5/5 — Ranking your recommendations"
            )


            if jobs_df.empty:

                ranked_jobs = pd.DataFrame()
                housing_recommendations = pd.DataFrame()

                status.update(
                    label="No jobs found for this search",
                    state="error",
                )

            else:

                karthika_profile = {
                    "occupation_category": occupation_category,
                    "previous_occupation": previous_occupation,
                    "field_of_study": field_of_study,
                    "years_of_experience": years_of_experience,
                    "TEER_category": teer_category,
                    "regulated_profession": (
                        regulated_profession == "Yes"
                    ),
                    "credential_recognition_status": (
                        credential_recognition_status
                    ),
                    "additional_skills": _all_skills,
                    "preferred_cities": [preferred_city],
                    "preferred_city": preferred_city,
                    "minimum_salary": 65000,
                    "max_commute_km": 30,
                    "max_rent_income_ratio": 0.30,
                    "top_jobs": 10,
                    "homes_per_job": 5,
                    "employment_probability": (
                        ei_result.employment_probability
                    ),
                    "predicted_income": (
                        ei_result.predicted_income or 65000
                    ),
                    "profile_fit_score": (
                        profile_result.profile_fit_score
                    ),
                }


                ranked_jobs = (
                    karthika_rank_jobs(
                        jobs_df,
                        karthika_profile,
                    )
                    .head(10)
                    .copy()
                )


                housing_recommendations = (
                    karthika_build_housing_recommendations(
                        housing_df
                        if housing_df is not None
                        else pd.DataFrame(),
                        ranked_jobs,
                        karthika_profile,
                    )
                )


                status.update(
                    label="Matches ready!",
                    state="complete",
                )


            # ---------------------------------------------------------------
            # Save results
            # ---------------------------------------------------------------

            st.session_state["pipeline_results"] = {
                "profile_result": profile_result,
                "ei_result": ei_result,
                "ranked_jobs": ranked_jobs,
                "housing_recommendations": (
                    housing_recommendations
                ),
                "preferred_city": preferred_city,
                "housing_df": housing_df,
            }


        except Exception as e:

            status.update(
                label=f"Pipeline failed: {e}",
                state="error",
            )

            st.exception(e)

            st.stop()


# ---------------------------------------------------------------------------
# Results dashboard
# ---------------------------------------------------------------------------

if "pipeline_results" not in st.session_state:

    st.info(
        "Fill out the form above and click "
        "**Find My Matches** to get recommendations."
    )

    st.stop()


results = st.session_state["pipeline_results"]

profile_result = results["profile_result"]
ei_result = results["ei_result"]
ranked_jobs = results["ranked_jobs"]
preferred_city = results["preferred_city"]
housing_df = results["housing_df"]
housing_recommendations = results.get(
    "housing_recommendations",
    pd.DataFrame(),
)


st.divider()

st.header("Your results")


# ---------------------------------------------------------------------------
# Career snapshot
# ---------------------------------------------------------------------------

st.subheader("Your career snapshot")

headline1, headline2 = st.columns(2)

headline1.metric(
    "Predicted occupation",
    profile_result.predicted_occupation,
)

if ei_result.predicted_income is not None:

    headline2.metric(
        "Expected annual income",
        f"${ei_result.predicted_income:,.0f}",
    )

else:

    headline2.metric(
        "Expected annual income",
        "N/A",
    )


support1, support2 = st.columns(2)

support1.metric(
    "Profile fit score",
    f"{profile_result.profile_fit_score:.0%}",
    help=(
        "How closely your background matches the training data "
        "for your predicted occupation."
    ),
)

support2.metric(
    "Employment probability",
    f"{ei_result.employment_probability:.0%}",
    help=(
        "A calibrated estimate of the likelihood of employment "
        "within the first year."
    ),
)

st.caption(
    "Probability is calibrated against held-out training data; "
    "it is an estimate, not a guarantee of employment."
)


st.divider()


# ---------------------------------------------------------------------------
# Recommended jobs
# ---------------------------------------------------------------------------

st.subheader("Recommended jobs for you")


if ranked_jobs.empty:

    st.info(
        "No job recommendations to display for this search."
    )

else:

    def first_existing(df, candidates):

        for c in candidates:

            if c in df.columns:
                return c

        return None


    title_col = first_existing(
        ranked_jobs,
        ["title", "job_title"],
    )

    company_col = first_existing(
        ranked_jobs,
        ["company", "company_name"],
    )

    salary_col = first_existing(
        ranked_jobs,
        [
            "salary",
            "salary_avg",
            "predicted_salary",
        ],
    )

    salary_min_col = first_existing(
        ranked_jobs,
        ["salary_min"],
    )

    salary_max_col = first_existing(
        ranked_jobs,
        ["salary_max"],
    )

    link_col = first_existing(
        ranked_jobs,
        [
            "redirect_url",
            "url",
            "link",
            "job_url",
        ],
    )


    table = pd.DataFrame()

    table["Title"] = (
        ranked_jobs[title_col]
        if title_col
        else ""
    )

    table["Company"] = (
        ranked_jobs[company_col]
        if company_col
        else ""
    )

    table["Match score"] = ranked_jobs["match_score"]


    def format_salary_range(row):

        smin = (
            row.get(salary_min_col)
            if salary_min_col
            else None
        )

        smax = (
            row.get(salary_max_col)
            if salary_max_col
            else None
        )

        smin = (
            None
            if pd.isna(smin) or smin in (0, "0")
            else float(smin)
        )

        smax = (
            None
            if pd.isna(smax) or smax in (0, "0")
            else float(smax)
        )

        if smin is None and smax is None:
            return "Salary not available"

        if smin is not None and smax is not None:
            return f"${smin:,.0f} – ${smax:,.0f}"

        return f"${(smin or smax):,.0f}"


    if salary_col:

        table["Salary"] = ranked_jobs[
            salary_col
        ].apply(
            lambda v: (
                "Salary not available"
                if pd.isna(v) or v in (0, "0")
                else f"${float(v):,.0f}"
            )
        )

    elif salary_min_col and salary_max_col:

        table["Salary"] = ranked_jobs.apply(
            format_salary_range,
            axis=1,
        )

    else:

        table["Salary"] = "Salary not available"


    table["Link"] = (
        ranked_jobs[link_col]
        if link_col
        else None
    )


    table = table.sort_values(
        "Match score",
        ascending=False,
    )


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
                help=(
                    "How well this job posting fits your profile "
                    "and preferences."
                ),
            ),
            "Link": st.column_config.LinkColumn(
                "Job posting",
                display_text="View job ↗",
            ),
        },
    )


    # -----------------------------------------------------------------------
    # Match score chart
    # -----------------------------------------------------------------------

    st.subheader("Match score comparison")

    chart_df = table.sort_values(
        "Match score",
        ascending=True,
    )

    fig = px.bar(
        chart_df,
        x="Match score",
        y="Title",
        orientation="h",
        color="Match score",
        color_continuous_scale="RdYlGn",
        range_color=[0, 100],
        labels={
            "Match score": "Match score (%)",
            "Title": "",
        },
    )

    fig.update_layout(
        showlegend=False,
        height=max(
            300,
            40 * len(chart_df),
        ),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )


st.divider()


# ---------------------------------------------------------------------------
# Housing affordability
# ---------------------------------------------------------------------------

st.subheader("Can you afford to live there?")


if housing_df is None:

    st.info(
        "Housing dataset not found — affordability can't be "
        "estimated for this run."
    )

else:

    city_housing_afford = housing_df[
        housing_df["city"]
        .fillna("")
        .astype(str)
        .str.lower()
        == preferred_city.strip().lower()
    ]


    if city_housing_afford.empty:

        st.info(
            f"No housing data on file for **{preferred_city}** — "
            "affordability can't be estimated."
        )

    elif ei_result.predicted_income is None:

        st.info(
            "Predicted income wasn't available for this profile "
            f"({ei_result.income_skipped_reason}), so affordability "
            "can't be calculated."
        )

    else:

        avg_rent = (
            city_housing_afford["monthly_rent"]
            .dropna()
            .astype(float)
            .mean()
        )

        monthly_income = (
            ei_result.predicted_income / 12
        )

        rent_to_income = (
            avg_rent / monthly_income
            if monthly_income
            else None
        )


        a1, a2, a3 = st.columns(3)

        a1.metric(
            f"Avg. monthly rent — {preferred_city.title()}",
            f"${avg_rent:,.0f}",
        )

        a2.metric(
            "Predicted monthly income",
            f"${monthly_income:,.0f}",
        )

        if rent_to_income is not None:

            a3.metric(
                "Rent-to-income ratio",
                f"{rent_to_income:.0%}",
            )


        if rent_to_income is not None:

            if rent_to_income <= 0.30:

                st.success(
                    f"🟢 **Affordable** — rent is about "
                    f"{rent_to_income:.0%} of predicted monthly "
                    "income."
                )

            elif rent_to_income <= 0.50:

                st.warning(
                    f"🟠 **Stretched** — rent is about "
                    f"{rent_to_income:.0%} of predicted monthly "
                    "income."
                )

            else:

                st.error(
                    f"🔴 **Unaffordable at this income level** — "
                    f"rent is about {rent_to_income:.0%} of predicted "
                    "monthly income."
                )


        st.caption(
            f"Based on {len(city_housing_afford)} housing data "
            f"point(s) for {preferred_city.title()}."
        )


st.divider()


# ---------------------------------------------------------------------------
# Folium interactive map
# ---------------------------------------------------------------------------

st.subheader("Job locations and nearby housing")


map_jobs = ranked_jobs.copy()
map_homes = housing_recommendations.copy()


# ---------------------------------------------------------------------------
# Validate job coordinates
# ---------------------------------------------------------------------------

if not map_jobs.empty:

    if "latitude" in map_jobs.columns:

        map_jobs["latitude"] = pd.to_numeric(
            map_jobs["latitude"],
            errors="coerce",
        )

    else:

        map_jobs["latitude"] = None


    if "longitude" in map_jobs.columns:

        map_jobs["longitude"] = pd.to_numeric(
            map_jobs["longitude"],
            errors="coerce",
        )

    else:

        map_jobs["longitude"] = None


    map_jobs = map_jobs.dropna(
        subset=[
            "latitude",
            "longitude",
        ]
    )


    # Ontario / Canada geographic sanity check.
    #
    # These bounds prevent obvious bad coordinates such as:
    #   - ocean locations
    #   - coordinates from another continent
    #   - malformed 0/0 coordinates
    #
    map_jobs = map_jobs[
        map_jobs["latitude"].between(
            40,
            60,
        )
        &
        map_jobs["longitude"].between(
            -95,
            -70,
        )
    ]


# ---------------------------------------------------------------------------
# Validate housing coordinates
# ---------------------------------------------------------------------------

if not map_homes.empty:

    # Re-apply the housing validation.
    map_homes = prepare_housing_data(
        map_homes,
        preferred_city,
    )

    map_homes["lat"] = pd.to_numeric(
        map_homes["lat"],
        errors="coerce",
    )

    map_homes["lon"] = pd.to_numeric(
        map_homes["lon"],
        errors="coerce",
    )

    map_homes = map_homes.dropna(
        subset=[
            "lat",
            "lon",
        ]
    )


    # Same geographic sanity check for housing.
    map_homes = map_homes[
        map_homes["lat"].between(
            40,
            60,
        )
        &
        map_homes["lon"].between(
            -95,
            -70,
        )
    ]


# ---------------------------------------------------------------------------
# Draw map
# ---------------------------------------------------------------------------

if map_jobs.empty and map_homes.empty:

    st.info(
        "No valid recommended job or nearby housing coordinates "
        "were available for the map."
    )

else:

    coordinate_pairs = []


    if not map_jobs.empty:

        coordinate_pairs.extend(
            map_jobs[
                [
                    "latitude",
                    "longitude",
                ]
            ]
            .astype(float)
            .values
            .tolist()
        )


    if not map_homes.empty:

        coordinate_pairs.extend(
            map_homes[
                [
                    "lat",
                    "lon",
                ]
            ]
            .astype(float)
            .values
            .tolist()
        )


    center_lat = (
        sum(p[0] for p in coordinate_pairs)
        / len(coordinate_pairs)
    )

    center_lon = (
        sum(p[1] for p in coordinate_pairs)
        / len(coordinate_pairs)
    )


    job_map = folium.Map(
        location=[
            center_lat,
            center_lon,
        ],
        zoom_start=11,
        control_scale=True,
    )


    jobs_layer = folium.FeatureGroup(
        name="Recommended jobs",
        show=True,
    )

    homes_layer = folium.FeatureGroup(
        name="Nearby housing",
        show=True,
    )


    # -----------------------------------------------------------------------
    # Job markers
    # -----------------------------------------------------------------------

    for _, job in map_jobs.iterrows():

        score = float(
            job.get(
                "karthika_job_score",
                job.get(
                    "match_score",
                    0,
                ),
            )
            or 0
        )


        marker_color = (
            "green"
            if score >= 70
            else "orange"
            if score >= 40
            else "red"
        )


        title = str(
            job.get(
                "title",
                "Job",
            )
        )

        company = str(
            job.get(
                "company",
                "",
            )
        )

        location = str(
            job.get(
                "location",
                "",
            )
        )


        salary_min = job.get(
            "salary_min",
            None,
        )

        salary_max = job.get(
            "salary_max",
            None,
        )


        salary_text = "Salary not available"


        try:

            if (
                pd.notna(salary_min)
                and float(salary_min) > 0
            ):

                if (
                    pd.notna(salary_max)
                    and float(salary_max) > 0
                ):

                    salary_text = (
                        f"${float(salary_min):,.0f} – "
                        f"${float(salary_max):,.0f}"
                    )

                else:

                    salary_text = (
                        f"${float(salary_min):,.0f}+"
                    )

            elif (
                pd.notna(salary_max)
                and float(salary_max) > 0
            ):

                salary_text = (
                    f"Up to ${float(salary_max):,.0f}"
                )

        except Exception:
            salary_text = "Salary not available"


        popup = (
            f"<b>{title}</b><br>"
            f"Company: {company}<br>"
            f"Location: {location}<br>"
            f"Salary: {salary_text}<br>"
            f"Match score: {score:.1f}%"
        )


        folium.Marker(
            location=[
                float(job["latitude"]),
                float(job["longitude"]),
            ],
            popup=folium.Popup(
                popup,
                max_width=320,
            ),
            tooltip=(
                f"Job: {title} "
                f"({score:.0f}%)"
            ),
            icon=folium.Icon(
                color=marker_color,
                icon="briefcase",
                prefix="fa",
            ),
        ).add_to(jobs_layer)


    # -----------------------------------------------------------------------
    # Housing markers
    # -----------------------------------------------------------------------

    for _, home in map_homes.iterrows():

        try:

            rent = float(
                home.get(
                    "monthly_rent",
                    0,
                )
                or 0
            )

        except Exception:

            rent = 0


        try:

            commute = float(
                home.get(
                    "commute_km",
                    0,
                )
                or 0
            )

        except Exception:

            commute = 0


        affordable = bool(
            home.get(
                "affordable",
                False,
            )
        )


        try:

            score = float(
                home.get(
                    "combined_score",
                    0,
                )
                or 0
            )

        except Exception:

            score = 0


        neighbourhood = str(
            home.get(
                "neighbourhood",
                "Housing",
            )
        )

        city = str(
            home.get(
                "city",
                "",
            )
        )

        job_title = str(
            home.get(
                "job_title",
                "",
            )
        )


        popup = (
            f"<b>{neighbourhood}, {city}</b><br>"
            f"Rent: ${rent:,.0f}/month<br>"
            f"Near: {job_title}<br>"
            f"Commute: {commute:.1f} km<br>"
            f"Affordable: "
            f"{'Yes' if affordable else 'No'}<br>"
            f"Combined score: {score:.1f}"
        )


        folium.Marker(
            location=[
                float(home["lat"]),
                float(home["lon"]),
            ],
            popup=folium.Popup(
                popup,
                max_width=320,
            ),
            tooltip=(
                f"Housing: {neighbourhood} "
                f"— {commute:.1f} km from job"
            ),
            icon=folium.Icon(
                color=(
                    "blue"
                    if affordable
                    else "orange"
                ),
                icon="home",
                prefix="fa",
            ),
        ).add_to(homes_layer)


    jobs_layer.add_to(job_map)
    homes_layer.add_to(job_map)


    folium.LayerControl(
        collapsed=False,
    ).add_to(job_map)


    job_map.fit_bounds(
        coordinate_pairs,
        padding=(25, 25),
    )


    st_folium(
        job_map,
        use_container_width=True,
        height=550,
    )


    st.caption(
        "🟢 Strong job match (≥70%)   "
        "🟠 Moderate (40–70%)   "
        "🔴 Weaker (<40%)   "
        "🔵 Nearby affordable housing   "
        "🟠 Housing above the affordability threshold"
    )


    st.info(
        "The map uses the Recommendation Engine. "
        "It shows the top recommended jobs and only nearby housing "
        "within the 30 km commute limit."
    )
```


