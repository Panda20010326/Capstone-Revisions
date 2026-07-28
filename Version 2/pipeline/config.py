"""
config.py
---------
Single source of truth for file paths and feature lists used across every
stage of the pipeline. Every other module imports from here so that moving
a file only ever means editing one line.
"""

from __future__ import annotations
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTIFACTS_DIR = os.path.join(BASE_DIR, "artifacts")
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Shared feature schema (identical across ProfileEncoder and the XGBoost
# classifier/regressor — this is what makes them chainable).
# ---------------------------------------------------------------------------
PROFILE_FEATURES = [
    "age",
    "sex",
    "admission_category",
    "world_region",
    "speaks_official_language",
    "education_level",
    "family_size",
    "field_of_study",
    "previous_occupation",
    "occupation_category",
    "years_of_experience",
    "teer_category",
    "credential_recognition_status",
    "regulated_profession",
]

NUMERIC_FEATURES = [
    "age",
    "speaks_official_language",
    "family_size",
    "years_of_experience",
    "regulated_profession",
]

CATEGORICAL_FEATURES = [
    "sex",
    "admission_category",
    "world_region",
    "education_level",
    "field_of_study",
    "previous_occupation",
    "occupation_category",
    "teer_category",
    "credential_recognition_status",
]

# ---------------------------------------------------------------------------
# ProfileEncoder artifact paths (Stage 1)
# Drop the files produced by 06_ProfileEncoder_revised.ipynb into
# artifacts/profile_encoder/ using these exact names, or the app falls back
# to a heuristic (see pipeline/profile_encoder.py).
# ---------------------------------------------------------------------------
PROFILE_ENCODER_DIR = os.path.join(ARTIFACTS_DIR, "profile_encoder")
PROFILE_ENCODER_MODEL_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_encoder_v1_1.keras")
MULTITASK_MODEL_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_multitask_model_v1_1.keras")
PREPROCESSOR_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_encoder_preprocessor_v1_1.joblib")
INCOME_SCALER_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_encoder_income_scaler_v1_1.joblib")
CATEGORY_ENCODER_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_encoder_category_encoder_v1_1.joblib")
PROFILE_ENCODER_CONFIG_PATH = os.path.join(PROFILE_ENCODER_DIR, "profile_encoder_config_v1_1.json")

# ---------------------------------------------------------------------------
# Employment / Income XGBoost artifact paths (Stage 2 + 3)
# These ship with the app already (Employment_Prediction/Austine Handoff/*).
# ---------------------------------------------------------------------------
XGB_DIR = os.path.join(ARTIFACTS_DIR, "xgboost")
CLASSIFIER_PATH = os.path.join(XGB_DIR, "Secondversion_XGB_classifier.pkl")
REGRESSOR_PATH = os.path.join(XGB_DIR, "Secondversion_XGB_regressor.pkl")
LABEL_ENCODERS_PATH = os.path.join(XGB_DIR, "Secondversion_EDA_preprocessing.pkl")
RARE_CATEGORY_MAP_PATH = os.path.join(XGB_DIR, "rare_category_map.pkl")

# Only call the income regressor if employment probability clears this bar
# (the regressor was trained only on employed profiles).
EMPLOYMENT_PROBABILITY_THRESHOLD = 0.40

# ---------------------------------------------------------------------------
# Adzuna (Stage 4)
# ---------------------------------------------------------------------------
ADZUNA_COUNTRY = "ca"
ADZUNA_RESULTS_PER_PAGE = 20
ADZUNA_MAX_PAGES = 3

# ---------------------------------------------------------------------------
# Folium map (Stage 6)
# ---------------------------------------------------------------------------
HOUSING_DATA_PATH = os.path.join(DATA_DIR, "housing_geocoded.csv")
DEFAULT_MAP_CENTER = (43.6532, -79.3832)  # Toronto
