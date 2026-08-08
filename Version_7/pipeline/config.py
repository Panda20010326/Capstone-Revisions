from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = ROOT / "artifacts"
XGB_DIR = ARTIFACTS_DIR / "xgboost"
PROFILE_DIR = ARTIFACTS_DIR / "profile_encoder"
DATA_DIR = ROOT / "data"

CLASSIFIER_PATH = XGB_DIR / "employment_classifier_corrected.pkl"
REGRESSOR_PATH = XGB_DIR / "income_regressor_corrected.pkl"
LABEL_ENCODERS_PATH = XGB_DIR / "label_encoders_corrected.pkl"
RARE_CATEGORY_MAP_PATH = XGB_DIR / "rare_category_map.pkl"

PROFILE_ENCODER_MODEL_PATH = PROFILE_DIR / "profile_encoder_v1_1.keras"
PREPROCESSOR_PATH = PROFILE_DIR / "profile_encoder_preprocessor_v1_1.joblib"
PROFILE_CONFIG_PATH = PROFILE_DIR / "profile_encoder_config_v1_1.json"

LOCAL_JOBS_DATASET_PATH = DATA_DIR / "processed_adzuna_jobs.csv"
HOUSING_DATA_PATH = DATA_DIR / "housing_geocoded.csv"

JOB_SOURCE_DEFAULT = os.getenv("JOB_SOURCE_DEFAULT", "local")
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID", "")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY", "")

# Compatibility names used by app.py
MULTITASK_MODEL_PATH = PROFILE_DIR / 'profile_multitask_model_v1_1.keras'
