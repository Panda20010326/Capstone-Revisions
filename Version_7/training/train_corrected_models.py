"""
train_corrected_models.py
-------------------------
Retrains the two runtime XGBoost models using only profile fields available
in the Streamlit form.

Important correction:
The old serialized models in the source project were built with
`employment_category`, which is an outcome-derived field and is not available
from the user form. The corrected models use `occupation_category` instead.

Run once when the training data changes:
    python training/train_corrected_models.py

The Streamlit app never trains models. It only loads the saved artifacts.
"""

from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    mean_absolute_error, mean_squared_error, r2_score
)
from xgboost import XGBClassifier, XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "newcomer_ontario_enriched.csv"
ARTIFACT_DIR = ROOT / "artifacts" / "xgboost"

FEATURES = [
    "age", "sex", "admission_category", "world_region",
    "speaks_official_language", "education_level", "family_size",
    "field_of_study", "previous_occupation", "occupation_category",
    "years_of_experience", "teer_category",
    "credential_recognition_status", "regulated_profession",
]

CATEGORICAL_FEATURES = [
    "sex", "admission_category", "world_region", "education_level",
    "field_of_study", "previous_occupation", "occupation_category",
    "teer_category", "credential_recognition_status",
]

HIGH_CARDINALITY = ["previous_occupation", "field_of_study"]
RARE_THRESHOLD = 0.01


def prepare_dataframe(df):
    work = df[FEATURES + ["employed", "annual_income"]].copy()
    rare_map = {}

    for col in HIGH_CARDINALITY:
        frequencies = work[col].value_counts(normalize=True)
        rare = frequencies[frequencies < RARE_THRESHOLD].index.tolist()
        rare_map[col] = rare
        work[col] = work[col].where(~work[col].isin(rare), "Other")

    encoders = {}
    for col in CATEGORICAL_FEATURES:
        encoder = LabelEncoder()
        work[col] = encoder.fit_transform(work[col].astype(str))
        encoders[col] = encoder

    return work, encoders, rare_map


def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    work, encoders, rare_map = prepare_dataframe(df)

    # Employment classifier
    X = work[FEATURES]
    y = work["employed"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos

    classifier = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    classifier.fit(X_train, y_train)

    probabilities = classifier.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= 0.50).astype(int)

    classifier_metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "f1": f1_score(y_test, predictions),
        "precision": precision_score(y_test, predictions),
        "recall": recall_score(y_test, predictions),
        "roc_auc": roc_auc_score(y_test, probabilities),
    }

    # Income regressor: only employed people, as in the original project.
    employed = work[work["employed"] == 1]
    X_income = employed[FEATURES]
    y_income = employed["annual_income"]

    X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
        X_income, y_income, test_size=0.20, random_state=42
    )

    regressor = XGBRegressor(random_state=42, verbosity=0)
    regressor.fit(X_train_i, y_train_i)
    income_pred = regressor.predict(X_test_i)

    income_metrics = {
        "mae": mean_absolute_error(y_test_i, income_pred),
        "rmse": np.sqrt(mean_squared_error(y_test_i, income_pred)),
        "r2": r2_score(y_test_i, income_pred),
    }

    joblib.dump(classifier, ARTIFACT_DIR / "employment_classifier_corrected.pkl")
    joblib.dump(regressor, ARTIFACT_DIR / "income_regressor_corrected.pkl")
    joblib.dump(encoders, ARTIFACT_DIR / "label_encoders_corrected.pkl")
    joblib.dump(rare_map, ARTIFACT_DIR / "rare_category_map.pkl")

    metadata = {
        "features": FEATURES,
        "classifier_metrics": classifier_metrics,
        "income_metrics": income_metrics,
        "threshold": 0.50,
        "note": "Uses occupation_category instead of employment_category to avoid outcome leakage and match app inputs.",
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("Corrected models saved to:", ARTIFACT_DIR)
    print("Classifier:", classifier_metrics)
    print("Income:", income_metrics)


if __name__ == "__main__":
    main()
