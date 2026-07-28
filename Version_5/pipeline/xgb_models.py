"""
xgb_models.py — Stage 2 (Employment XGBoost) + Stage 3 (Income XGBoost)
-------------------------------------------------------------------------
This is the original Employment_Prediction/Austine Handoff/model_interface.py,
adapted to load its four .pkl artifacts from artifacts/xgboost/ (see
pipeline/config.py) instead of the current working directory, and wrapped in
a small class so app.py can call one function per stage as shown in the
pipeline diagram.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd

from . import config

# The trained booster .pkl files were fit with the profile's occupation-category
# column literally named "employment_category" (verified via
# classifier.get_booster().feature_names), even though the label encoder dict
# and every other artifact (ProfileEncoder, rare_category_map) key it as
# "occupation_category". Same values, same encoding -- just a naming mismatch
# baked into these two pickles at training time. Rename right before
# prediction so XGBoost's strict feature-name check doesn't reject the row.
# If Employment_Prediction/Austine Handoff models get retrained, check
# `classifier.get_booster().feature_names` again and update/remove this.
_XGB_COLUMN_RENAME = {"occupation_category": "employment_category"}


@dataclass
class EmploymentIncomeResult:
    employment_probability: float
    predicted_income: Optional[float]  # None if employment probability is below threshold
    income_skipped_reason: Optional[str] = None


class EmploymentIncomeModels:
    def __init__(self):
        self.classifier = joblib.load(config.CLASSIFIER_PATH)
        self.regressor = joblib.load(config.REGRESSOR_PATH)
        self.label_encoders = joblib.load(config.LABEL_ENCODERS_PATH)  # dict of LabelEncoders
        self.rare_category_map = joblib.load(config.RARE_CATEGORY_MAP_PATH)  # dict col -> rare values

    def preprocess_profile(self, user_profile: dict) -> pd.DataFrame:
        """Raw, human-readable profile dict -> single-row encoded DataFrame."""
        row = pd.DataFrame([user_profile])[config.PROFILE_FEATURES]

        for col, rare_categories in self.rare_category_map.items():
            if col in row.columns:
                row[col] = row[col].where(~row[col].isin(rare_categories), "Other")

        for col, encoder in self.label_encoders.items():
            if col not in row.columns:
                continue
            value = row.at[0, col]
            if value not in encoder.classes_:
                value = "Other" if "Other" in encoder.classes_ else encoder.classes_[0]
                row.at[0, col] = value
            row[col] = encoder.transform(row[col])

        return row

    def predict_employment(self, features: pd.DataFrame) -> float:
        features = features.rename(columns=_XGB_COLUMN_RENAME)
        return float(self.classifier.predict_proba(features)[0, 1])

    def predict_income(self, features: pd.DataFrame) -> float:
        features = features.rename(columns=_XGB_COLUMN_RENAME)
        return float(self.regressor.predict(features)[0])

    def run(self, user_profile: dict,
            employment_threshold: float = config.EMPLOYMENT_PROBABILITY_THRESHOLD) -> EmploymentIncomeResult:
        """Runs Stage 2 then Stage 3, skipping income if unlikely to be employed."""
        features = self.preprocess_profile(user_profile)
        employment_probability = self.predict_employment(features)

        if employment_probability < employment_threshold:
            return EmploymentIncomeResult(
                employment_probability=employment_probability,
                predicted_income=None,
                income_skipped_reason=(
                    f"Employment probability ({employment_probability:.0%}) is below the "
                    f"{employment_threshold:.0%} threshold, so income was not estimated — "
                    "the regressor was only trained on employed profiles."
                ),
            )

        predicted_income = self.predict_income(features)
        return EmploymentIncomeResult(
            employment_probability=employment_probability,
            predicted_income=predicted_income,
        )


_models_instance: Optional[EmploymentIncomeModels] = None


def get_models() -> EmploymentIncomeModels:
    global _models_instance
    if _models_instance is None:
        _models_instance = EmploymentIncomeModels()
    return _models_instance
