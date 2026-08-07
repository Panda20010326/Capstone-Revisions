from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import joblib
import pandas as pd

from . import config

FEATURES = [
    "age", "sex", "admission_category", "world_region",
    "speaks_official_language", "education_level", "family_size",
    "field_of_study", "previous_occupation", "occupation_category",
    "years_of_experience", "teer_category",
    "credential_recognition_status", "regulated_profession",
]


@dataclass
class PredictionResult:
    employment_probability: float
    predicted_income: Optional[float]
    income_skipped_reason: Optional[str] = None


class XGBModels:
    """Loads trained artifacts once and performs inference only."""

    def __init__(self):
        self.classifier = joblib.load(config.CLASSIFIER_PATH)
        self.regressor = joblib.load(config.REGRESSOR_PATH)
        self.encoders = joblib.load(config.LABEL_ENCODERS_PATH)
        self.rare_map = joblib.load(config.RARE_CATEGORY_MAP_PATH)

    def _encode(self, profile: dict) -> pd.DataFrame:
        row = pd.DataFrame([profile])[FEATURES].copy()

        for col, rare_values in self.rare_map.items():
            if col in row.columns:
                row[col] = row[col].where(~row[col].isin(rare_values), "Other")

        for col, encoder in self.encoders.items():
            if col not in row.columns:
                continue
            value = row.at[0, col]
            if value not in encoder.classes_:
                value = "Other" if "Other" in encoder.classes_ else encoder.classes_[0]
            row.at[0, col] = encoder.transform([value])[0]

        for col in self.encoders:
            if col in row.columns:
                row[col] = pd.to_numeric(row[col], errors="raise").astype("int64")

        return row

    def run(self, profile: dict, income_threshold: float = 0.50) -> PredictionResult:
        features = self._encode(profile)
        employment_probability = float(
            self.classifier.predict_proba(features)[0, 1]
        )

        if employment_probability < income_threshold:
            return PredictionResult(
                employment_probability=employment_probability,
                predicted_income=None,
                income_skipped_reason=(
                    f"Income is only estimated for profiles above the "
                    f"{income_threshold:.0%} employment-probability threshold."
                ),
            )

        predicted_income = max(
            0.0, float(self.regressor.predict(features)[0])
        )
        return PredictionResult(
            employment_probability=employment_probability,
            predicted_income=predicted_income,
        )


@__import__("functools").lru_cache(maxsize=1)
def get_models() -> XGBModels:
    return XGBModels()
