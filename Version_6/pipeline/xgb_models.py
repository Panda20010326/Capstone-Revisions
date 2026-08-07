```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import joblib
import pandas as pd

from . import config


FEATURES = [
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
        # IMPORTANT:
        # Create the DataFrame with object dtype so that categorical
        # strings can safely be replaced with encoded integers.
        row = pd.DataFrame(
            [[profile.get(feature) for feature in FEATURES]],
            columns=FEATURES,
            dtype=object,
        )

        # ---------------------------------------------------------------
        # Handle rare categories
        # ---------------------------------------------------------------
        for col, rare_values in self.rare_map.items():
            if col not in row.columns:
                continue

            value = row.at[0, col]

            if value is not None and value in rare_values:
                row.at[0, col] = "Other"

        # ---------------------------------------------------------------
        # Encode categorical variables
        # ---------------------------------------------------------------
        for col, encoder in self.encoders.items():
            if col not in row.columns:
                continue

            value = row.at[0, col]

            # The encoders were trained on categorical/string values.
            # Convert incoming categorical values to strings before
            # checking them against encoder.classes_.
            if pd.notna(value):
                value = str(value)

            # Handle values that were not present during training.
            if value not in encoder.classes_:
                if "Other" in encoder.classes_:
                    value = "Other"
                else:
                    value = encoder.classes_[0]

            encoded_value = encoder.transform([value])[0]

            # Because row was created with dtype=object, assigning
            # the integer encoded value is safe.
            row.at[0, col] = encoded_value

        # ---------------------------------------------------------------
        # Convert encoded columns to numeric integers
        # ---------------------------------------------------------------
        for col in self.encoders:
            if col in row.columns:
                row[col] = pd.to_numeric(
                    row[col],
                    errors="raise",
                ).astype("int64")

        # ---------------------------------------------------------------
        # Convert remaining numerical features
        # ---------------------------------------------------------------
        numerical_features = [
            col for col in FEATURES
            if col not in self.encoders
        ]

        for col in numerical_features:
            row[col] = pd.to_numeric(
                row[col],
                errors="raise",
            )

        # Make sure feature order is exactly what the models expect.
        row = row[FEATURES]

        return row

    def run(
        self,
        profile: dict,
        income_threshold: float = 0.50,
    ) -> PredictionResult:

        features = self._encode(profile)

        # ---------------------------------------------------------------
        # Employment prediction
        # ---------------------------------------------------------------
        employment_probability = float(
            self.classifier.predict_proba(features)[0, 1]
        )

        # ---------------------------------------------------------------
        # Income prediction
        # ---------------------------------------------------------------
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
            0.0,
            float(self.regressor.predict(features)[0]),
        )

        return PredictionResult(
            employment_probability=employment_probability,
            predicted_income=predicted_income,
        )


@__import__("functools").lru_cache(maxsize=1)
def get_models() -> XGBModels:
    return XGBModels()
```


