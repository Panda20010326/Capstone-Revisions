"""
profile_encoder.py — Stage 1 of the pipeline
---------------------------------------------
Wraps the multi-task ProfileEncoder network (06_ProfileEncoder_revised.ipynb)
and exposes a single entry point:

    encode_profile(user_profile: dict) -> ProfileEncoderResult

which returns:
    - embedding            : 16-d numpy array (None if running in fallback mode)
    - predicted_occupation : str, one of the employment_category labels
    - profile_fit_score    : float 0-1, confidence of the occupation prediction
    - used_fallback        : bool, True if the real network wasn't available

FALLBACK MODE
-------------
The notebook that trains the ProfileEncoder (06_ProfileEncoder_revised.ipynb)
saves five files (profile_encoder_v1_1.keras, profile_multitask_model_v1_1.keras,
profile_encoder_preprocessor_v1_1.joblib, profile_encoder_income_scaler_v1_1.joblib,
profile_encoder_category_encoder_v1_1.joblib) plus profile_encoder_config_v1_1.json.

As of this integration, only the config JSON ships with the repo — the trained
model files still need to be produced by re-running that notebook and copied
into artifacts/profile_encoder/. Until then, this module uses a transparent
heuristic fallback (keyword-matched occupation category + neutral fit score)
so the rest of the pipeline can still be built, tested, and demoed end-to-end.
Swap in the real files and nothing else in the app needs to change.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

from . import config

_KERAS_CUSTOM_OBJECTS_NAME = "FeatureSlice"


@dataclass
class ProfileEncoderResult:
    embedding: Optional[np.ndarray]
    predicted_occupation: str
    profile_fit_score: float
    category_probabilities: dict = field(default_factory=dict)
    used_fallback: bool = False


class _RealProfileEncoder:
    """Loads the actual trained Keras artifacts."""

    def __init__(self):
        # Imported lazily: tensorflow is a heavy dependency and only needed
        # if the real artifacts are present.
        import joblib
        from tensorflow import keras
        from tensorflow.keras import layers

        @keras.utils.register_keras_serializable(package="ProfileEncoder")
        class FeatureSlice(layers.Layer):
            """Must match the layer defined in 06_ProfileEncoder_revised.ipynb."""

            def __init__(self, start_index, end_index=None, **kwargs):
                super().__init__(**kwargs)
                self.start_index = int(start_index)
                self.end_index = None if end_index is None else int(end_index)

            def call(self, inputs):
                return inputs[:, self.start_index:self.end_index]

            def compute_output_shape(self, input_shape):
                width = (input_shape[-1] - self.start_index) if self.end_index is None \
                    else (self.end_index - self.start_index)
                return (input_shape[0], width)

            def get_config(self):
                cfg = super().get_config()
                cfg.update({"start_index": self.start_index, "end_index": self.end_index})
                return cfg

        self.embedding_model = keras.models.load_model(
            config.PROFILE_ENCODER_MODEL_PATH,
            custom_objects={_KERAS_CUSTOM_OBJECTS_NAME: FeatureSlice},
            compile=False,
        )
        self.multitask_model = keras.models.load_model(
            config.MULTITASK_MODEL_PATH,
            custom_objects={_KERAS_CUSTOM_OBJECTS_NAME: FeatureSlice},
            compile=False,
        )
        self.preprocessor = joblib.load(config.PREPROCESSOR_PATH)
        self.category_encoder = joblib.load(config.CATEGORY_ENCODER_PATH)

        with open(config.PROFILE_ENCODER_CONFIG_PATH, encoding="utf-8") as f:
            self.model_config = json.load(f)

    def encode(self, user_profile: dict) -> ProfileEncoderResult:
        row = pd.DataFrame([user_profile])[config.PROFILE_FEATURES]
        processed = self.preprocessor.transform(row).astype("float32")

        embedding = self.embedding_model.predict(processed, verbose=0)[0]
        predictions = self.multitask_model.predict(processed, verbose=0)

        category_probs = predictions["category_output"][0]
        top_index = int(np.argmax(category_probs))
        predicted_occupation = str(self.category_encoder.inverse_transform([top_index])[0])
        profile_fit_score = float(category_probs[top_index])

        prob_dict = {
            str(label): float(prob)
            for label, prob in zip(self.category_encoder.classes_, category_probs)
        }

        return ProfileEncoderResult(
            embedding=embedding,
            predicted_occupation=predicted_occupation,
            profile_fit_score=profile_fit_score,
            category_probabilities=prob_dict,
            used_fallback=False,
        )


# ---------------------------------------------------------------------------
# Heuristic fallback — used only if the trained artifacts above aren't found.
# ---------------------------------------------------------------------------
_FALLBACK_CATEGORY_MAP = {
    "Business, Finance & Administration": ["business", "finance", "account", "administra", "bank"],
    "Natural & Applied Sciences": ["data", "software", "engineer", "developer", "IT", "computer", "cyber"],
    "Health": ["nurse", "health", "medical", "pharma", "clinical", "care"],
    "Sales & Service": ["sales", "customer", "retail", "hospitality"],
    "Trades, Transport & Equipment Operators": ["driver", "mechanic", "electric", "plumb", "weld", "construction"],
    "Manufacturing & Utilities": ["manufactur", "production", "warehouse", "assembl"],
    "Education, Law & Social Services": ["teach", "education", "legal", "law", "social work", "counsel"],
    "Arts, Culture & Recreation": ["design", "artist", "media", "writer", "recreation", "sport"],
    "Management": ["manager", "director", "supervisor", "lead", "executive", "coordinat"],
}


def _fallback_encode(user_profile: dict) -> ProfileEncoderResult:
    text = " ".join(str(user_profile.get(k, "")) for k in
                     ("previous_occupation", "occupation_category", "field_of_study")).lower()

    best_category, best_hits = user_profile.get("occupation_category") or "Business, Finance & Administration", 0
    for category, keywords in _FALLBACK_CATEGORY_MAP.items():
        hits = sum(1 for kw in keywords if kw in text)
        if hits > best_hits:
            best_category, best_hits = category, hits

    # Neutral confidence: a bit more confident if we actually matched keywords.
    fit_score = 0.55 if best_hits == 0 else min(0.85, 0.55 + 0.10 * best_hits)

    return ProfileEncoderResult(
        embedding=None,
        predicted_occupation=best_category,
        profile_fit_score=fit_score,
        category_probabilities={},
        used_fallback=True,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
_encoder_instance = None
_encoder_load_attempted = False


def _artifacts_present() -> bool:
    required = [
        config.PROFILE_ENCODER_MODEL_PATH,
        config.MULTITASK_MODEL_PATH,
        config.PREPROCESSOR_PATH,
        config.CATEGORY_ENCODER_PATH,
    ]
    return all(os.path.exists(p) for p in required)


def encode_profile(user_profile: dict) -> ProfileEncoderResult:
    """Main entry point used by the Streamlit app."""
    global _encoder_instance, _encoder_load_attempted

    if not _artifacts_present():
        return _fallback_encode(user_profile)

    if _encoder_instance is None and not _encoder_load_attempted:
        _encoder_load_attempted = True
        try:
            _encoder_instance = _RealProfileEncoder()
        except Exception as exc:  # noqa: BLE001 - surface as fallback, don't crash the app
            print(f"[profile_encoder] Failed to load trained artifacts, using fallback: {exc}")
            _encoder_instance = None

    if _encoder_instance is None:
        return _fallback_encode(user_profile)

    return _encoder_instance.encode(user_profile)
