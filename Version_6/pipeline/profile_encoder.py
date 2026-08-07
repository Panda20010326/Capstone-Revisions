from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import pandas as pd

from . import config


@dataclass
class ProfileResult:
    predicted_occupation: str
    profile_fit_score: float
    used_fallback: bool
    embedding_available: bool = False


OCCUPATION_MAP = {
    "Accountant": "Business, Finance & Administration",
    "Administrative Officer": "Business, Finance & Administration",
    "Bookkeeper": "Business, Finance & Administration",
    "Data Analyst": "Natural & Applied Sciences",
    "IT Analyst": "Natural & Applied Sciences",
    "Software Developer": "Natural & Applied Sciences",
    "Civil Engineer": "Natural & Applied Sciences",
    "Mechanical Engineer": "Natural & Applied Sciences",
    "Dentist": "Health",
    "Medical Laboratory Technologist": "Health",
    "Pharmacist": "Health",
    "Physician": "Health",
    "Registered Nurse": "Health",
    "Early Childhood Educator": "Education, Law & Social Services",
    "Lawyer": "Education, Law & Social Services",
    "Paralegal": "Education, Law & Social Services",
    "Social Worker": "Education, Law & Social Services",
    "Teacher": "Education, Law & Social Services",
    "Construction Manager": "Management",
    "Operations Manager": "Management",
    "Restaurant Manager": "Management",
    "Retail Manager": "Management",
    "Food Service Supervisor": "Management",
    "Assembly Line Worker": "Manufacturing & Utilities",
    "Machine Operator": "Manufacturing & Utilities",
    "Production Worker": "Manufacturing & Utilities",
    "Customer Service Representative": "Sales & Service",
    "Cashier": "Sales & Service",
    "Retail Sales Associate": "Sales & Service",
    "Auto Mechanic": "Trades, Transport & Equipment Operators",
    "Electrician": "Trades, Transport & Equipment Operators",
    "Plumber": "Trades, Transport & Equipment Operators",
    "Truck Driver": "Trades, Transport & Equipment Operators",
    "Warehouse Labourer": "Trades, Transport & Equipment Operators",
    "Welder": "Trades, Transport & Equipment Operators",
}


def _fallback(profile: dict) -> ProfileResult:
    category = str(profile.get("occupation_category") or "").strip()
    previous = str(profile.get("previous_occupation") or "").strip()

    if category:
        return ProfileResult(
            predicted_occupation=category,
            profile_fit_score=0.75,
            used_fallback=True,
        )

    mapped = OCCUPATION_MAP.get(previous)
    if mapped:
        return ProfileResult(
            predicted_occupation=mapped,
            profile_fit_score=0.65,
            used_fallback=True,
        )

    field = str(profile.get("field_of_study") or "").lower()
    if any(k in field for k in ["computer", "information", "engineering", "mathematics", "science"]):
        mapped = "Natural & Applied Sciences"
    elif any(k in field for k in ["nursing", "health", "pharmacy"]):
        mapped = "Health"
    elif any(k in field for k in ["business", "accounting", "finance", "economics"]):
        mapped = "Business, Finance & Administration"
    else:
        mapped = "Other"

    return ProfileResult(
        predicted_occupation=mapped,
        profile_fit_score=0.50,
        used_fallback=True,
    )


def encode_profile(user_profile: dict) -> ProfileResult:
    """
    Runtime profile stage.

    The source project contains the 16-dimensional ProfileEncoder artifact,
    but it does not contain the category head / category encoder needed to
    turn that embedding into an occupation prediction. Therefore this module
    deliberately does not pretend the embedding alone can predict a category.

    If the full multitask model artifacts are later supplied, this function
    is the single place to add the neural inference path.
    """
    return _fallback(user_profile)
