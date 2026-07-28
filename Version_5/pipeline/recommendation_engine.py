from __future__ import annotations
import math
import re
from typing import Any
import numpy as np
import pandas as pd
from .explanation import generate_explanation

DEFAULT_WEIGHTS = {
    "occupation": 0.25, "skills": 0.25, "salary": 0.15,
    "location": 0.10, "contract": 0.10, "profile_fit": 0.10,
    "employment": 0.05,
}

OCCUPATION_KEYWORDS = {
    "Business, Finance and Administration": [
        "business analyst", "financial analyst", "accountant", "finance",
        "bookkeeper", "administrative", "payroll", "banking", "auditor"
    ],
    "Natural and Applied Sciences": [
        "data analyst", "data scientist", "software", "developer", "engineer",
        "cybersecurity", "machine learning", "programmer", "cloud", "network"
    ],
    "Health": ["nurse", "health", "medical", "pharmacy", "clinical", "caregiver", "therapist"],
    "Sales and Service": ["sales", "customer service", "retail", "representative", "hospitality"],
    "Trades, Transport and Equipment": ["driver", "mechanic", "electrician", "plumber", "welder", "construction"],
    "Manufacturing and Utilities": ["manufacturing", "production", "machine operator", "warehouse", "assembler"],
    "Education, Law and Social Services": ["teacher", "education", "legal", "law", "social worker", "counsellor"],
    "Art, Culture, Recreation and Sport": ["designer", "artist", "media", "writer", "recreation", "sport"],
    "Management": ["manager", "director", "supervisor", "lead", "executive", "coordinator"],
}

def _text(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value).strip().lower()

def _skill_score(skills: list[str], job_text: str):
    if not skills:
        return 0.5, []
    matched = [s for s in skills if re.search(r"\b" + re.escape(s.lower()) + r"\b", job_text)]
    return len(matched) / len(skills), matched

def _occupation_score(occupation: str | None, job_text: str) -> float:
    if not occupation:
        return 0.5
    keywords = OCCUPATION_KEYWORDS.get(str(occupation).strip(), [])
    if not keywords:
        return 0.5
    hits = sum(k in job_text for k in keywords)
    return 0.15 if hits == 0 else min(1.0, 0.70 + 0.10 * (hits - 1))

def _salary_score(predicted_income, salary_min, salary_max) -> float:
    if predicted_income is None:
        return 0.5
    values = []
    for value in (salary_min, salary_max):
        try:
            value = float(value)
            if np.isfinite(value) and value > 0:
                values.append(value)
        except (TypeError, ValueError):
            pass
    if not values:
        return 0.5
    advertised = sum(values) / len(values)
    return max(0.0, 1.0 - abs(advertised - float(predicted_income)) / max(float(predicted_income), 1.0))

def _location_score(preferred_city, location) -> float:
    if not preferred_city:
        return 0.5
    preferred, actual = _text(preferred_city), _text(location)
    if not actual:
        return 0.4
    return 1.0 if preferred in actual or actual in preferred else 0.25

def _contract_score(profile, job) -> float:
    preferred_contract = _text(profile.get("preferred_contract_type"))
    preferred_arrangement = _text(profile.get("preferred_work_arrangement"))
    job_text = " ".join([_text(job.get("title")), _text(job.get("description")),
                         _text(job.get("contract_type")), _text(job.get("contract_time"))])
    scores = []
    if preferred_contract:
        scores.append(1.0 if preferred_contract in job_text or preferred_contract.replace(" ", "_") in job_text else 0.3)
    if preferred_arrangement:
        scores.append(1.0 if preferred_arrangement in job_text else 0.4)
    return sum(scores) / len(scores) if scores else 0.5

def score_job(user_profile: dict[str, Any], job: dict[str, Any],
              predictions: dict[str, Any], weights=None) -> dict[str, Any]:
    weights = weights or DEFAULT_WEIGHTS
    job_text = " ".join([_text(job.get("title")), _text(job.get("description")), _text(job.get("category"))])
    skills = [str(s).strip() for s in user_profile.get("skills", []) if str(s).strip()]
    skills_score, matched_skills = _skill_score(skills, job_text)
    breakdown = {
        "occupation": _occupation_score(predictions.get("predicted_occupation"), job_text),
        "skills": skills_score,
        "salary": _salary_score(predictions.get("predicted_income"), job.get("salary_min"), job.get("salary_max")),
        "location": _location_score(user_profile.get("preferred_city"), job.get("location")),
        "contract": _contract_score(user_profile, job),
        "profile_fit": float(predictions.get("profile_fit_score", 0.5)),
        "employment": float(predictions.get("employment_probability", 0.5)),
    }
    breakdown = {k: min(max(v, 0.0), 1.0) for k, v in breakdown.items()}
    total = sum(breakdown[k] * weights.get(k, 0) for k in breakdown) / sum(weights.values())
    explanation = generate_explanation(user_profile, job, predictions, breakdown, matched_skills)
    return {**job, "match_score": round(total * 100, 2),
            "score_breakdown": breakdown, "matched_skills": matched_skills,
            "explanation": explanation}

def recommend_jobs(user_profile: dict[str, Any], predictions: dict[str, Any],
                   jobs: pd.DataFrame | list[dict[str, Any]], top_n: int = 10,
                   weights=None) -> pd.DataFrame:
    records = jobs.to_dict("records") if isinstance(jobs, pd.DataFrame) else list(jobs)
    ranked = [score_job(user_profile, job, predictions, weights) for job in records]
    ranked.sort(key=lambda x: x["match_score"], reverse=True)
    return pd.DataFrame(ranked[:top_n])
