from __future__ import annotations

import os
import requests
import pandas as pd


def get_jobs_multi_page(keyword: str, city: str, pages: int = 3, results_per_page: int = 20) -> pd.DataFrame:
    app_id = os.getenv("ADZUNA_APP_ID", "")
    app_key = os.getenv("ADZUNA_APP_KEY", "")
    if not app_id or not app_key:
        raise RuntimeError("ADZUNA_APP_ID and ADZUNA_APP_KEY are required for live Adzuna mode.")

    records = []
    for page in range(1, pages + 1):
        url = f"https://api.adzuna.com/v1/api/jobs/ca/search/{page}"
        params = {
            "app_id": app_id,
            "app_key": app_key,
            "what": keyword,
            "where": city,
            "results_per_page": results_per_page,
            "content-type": "application/json",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        results = response.json().get("results", [])
        if not results:
            break
        records.extend(_normalize_job(j) for j in results)

    return pd.DataFrame(records)


def _normalize_job(job: dict) -> dict:
    return {
        "id": job.get("id"),
        "title": job.get("title"),
        "company": (job.get("company") or {}).get("display_name"),
        "location": (job.get("location") or {}).get("display_name"),
        "salary_min": job.get("salary_min"),
        "salary_max": job.get("salary_max"),
        "salary_is_predicted": job.get("salary_is_predicted"),
        "latitude": job.get("latitude"),
        "longitude": job.get("longitude"),
        "description": job.get("description"),
        "category": (job.get("category") or {}).get("label"),
        "category_tag": (job.get("category") or {}).get("tag"),
        "contract_type": job.get("contract_type"),
        "contract_time": job.get("contract_time"),
        "created": job.get("created"),
        "url": job.get("redirect_url"),
    }


def build_search_keyword(predicted_occupation: str, previous_occupation: str, override: str = "") -> str:
    if override.strip():
        return override.strip()
    if predicted_occupation and predicted_occupation != "Other":
        return predicted_occupation
    return previous_occupation or "jobs"
