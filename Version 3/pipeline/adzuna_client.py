"""
adzuna_client.py — Stage 4 of the pipeline
--------------------------------------------
Wraps Employment_Prediction/Adzuna_Notebooks/adzuna_api.py.

SECURITY NOTE
-------------
The original adzuna_api.py had the APP_ID and APP_KEY hardcoded directly in
the source file, which means they're already sitting in plain text in the
shared project zip. Adzuna keys are free and easy to reissue — treat those
two values as already-leaked and get fresh ones at
https://developer.adzuna.com/ before deploying this anywhere public.

This module reads credentials in this priority order:
    1. Streamlit secrets  -> st.secrets["ADZUNA_APP_ID"] / ["ADZUNA_APP_KEY"]
    2. Environment vars    -> ADZUNA_APP_ID / ADZUNA_APP_KEY
    3. (fallback)           -> the original hardcoded demo keys, so the app
                               still runs out of the box for local testing.
Set up option 1 or 2 with your own keys before sharing the app further.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import requests

from . import config

# Last-resort fallback so the app still runs before you've configured your
# own keys. Replace / remove once you have your own Adzuna credentials.
_FALLBACK_APP_ID = "f5a79969"
_FALLBACK_APP_KEY = "642ab555a4cb934eebd6c21b566622c3"


def _get_credentials() -> tuple[str, str]:
    try:
        import streamlit as st
        app_id = st.secrets.get("ADZUNA_APP_ID")
        app_key = st.secrets.get("ADZUNA_APP_KEY")
        if app_id and app_key:
            return app_id, app_key
    except Exception:
        pass  # no secrets.toml configured, or not running inside streamlit

    app_id = os.environ.get("ADZUNA_APP_ID")
    app_key = os.environ.get("ADZUNA_APP_KEY")
    if app_id and app_key:
        return app_id, app_key

    return _FALLBACK_APP_ID, _FALLBACK_APP_KEY


def get_jobs(keyword: str, city: str, page: int = 1,
             results_per_page: int = config.ADZUNA_RESULTS_PER_PAGE) -> pd.DataFrame:
    """Single-page job search — same behaviour as the original adzuna_api.py."""
    app_id, app_key = _get_credentials()
    url = f"https://api.adzuna.com/v1/api/jobs/{config.ADZUNA_COUNTRY}/search/{page}"
    params = {
        "app_id": app_id,
        "app_key": app_key,
        "what": keyword,
        "where": city,
        "results_per_page": results_per_page,
        "content-type": "application/json",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
    except requests.exceptions.RequestException as exc:
        print(f"[adzuna_client] request failed: {exc}")
        return pd.DataFrame()

    data = response.json()
    jobs = data.get("results", [])

    clean_jobs = []
    for job in jobs:
        clean_jobs.append({
            "id": job.get("id"),
            "title": job.get("title"),
            "company": job.get("company", {}).get("display_name"),
            "location": job.get("location", {}).get("display_name"),
            "salary_min": job.get("salary_min"),
            "salary_max": job.get("salary_max"),
            "salary_is_predicted": job.get("salary_is_predicted"),
            "latitude": job.get("latitude"),
            "longitude": job.get("longitude"),
            "description": job.get("description"),
            "category": job.get("category", {}).get("label"),
            "category_tag": job.get("category", {}).get("tag"),
            "contract_type": job.get("contract_type"),
            "contract_time": job.get("contract_time"),
            "created": job.get("created"),
            "url": job.get("redirect_url"),
        })

    return pd.DataFrame(clean_jobs)


def get_jobs_multi_page(keyword: str, city: str,
                         max_pages: int = config.ADZUNA_MAX_PAGES,
                         results_per_page: int = config.ADZUNA_RESULTS_PER_PAGE) -> pd.DataFrame:
    """Pulls several pages and de-duplicates, for a wider candidate pool
    before the Recommendation Engine ranks and trims it down."""
    frames = []
    for page in range(1, max_pages + 1):
        page_df = get_jobs(keyword, city, page=page, results_per_page=results_per_page)
        if page_df.empty:
            break
        frames.append(page_df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    if "id" in combined.columns:
        combined = combined.drop_duplicates(subset="id")
    return combined


def build_search_keyword(predicted_occupation: Optional[str],
                          previous_occupation: Optional[str],
                          user_override: Optional[str] = None) -> str:
    """Decides what to actually type into Adzuna's `what=` parameter."""
    if user_override and user_override.strip():
        return user_override.strip()
    if previous_occupation and str(previous_occupation).strip().lower() not in ("", "none", "nan"):
        return str(previous_occupation).strip()
    if predicted_occupation and predicted_occupation != "Not Employed":
        return str(predicted_occupation)
    return "entry level"
