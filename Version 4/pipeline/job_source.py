"""
job_source.py — Stage 4 of the pipeline, offline mode
--------------------------------------------------------
An alternative to adzuna_client.py: serves jobs from a dataset that was
already pulled from Adzuna and saved to CSV
(originally Employment_Prediction/Datasets/processed_adzuna_jobs.csv,
collected by adzuna_job_analysis.py), instead of calling the live API.

Same output schema as adzuna_client.get_jobs_multi_page() (id, title,
company, location, salary_min, salary_max, latitude, longitude,
description, category, contract_type, contract_time, url, ...), so
app.py and recommend_jobs() don't need to know which source is in use.

Use this when:
    - you don't have (or don't want to spend) Adzuna API quota
    - you want fully reproducible demos/screenshots that don't depend on
      what's currently posted
    - you're offline, or the live API is down / rate-limited

Trade-off: this dataset was collected from ONE search ("data analyst" in
Toronto, run once) rather than a live query per user, so it's necessarily
narrower and gets staler over time. See dataset_info() / the README for
exact numbers. It skews toward IT/data/analyst-style roles in Toronto —
searches for very different occupations or other cities will fall back to
the closest available matches rather than an empty result.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config

_cached_df: pd.DataFrame | None = None


def _load_dataset() -> pd.DataFrame:
    global _cached_df
    if _cached_df is None:
        _cached_df = pd.read_csv(config.LOCAL_JOBS_DATASET_PATH)
    return _cached_df


def _tokenize(text) -> set[str]:
    return set(re.findall(r"[a-zA-Z]+", str(text).lower()))


def get_jobs_from_dataset(keyword: str, city: str | None = None,
                           top_n: int | None = None) -> pd.DataFrame:
    """Filters the local dataset by keyword (title/description/category) and,
    if any rows match, further narrows by city. Never returns empty just
    because nothing matched -- falls back to the broader/full set instead,
    so the recommendation engine always has something to rank and explain.
    """
    df = _load_dataset().copy()
    if df.empty:
        return df

    filtered = df
    keyword_tokens = _tokenize(keyword)
    if keyword_tokens:
        haystack = (
            df["title"].fillna("") + " " +
            df["description"].fillna("") + " " +
            df["category"].fillna("")
        )
        mask = haystack.apply(lambda text: bool(keyword_tokens & _tokenize(text)))
        if mask.any():
            filtered = df[mask]
        # else: keyword matched nothing -- keep the full dataset rather than
        # returning zero rows.

    if city:
        city_tokens = _tokenize(city)
        location_mask = filtered["location"].fillna("").apply(
            lambda text: bool(city_tokens & _tokenize(text))
        )
        if location_mask.any():
            filtered = filtered[location_mask]
        # else: keep the keyword-filtered results even if the city doesn't
        # match -- this dataset only covers the Toronto area, so an
        # out-of-town search would otherwise return nothing.

    if top_n:
        filtered = filtered.head(top_n)

    return filtered.reset_index(drop=True)


def dataset_info() -> dict:
    """Summary stats shown in the UI so users know what they're searching."""
    df = _load_dataset()
    return {
        "total_jobs": len(df),
        "unique_companies": df["company"].nunique() if "company" in df.columns else None,
        "unique_locations": df["location"].nunique() if "location" in df.columns else None,
        "categories": sorted(df["category"].dropna().unique().tolist()) if "category" in df.columns else [],
        "date_min": df["created"].min() if "created" in df.columns else None,
        "date_max": df["created"].max() if "created" in df.columns else None,
    }
