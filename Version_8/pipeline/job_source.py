from __future__ import annotations

import re
import pandas as pd
from . import config


def _norm_city(value: object) -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    aliases = {
        "toronto, ontario": "toronto",
        "city of toronto": "toronto",
        "ottawa, ontario": "ottawa",
        "hamilton, ontario": "hamilton",
        "mississauga, ontario": "mississauga",
        "burlington, ontario": "burlington",
        "oakville, ontario": "oakville",
    }
    return aliases.get(text, text)


def _location_matches_city(location: object, city: object) -> bool:
    location_text = _norm_city(location)
    city_text = _norm_city(city)
    if not city_text:
        return True
    # Match the requested city as a location component, while avoiding
    # accidental substring matches inside unrelated place names.
    parts = {p.strip() for p in re.split(r"[,|;/]", location_text) if p.strip()}
    if city_text in parts:
        return True
    return bool(re.search(rf"\b{re.escape(city_text)}\b", location_text))


def get_jobs_from_dataset(keyword: str, city: str) -> pd.DataFrame:
    """Return local jobs only when the requested city is actually represented.

    The previous implementation fell back to keyword-only Toronto records when a
    city was absent. That made a user's Ottawa/Mississauga/etc. selection silently
    return Toronto jobs. The local dataset is Toronto-heavy, so an unavailable
    city now returns an empty frame; the app can then use Live Adzuna when keys are
    configured.
    """
    df = pd.read_csv(config.LOCAL_JOBS_DATASET_PATH)
    if df.empty:
        return df

    location_series = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str)
    city_mask = location_series.map(lambda value: _location_matches_city(value, city))
    city_jobs = df[city_mask].copy()

    # Never substitute jobs from another city.
    if city_jobs.empty:
        return df.iloc[0:0].copy()

    text_cols = [c for c in ["title", "description", "category", "location"] if c in city_jobs.columns]
    text = city_jobs[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    query = str(keyword or "").strip().lower()

    if query:
        keyword_mask = text.str.contains(query, regex=False)
        filtered = city_jobs[keyword_mask].copy()
        # If the city has jobs but the exact keyword is absent, keep the city's
        # jobs rather than searching another city. Karthika's ranking will score
        # the best matches.
        if filtered.empty:
            filtered = city_jobs.copy()
    else:
        filtered = city_jobs.copy()

    return filtered.head(250)


def dataset_cities() -> list[str]:
    df = pd.read_csv(config.LOCAL_JOBS_DATASET_PATH, usecols=["location"])
    locations = df["location"].dropna().astype(str)
    cities = set()
    for value in locations:
        for part in re.split(r"[,|;/]", value):
            part = part.strip()
            if part:
                cities.add(_norm_city(part).title())
    return sorted(cities)


def dataset_info() -> dict:
    df = pd.read_csv(config.LOCAL_JOBS_DATASET_PATH)
    return {
        "total_jobs": len(df),
        "unique_companies": df["company"].nunique() if "company" in df.columns else 0,
        "unique_locations": df["location"].nunique() if "location" in df.columns else 0,
        "date_min": df["created"].min() if "created" in df.columns else None,
        "date_max": df["created"].max() if "created" in df.columns else None,
        "categories": sorted(df["category"].dropna().astype(str).unique())[:10] if "category" in df.columns else [],
    }
