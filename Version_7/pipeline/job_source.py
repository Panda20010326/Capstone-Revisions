from __future__ import annotations
import pandas as pd
from . import config


def get_jobs_from_dataset(keyword: str, city: str) -> pd.DataFrame:
    df = pd.read_csv(config.LOCAL_JOBS_DATASET_PATH)
    if df.empty:
        return df

    text_cols = [c for c in ["title", "description", "category", "location"] if c in df.columns]
    text = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).str.lower()
    query = str(keyword).strip().lower()
    city_q = str(city).strip().lower()

    keyword_mask = text.str.contains(query, regex=False) if query else pd.Series(True, index=df.index)
    city_mask = df.get("location", pd.Series("", index=df.index)).fillna("").astype(str).str.lower().str.contains(city_q, regex=False) if city_q else pd.Series(True, index=df.index)

    filtered = df[keyword_mask & city_mask].copy()
    if filtered.empty:
        filtered = df[keyword_mask].copy()
    if filtered.empty:
        filtered = df.copy()

    return filtered.head(250)


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
