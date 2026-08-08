"""
Fetch Adzuna job postings for additional major Ontario cities and merge
them into processed_adzuna_jobs.csv.

"""

from __future__ import annotations

import time
import pandas as pd

from adzuna_client import get_jobs_multi_page

# ---- Config -----------------------------------------------------------

CITIES = [
    "Mississauga, Ontario",
    "Ottawa, Ontario",
    "Hamilton, Ontario",
    "Brampton, Ontario",
    "Kitchener, Ontario",
    "London, Ontario",
    "Windsor, Ontario",
    "Waterloo, Ontario",
    "Markham, Ontario",
]

# newcomer settlement platform where users come from varied occupations.
KEYWORDS = [
    "Data Analyst",
    "Software Developer",
    "Customer Service Representative",
    "Registered Nurse",
    "Accountant",
    "Warehouse Associate",
    "Administrative Assistant",
    "Marketing Coordinator",
    "Truck Driver",
    "Personal Support Worker",
]

PAGES_PER_QUERY = 2   
RESULTS_PER_PAGE = 20
SLEEP_BETWEEN_CALLS = 1.0 
EXISTING_CSV = "processed_adzuna_jobs.csv"
OUTPUT_CSV = "processed_adzuna_jobs_expanded.csv"

# ---- Fetch --------------------------------------------------------------

def fetch_all() -> pd.DataFrame:
    all_frames = []
    total_queries = len(CITIES) * len(KEYWORDS)
    done = 0

    for city in CITIES:
        for keyword in KEYWORDS:
            done += 1
            print(f"[{done}/{total_queries}] {keyword!r} in {city!r} ...", end=" ")
            try:
                df = get_jobs_multi_page(
                    keyword=keyword,
                    city=city,
                    pages=PAGES_PER_QUERY,
                    results_per_page=RESULTS_PER_PAGE,
                )
                print(f"got {len(df)} rows")
                if not df.empty:
                    all_frames.append(df)
            except Exception as e:
                print(f"FAILED: {e}")
            time.sleep(SLEEP_BETWEEN_CALLS)

    if not all_frames:
        return pd.DataFrame()

    combined = pd.concat(all_frames, ignore_index=True)
    combined = combined.drop_duplicates(subset="id")
    return combined


def merge_with_existing(new_df: pd.DataFrame) -> pd.DataFrame:
    try:
        existing = pd.read_csv(EXISTING_CSV)
    except FileNotFoundError:
        print(f"Note: {EXISTING_CSV} not found locally, saving new rows only.")
        return new_df

    merged = pd.concat([existing, new_df], ignore_index=True)
    merged = merged.drop_duplicates(subset="id")
    return merged


if __name__ == "__main__":
    new_jobs = fetch_all()
    print(f"\nFetched {len(new_jobs)} unique new job rows.")

    final_df = merge_with_existing(new_jobs)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved merged dataset ({len(final_df)} rows total) to {OUTPUT_CSV}")

    if "location" in final_df.columns and not final_df.empty:
        print("\nRows per city in final dataset:")
        print(final_df["location"].value_counts().to_string())
    else:
        print("\nNo location data available to summarize.")
