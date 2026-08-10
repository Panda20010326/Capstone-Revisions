"""Karthika recommendation engine converted from Recommendation_Engine_Karthika.ipynb.

Runtime/inference module only. The original notebook's hard-coded profile and
file-loading cells were removed so Streamlit can pass its live predictions and
DataFrames into the same recommendation logic.
"""
from __future__ import annotations

import re
from math import radians, sin, cos, sqrt, atan2
from typing import Any

import pandas as pd
# Optional GIS support used to reject housing points that fall in water.
# The module still runs if GeoPandas/Shapely are not installed.
try:
    import geopandas as gpd
    from shapely.geometry import Point
                                                    
except ImportError:
    gpd = None
    Point = None

def clean_text(text: Any) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def count_matching_keywords(text: Any, keyword_list: list[str]) -> float:
    if not keyword_list:
        return 0.0
    cleaned_text = clean_text(text)
    valid = [clean_text(k).strip() for k in keyword_list if clean_text(k).strip()]
    if not valid:
        return 0.0
    matches = 0
    for keyword in valid:
        if re.search(r"\b" + re.escape(keyword) + r"\b", cleaned_text):
            matches += 1
    return matches / len(valid)


def city_is_preferred(location: Any, preferred_cities: list[str]) -> float:
    location_text = clean_text(location)
    for city in preferred_cities:
        city_text = clean_text(city)
        if city_text and city_text in location_text:
            return 1.0
    return 0.0


def score_salary(salary: Any, minimum_salary: float) -> float:
    salary = pd.to_numeric(salary, errors="coerce")
    if pd.isna(salary) or salary <= 0:
        return 0.35
    if salary >= minimum_salary:
        return 1.0
    return float(salary / minimum_salary)


def _job_salary(job: dict[str, Any]) -> float | None:
    salary = pd.to_numeric(job.get("salary"), errors="coerce")
    if pd.notna(salary) and salary > 0:
        return float(salary)
    vals = []
    for key in ("salary_min", "salary_max"):
        value = pd.to_numeric(job.get(key), errors="coerce")
        if pd.notna(value) and value > 0:
            vals.append(float(value))
    return sum(vals) / len(vals) if vals else None


def _resolve_job_location(row: pd.Series) -> str:
    return str(row.get("location", row.get("city", "")))


def rank_jobs(jobs_df: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    """Rank jobs using the scoring logic from Karthika's notebook."""
    if jobs_df.empty:
        return jobs_df.copy()

    jobs = jobs_df.copy()
    if "salary" not in jobs.columns:
        jobs["salary"] = jobs.apply(_job_salary, axis=1)

    preferred_cities = profile.get("preferred_cities", [])
    if not preferred_cities and profile.get("preferred_city"):
        preferred_cities = [profile["preferred_city"]]

    profile_terms = [
        profile.get("occupation_category", ""),
        profile.get("previous_occupation", ""),
        profile.get("field_of_study", ""),
    ] + list(profile.get("additional_skills", profile.get("skills", [])) or [])
    profile_terms = [str(t) for t in profile_terms if str(t).strip()]

    employment_factor = float(profile.get("employment_probability", 1.0))
    profile_fit_score = float(profile.get("profile_fit_score", 0.5))
    minimum_salary = float(profile.get("minimum_salary", 0) or 0)

    scores = []
    for _, job in jobs.iterrows():
        job_text = " ".join([
            str(job.get("title", "")),
            str(job.get("description", "")),
            str(job.get("category", "")),
        ])
        background_score = count_matching_keywords(job_text, profile_terms)
        title_score = count_matching_keywords(
            str(job.get("title", "")),
            [str(profile.get("previous_occupation", ""))],
        )
        salary_score = score_salary(job.get("salary"), minimum_salary) if minimum_salary > 0 else 0.35
        city_score = city_is_preferred(_resolve_job_location(job), preferred_cities)

        content_score = (
            0.35 * background_score
            + 0.20 * title_score
            + 0.20 * salary_score
            + 0.10 * city_score
            + 0.15 * profile_fit_score
        )
        scores.append(100.0 * content_score * employment_factor)

    jobs["karthika_job_score"] = scores
    jobs["match_score"] = jobs["karthika_job_score"]
    return jobs.sort_values("karthika_job_score", ascending=False).reset_index(drop=True)


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    earth_radius_km = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1
    a = sin(delta_lat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(delta_lon / 2) ** 2
    return earth_radius_km * 2 * atan2(sqrt(a), sqrt(1 - a))


def _valid_coordinates(lat: Any, lon: Any) -> bool:
    lat = pd.to_numeric(lat, errors="coerce")
    lon = pd.to_numeric(lon, errors="coerce")
    if pd.isna(lat) or pd.isna(lon):
        return False
    return 40.0 <= float(lat) <= 60.0 and -95.0 <= float(lon) <= -70.0


# Conservative land envelopes plus a Toronto shoreline guard.
# These are safeguards for the geocoded CMHC points, not a replacement for a
# real GIS boundary dataset. The Toronto shoreline guard is deliberately
# conservative because the supplied Toronto housing file contains several
# synthetic points south of the actual mainland.
CITY_LAND_BOUNDS = {
    "toronto": (43.59, 43.82, -79.65, -79.12),
    "mississauga": (43.50, 43.75, -79.82, -79.42),
    "burlington": (43.30, 43.50, -80.10, -79.65),
    "hamilton": (43.15, 43.38, -80.15, -79.65),
    "oakville": (43.38, 43.55, -79.82, -79.48),
    "ottawa": (45.20, 45.65, -76.10, -75.40),
    "oshawa": (43.85, 44.05, -79.10, -78.65),
    "barrie": (44.25, 44.55, -79.90, -79.45),
    "kingston": (44.19, 44.40, -76.75, -76.20),
    "belleville": (44.12, 44.32, -77.65, -77.15),
    "windsor": (42.24, 42.36, -83.15, -82.95),
    "thunder bay": (48.15, 48.60, -89.55, -89.15),
    "st. catharines": (43.00, 43.22, -79.45, -79.00),
    "brantford": (43.00, 43.28, -80.45, -80.05),
    "greater sudbury": (46.30, 46.70, -81.30, -80.70),
    "guelph": (43.38, 43.70, -80.45, -80.05),
    "kitchener waterloo": (43.28, 43.62, -80.72, -80.28),
    "london": (42.80, 43.18, -81.50, -81.00),
    "peterborough": (44.12, 44.48, -78.55, -78.05),
}


def normalize_city(value: Any) -> str:
    text = clean_text(value)
    aliases = {
        "city of toronto": "toronto",
        "toronto ontario": "toronto",
        "mississauga ontario": "mississauga",
        "hamilton ontario": "hamilton",
        "ottawa ontario": "ottawa",
        "burlington ontario": "burlington",
        "oakville ontario": "oakville",
        "brantford ontario": "brantford",
        "greater sudbury ontario": "greater sudbury",
        "sudbury": "greater sudbury",
        "sudbury ontario": "greater sudbury",
        "guelph ontario": "guelph",
        "kitchener-waterloo": "kitchener waterloo",
        "kitchener waterloo ontario": "kitchener waterloo",
        "kitchener": "kitchener waterloo",
        "waterloo": "kitchener waterloo",
        "london ontario": "london",
        "peterborough ontario": "peterborough",
        "st catharines": "st. catharines",
        "st catharines ontario": "st. catharines",
        "st. catharines ontario": "st. catharines",
        "thunder bay ontario": "thunder bay",
        "windsor ontario": "windsor",
        "oshawa ontario": "oshawa",
        "barrie ontario": "barrie",
        "kingston ontario": "kingston",
        "belleville ontario": "belleville",
    }
    return aliases.get(text, text)


def _toronto_shoreline_lat(lon: float) -> float:
    # Approximate mainland shoreline. Points below this line are treated as
    # offshore for this synthetic CMHC Toronto dataset.
    anchors = [
        (-79.65, 43.605),
        (-79.55, 43.605),
        (-79.48, 43.610),
        (-79.43, 43.615),
        (-79.38, 43.625),
        (-79.33, 43.625),
        (-79.28, 43.635),
        (-79.22, 43.650),
        (-79.12, 43.665),
    ]
    if lon <= anchors[0][0]:
        return anchors[0][1]
    if lon >= anchors[-1][0]:
        return anchors[-1][1]
    for (lon1, lat1), (lon2, lat2) in zip(anchors, anchors[1:]):
        if lon1 <= lon <= lon2:
            fraction = (lon - lon1) / (lon2 - lon1)
            return lat1 + fraction * (lat2 - lat1)
    return 43.62


def _load_canada_land_geometry() -> Any:
    """
    Load Natural Earth country geometry through GeoPandas when available.
    The result is cached on the function so the shapefile is not reloaded
    for every housing point.
    """
    if hasattr(_load_canada_land_geometry, "_cached"):
        return _load_canada_land_geometry._cached

    if gpd is None:
        _load_canada_land_geometry._cached = None
        return None

    try:
        # GeoPandas provides a built-in Natural Earth low-resolution dataset
        # in many common installations. If unavailable, we fall back safely.
        dataset_path = gpd.datasets.get_path("naturalearth_lowres")
        world = gpd.read_file(dataset_path)
        canada = world[world["name"].astype(str).str.lower().eq("canada")]

        if canada.empty:
            _load_canada_land_geometry._cached = None
            return None

        geom = canada.geometry.unary_union
        _load_canada_land_geometry._cached = geom
        return geom
    except Exception:
        _load_canada_land_geometry._cached = None
        return None

def _point_is_on_canadian_land(lat: float, lon: float) -> bool:
    """
    Return True when the coordinate is on Canadian land.

                                                                         
                                                                            
    If GIS support is unavailable, the existing city bounds / shoreline
    safeguards remain the fallback.
    """
    if not _valid_coordinates(lat, lon):
        return False

    canada_geom = _load_canada_land_geometry()

    if canada_geom is None or Point is None:
            
                                                 
        return True
                         
                

    try:
        point = Point(float(lon), float(lat))
        return bool(canada_geom.contains(point) or canada_geom.touches(point))
    except Exception:
        return True


def _hamilton_harbour_guard(lat: float, lon: float) -> bool:
    """
    Extra safeguard for Hamilton Harbour / western Lake Ontario candidate points.
    """
    if 43.275 <= lat <= 43.325 and -79.885 <= lon <= -79.755:
        return False

    if 43.305 <= lat <= 43.365 and -79.790 <= lon <= -79.675:
        return False

    return True

def _looks_like_land(city: Any, lat: float, lon: float) -> bool:
    """
    Validate a housing coordinate before it is used by the recommender/map.
    """
    if not _valid_coordinates(lat, lon):
        return False

    city_key = normalize_city(city)

    bounds = CITY_LAND_BOUNDS.get(city_key)
    if bounds is not None:
        min_lat, max_lat, min_lon, max_lon = bounds
        if not (min_lat <= lat <= max_lat and min_lon <= lon <= max_lon):
            return False

    if city_key == "toronto" and lat < _toronto_shoreline_lat(lon):
        return False

    if city_key == "hamilton" and not _hamilton_harbour_guard(lat, lon):
        return False

    if not _point_is_on_canadian_land(lat, lon):
        return False

    return True
def prepare_housing_data(
    housing_df: pd.DataFrame,
    target_city: str | None = None,
    strict_target_city: bool = False,
) -> pd.DataFrame:
    """Normalize coordinates and remove invalid/offshore points.

    By default, housing from all available Ontario centres is retained so the
    recommender can compare a selected job with more affordable nearby cities.
    Set ``strict_target_city=True`` when the application should show housing
    only in the selected city.
    """
    if housing_df.empty:
        return housing_df.copy()

    housing = housing_df.copy()
    if "lat" not in housing.columns and "latitude" in housing.columns:
        housing["lat"] = housing["latitude"]
    if "lon" not in housing.columns and "longitude" in housing.columns:
        housing["lon"] = housing["longitude"]
    if "monthly_rent" not in housing.columns and "rent_price" in housing.columns:
        housing["monthly_rent"] = housing["rent_price"]

    housing["lat"] = pd.to_numeric(housing["lat"], errors="coerce")
    housing["lon"] = pd.to_numeric(housing["lon"], errors="coerce")
    housing = housing.dropna(subset=["lat", "lon"])
    housing = housing[
        housing.apply(
            lambda r: _valid_coordinates(r["lat"], r["lon"]),
            axis=1,
        )
    ]
    housing = housing[
        housing.apply(
            lambda r: _looks_like_land(r.get("city", ""), float(r["lat"]), float(r["lon"])),
            axis=1,
        )
    ]

    # City-only mode is still available, but Ontario-wide comparison is the
    # default. The commute filter later decides which other centres are
    # realistically close enough to the selected job.
    if strict_target_city and target_city and "city" in housing.columns:
        target = normalize_city(target_city)
        city_mask = housing["city"].map(normalize_city).eq(target)
        city_housing = housing[city_mask].copy()
        if not city_housing.empty:
            housing = city_housing

    return housing.reset_index(drop=True)


def filter_housing_near_jobs(
    housing_df: pd.DataFrame,
    ranked_jobs: pd.DataFrame,
    max_commute_km: float = 75.0,
    target_city: str | None = None,
) -> pd.DataFrame:
    """Keep only housing points that are realistically close to one of the ranked jobs."""
    if housing_df.empty or ranked_jobs.empty:
        return housing_df.iloc[0:0].copy()

    # default to Ontario-wide comparison; do not enforce strict city-only filtering here
    homes = prepare_housing_data(housing_df, target_city, False)
    if homes.empty:
        return homes

    job_points = []
    for _, job in ranked_jobs.iterrows():
        lat = job.get("latitude", job.get("lat"))
        lon = job.get("longitude", job.get("lon"))
        if _valid_coordinates(lat, lon):
            job_points.append((float(lat), float(lon)))

    if not job_points:
        return homes.iloc[0:0].copy()

    def near_job(row: pd.Series) -> bool:
        return min(
            haversine_distance(float(row["lat"]), float(row["lon"]), jlat, jlon)
            for jlat, jlon in job_points
        ) <= max_commute_km

    return homes.loc[homes.apply(near_job, axis=1)].reset_index(drop=True)


def max_affordable_rent(annual_salary: float, profile: dict[str, Any]) -> float:
    ratio = float(profile.get("max_rent_income_ratio", 0.30))
    salary = float(annual_salary or 0)
    if salary <= 0:
        salary = float(profile.get("predicted_income", profile.get("minimum_salary", 65000)) or 65000)
    return salary / 12.0 * ratio


def rank_housing_for_job(housing_df: pd.DataFrame, job: pd.Series, profile: dict[str, Any]) -> pd.DataFrame:
    """Housing ranking adapted directly from Karthika's notebook."""
    if housing_df.empty:
        return housing_df.copy()

    city_scope = clean_text(profile.get("housing_city_scope", "ontario"))
    strict_city = city_scope in {"preferred city", "selected city", "city only", "preferred_city", "selected_city", "city_only"}
    housing = prepare_housing_data(
        housing_df,
        profile.get("preferred_city"),
        strict_target_city=strict_city,
    )
    job_lat = pd.to_numeric(job.get("latitude", job.get("lat")), errors="coerce")
    job_lon = pd.to_numeric(job.get("longitude", job.get("lon")), errors="coerce")
    if pd.isna(job_lat) or pd.isna(job_lon):
        return housing.iloc[0:0].copy()

    salary = _job_salary(job)
    if salary is None:
        salary = float(profile.get("predicted_income", profile.get("minimum_salary", 65000)) or 65000)
    affordable_rent = max_affordable_rent(salary, profile)
    default_commute_km = 30.0 if strict_city else 75.0
    max_commute = float(profile.get("max_commute_km", default_commute_km))
    preferred_cities = profile.get("preferred_cities", [])
    bedroom_type = clean_text(profile.get("bedroom_type", "2 Bedroom"))

    rows = []
    for _, home in housing.iterrows():
        distance = haversine_distance(float(home["lat"]), float(home["lon"]), float(job_lat), float(job_lon))
        if distance > max_commute:
            continue

        rent = pd.to_numeric(home.get("monthly_rent"), errors="coerce")
        if pd.isna(rent) or rent <= 0:
            continue

        monthly_income = float(salary) / 12.0 if float(salary) > 0 else 0.0
        rent_to_income_ratio = float(rent) / monthly_income if monthly_income > 0 else float("inf")

        if rent <= affordable_rent:
            affordability = 1.0 - (float(rent) / affordable_rent)
            affordability = max(0.0, min(1.0, affordability))
        else:
            affordability = max(0.0, min(1.0, (affordable_rent / float(rent)) * 0.5))

        commute_score = max(0.0, min(1.0, 1.0 - distance / max_commute))
        bedroom_score = 1.0 if bedroom_type and clean_text(home.get("bedroom_type")) == bedroom_type else 0.0
        city_score = city_is_preferred(home.get("city", ""), preferred_cities)

        housing_score = 100.0 * (
            0.45 * affordability
            + 0.30 * commute_score
            + 0.15 * bedroom_score
            + 0.10 * city_score
        )
        job_score = float(job.get("karthika_job_score", job.get("match_score", 0)) or 0)
        combined_score = 0.55 * job_score + 0.45 * housing_score

        result = home.to_dict()
        result.update({
            "job_title": str(job.get("title", "")),
            "job_company": str(job.get("company", "")),
            "job_location": str(job.get("location", "")),
            "job_lat": float(job_lat),
            "job_lon": float(job_lon),
            "job_salary": float(salary),
            "job_score": job_score,
            "max_affordable_rent": float(affordable_rent),
            "monthly_income": round(monthly_income, 2),
            "rent_to_income_ratio": round(rent_to_income_ratio, 4),
            "rent_to_income_pct": round(rent_to_income_ratio * 100.0, 1),
            "monthly_rent_gap": round(float(affordable_rent) - float(rent), 2),
            "affordability_status": "Affordable" if float(rent) <= affordable_rent else "Over 30% threshold",
            "commute_km": round(float(distance), 2),
            "commute_method": "Haversine straight-line distance",
            "affordable": bool(float(rent) <= affordable_rent),
            "housing_score": round(housing_score, 2),
            "combined_score": round(combined_score, 2),
        })
        rows.append(result)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("combined_score", ascending=False).head(
        int(profile.get("homes_per_job", 5))
    ).reset_index(drop=True)

def get_available_housing_cities(housing_df: pd.DataFrame) -> list[str]:
    """Return the Ontario housing centres available in the supplied dataset."""
    if housing_df.empty or "city" not in housing_df.columns:
        return []
    cities = housing_df["city"].dropna().astype(str).str.strip()
    return sorted(c for c in cities.unique().tolist() if c)

def build_housing_recommendations(
    housing_df: pd.DataFrame,
    ranked_jobs: pd.DataFrame,
    profile: dict[str, Any],
) -> pd.DataFrame:
    """Return nearby housing recommendations for the Karthika-ranked jobs."""
    if housing_df.empty or ranked_jobs.empty:
        return pd.DataFrame()

    all_matches = []
    for _, job in ranked_jobs.head(int(profile.get("top_jobs", 10))).iterrows():
        matches = rank_housing_for_job(housing_df, job, profile)
        if not matches.empty:
            all_matches.append(matches)

    if not all_matches:
        return pd.DataFrame()
    return pd.concat(all_matches, ignore_index=True).sort_values(
        "combined_score", ascending=False
    ).reset_index(drop=True)
