"""
Housing and Commute Recommendation Engine

Receives jobs which are already ranked by the existing job recommendation engine and scores housing using affordability, commute distance, bedroom preference,
preferred city, and the existing job score.

Commute distance defaults to straight-line (haversine) so this module works fully offline.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from .commute_distance import (
    STRAIGHT_LINE,
    StraightLineDistanceProvider,
    haversine_distance,
    straight_line_commute,
)
from .housing_explanation import generate_housing_explanation
from .land_bounds import filter_land_safe_records


DEFAULT_HOUSING_WEIGHTS = {
    "affordability": 0.45,
    "commute": 0.30,
    "bedroom": 0.15,
    "city": 0.10,
}

DEFAULT_COMBINED_WEIGHTS = {
    "job": 0.55,
    "housing": 0.45,
}


def clean_text(value: Any) -> str:
    """Convert text to lowercase and remove special characters."""
    if pd.isna(value):
        return ""

    text = str(value).lower()
    text = re.sub(r"[^a-z0-9 ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def city_is_preferred(city: Any, preferred_cities: list[str]) -> float:
    """Return 1.0 if the city matches one of the preferred cities."""
    cleaned_city = clean_text(city)
    matches = any(clean_text(item) == cleaned_city for item in preferred_cities)
    return 1.0 if matches else 0.0


def _resolve_coordinates(record: dict[str, Any]) -> tuple[float, float]:
    """Read lat/lon from a job or housing record, accepting either lat/lon or latitude/longitude column names. Skips missing or NaN values."""
    lat = next(
        (record[key] for key in ("lat", "latitude") if key in record and pd.notna(record[key])),
        None,
    )
    lon = next(
        (record[key] for key in ("lon", "longitude") if key in record and pd.notna(record[key])),
        None,
    )

    if lat is None or lon is None:
        raise ValueError("Record must contain a numeric lat/lon or latitude/longitude.")

    return float(lat), float(lon)


def get_job_salary(job: dict[str, Any], profile: dict[str, Any]) -> float:
    """Resolve annual salary from job salary fields or predicted income."""
    salary = pd.to_numeric(job.get("salary"), errors="coerce")
    if pd.notna(salary) and salary > 0:
        return float(salary)

    salary_min = pd.to_numeric(job.get("salary_min"), errors="coerce")
    salary_max = pd.to_numeric(job.get("salary_max"), errors="coerce")

    values = [
        float(value)
        for value in (salary_min, salary_max)
        if pd.notna(value) and value > 0
    ]
    if values:
        return sum(values) / len(values)

    predicted_income = pd.to_numeric(profile.get("predicted_income"), errors="coerce")
    if pd.notna(predicted_income) and predicted_income > 0:
        return float(predicted_income)

    minimum_salary = pd.to_numeric(profile.get("minimum_salary"), errors="coerce")
    if pd.notna(minimum_salary) and minimum_salary > 0:
        return float(minimum_salary)

    raise ValueError(
        "Provide job salary, salary_min/salary_max, predicted_income, or minimum_salary."
    )


def max_affordable_rent(annual_salary: float, profile: dict[str, Any]) -> float:
    """Calculate maximum affordable monthly rent."""
    rent_ratio = float(profile.get("max_rent_income_ratio", 0.30))
    if not 0 < rent_ratio <= 1:
        raise ValueError("max_rent_income_ratio must be between 0 and 1.")

    if annual_salary <= 0:
        raise ValueError("Annual salary must be greater than zero.")

    return annual_salary / 12 * rent_ratio


def calculate_affordability_score(monthly_rent: float, affordable_rent: float) -> float:
    """Calculate affordability score between 0 and 1.

    Homes at or below the affordable limit score at least 0.60, rising asrent drops further below the limit. 
    Homes above the limit lose score quickly and hit 0 once rent is 50% over budget.
    """
    if monthly_rent <= 0 or affordable_rent <= 0:
        return 0.0

    rent_ratio = monthly_rent / affordable_rent

    if rent_ratio <= 1:
        score = 0.60 + 0.40 * (1 - rent_ratio)
    else:
        score = 1.0 - 2.0 * (rent_ratio - 1)

    return max(0.0, min(1.0, score))


def _validate_weights(weights: dict[str, float], expected_keys: set[str], name: str) -> None:
    """Check that a weights dict has exactly the expected keys and sums to 1."""
    if set(weights) != expected_keys:
        raise ValueError(f"{name} must contain exactly: {sorted(expected_keys)}")

    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"{name} must add up to 1.0 (currently {total:.4f}).")


def score_housing(
    profile: dict[str, Any],
    job: dict[str, Any],
    home: dict[str, Any],
    housing_weights: dict[str, float] | None = None,
    combined_weights: dict[str, float] | None = None,
    commute: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Score one housing option for one already-ranked job.
    """
    housing_weights = housing_weights or DEFAULT_HOUSING_WEIGHTS
    combined_weights = combined_weights or DEFAULT_COMBINED_WEIGHTS

    _validate_weights(housing_weights, {"affordability", "commute", "bedroom", "city"}, "housing_weights")
    _validate_weights(combined_weights, {"job", "housing"}, "combined_weights")

    job_lat, job_lon = _resolve_coordinates(job)
    home_lat, home_lon = _resolve_coordinates(home)

    monthly_rent = home.get("monthly_rent", home.get("rent_price"))
    if monthly_rent is None:
        raise ValueError("Housing record must contain monthly_rent or rent_price.")
    monthly_rent = float(monthly_rent)

    annual_salary = get_job_salary(job, profile)
    affordable_rent = max_affordable_rent(annual_salary, profile)

    if commute is None:
        commute = straight_line_commute((home_lat, home_lon), (job_lat, job_lon))

    commute_km = float(commute["distance_km"])
    commute_minutes = commute.get("duration_minutes")
    distance_source = commute.get("source", STRAIGHT_LINE)

    affordability_score = calculate_affordability_score(monthly_rent, affordable_rent)

    max_commute_km = float(profile.get("max_commute_km", 30))
    if max_commute_km <= 0:
        raise ValueError("max_commute_km must be greater than zero.")

    commute_score = max(0.0, min(1.0, 1 - commute_km / max_commute_km))

    preferred_bedroom = clean_text(profile.get("bedroom_type"))
    home_bedroom = clean_text(home.get("bedroom_type"))
    bedroom_score = 1.0 if preferred_bedroom and preferred_bedroom == home_bedroom else 0.0

    city_score = city_is_preferred(home.get("city"), profile.get("preferred_cities", []))

    score_breakdown = {
        "affordability": round(affordability_score, 4),
        "commute": round(commute_score, 4),
        "bedroom": bedroom_score,
        "city": city_score,
    }

    housing_score = 100 * sum(
        score_breakdown[key] * housing_weights[key] for key in housing_weights
    )

    job_score = pd.to_numeric(job.get("job_score", job.get("match_score", 0)), errors="coerce")
    job_score = 0.0 if pd.isna(job_score) else float(job_score)
    if 0 <= job_score <= 1:
        job_score *= 100

    combined_score = combined_weights["job"] * job_score + combined_weights["housing"] * housing_score

    if commute_km <= max_commute_km * 0.50:
        commute_category = "Short"
    elif commute_km <= max_commute_km:
        commute_category = "Acceptable"
    elif commute_km <= max_commute_km * 1.50:
        commute_category = "Long"
    else:
        commute_category = "Very Long"

    if housing_score >= 80:
        housing_match_level = "Excellent Match"
    elif housing_score >= 65:
        housing_match_level = "Good Match"
    elif housing_score >= 50:
        housing_match_level = "Moderate Match"
    else:
        housing_match_level = "Low Match"

    result = {
        **home,
        "lat": home_lat,
        "lon": home_lon,
        "monthly_rent": round(monthly_rent, 2),
        "job_title": job.get("title", ""),
        "job_company": job.get("company", ""),
        "job_location": job.get("location", ""),
        "job_salary": round(annual_salary, 2),
        "job_score": round(job_score, 2),
        "job_lat": job_lat,
        "job_lon": job_lon,
        "max_affordable_rent": round(affordable_rent, 2),
        "rent_difference": round(affordable_rent - monthly_rent, 2),
        "affordable": monthly_rent <= affordable_rent,
        "commute_km": round(commute_km, 2),
        "commute_minutes": round(commute_minutes, 1) if commute_minutes is not None else None,
        "distance_source": distance_source,
        "commute_category": commute_category,
        "score_breakdown": score_breakdown,
        "affordability_score": round(affordability_score, 4),
        "commute_score": round(commute_score, 4),
        "bedroom_score": bedroom_score,
        "city_score": city_score,
        "housing_score": round(housing_score, 2),
        "housing_match_level": housing_match_level,
        "combined_score": round(combined_score, 2),
    }

    result["explanation"] = generate_housing_explanation(
        profile=profile,
        job=job,
        home=result,
        score_breakdown=score_breakdown,
    )

    return result


def recommend_housing(
    profile: dict[str, Any],
    ranked_jobs: pd.DataFrame | list[dict[str, Any]],
    housing: pd.DataFrame | list[dict[str, Any]],
    top_n_per_job: int = 3,
    top_jobs: int | None = None,
    affordable_only: bool = False,
    housing_weights: dict[str, float] | None = None,
    combined_weights: dict[str, float] | None = None,
    distance_provider: Any | None = None,
) -> pd.DataFrame:
    """Return the best housing options for each already-ranked job.  
    """
    if top_n_per_job <= 0:
        raise ValueError("top_n_per_job must be greater than zero.")

    job_records = (
        ranked_jobs.to_dict("records") if isinstance(ranked_jobs, pd.DataFrame) else list(ranked_jobs)
    )
    housing_records = (
        housing.to_dict("records") if isinstance(housing, pd.DataFrame) else list(housing)
    )

    # Drop housing candidates whose coordinates land in water (lakes,harbours, rivers) rather than the city they claim to be in, 
    # before they ever reach scoring or the map.
    housing_records = filter_land_safe_records(housing_records)

    if top_jobs is not None:
        if top_jobs <= 0:
            raise ValueError("top_jobs must be greater than zero.")
        job_records = job_records[:top_jobs]

    if not job_records or not housing_records:
        return pd.DataFrame()

    distance_provider = distance_provider or StraightLineDistanceProvider()

    home_coordinates = [_resolve_coordinates(home) for home in housing_records]

    recommendations = []

    for job in job_records:
        commutes = distance_provider.distances(_resolve_coordinates(job), home_coordinates)

        scored_homes = []

        for home, commute in zip(housing_records, commutes):
            result = score_housing(
                profile=profile,
                job=job,
                home=home,
                housing_weights=housing_weights,
                combined_weights=combined_weights,
                commute=commute,
            )

            if affordable_only and not result["affordable"]:
                continue

            scored_homes.append(result)

        scored_homes.sort(key=lambda item: item["combined_score"], reverse=True)
        recommendations.extend(scored_homes[:top_n_per_job])

    recommendations.sort(key=lambda item: item["combined_score"], reverse=True)

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    test_profile = {
        "predicted_income": 75000,
        "preferred_cities": ["Toronto", "Mississauga"],
        "bedroom_type": "1 Bedroom",
        "max_rent_income_ratio": 0.30,
        "max_commute_km": 30,
    }

    test_job = {
        "title": "Data Analyst",
        "company": "ABC Company",
        "location": "Toronto",
        "salary_min": 70000,
        "salary_max": 80000,
        "lat": 43.6532,
        "lon": -79.3832,
        "match_score": 85,
    }

    test_home = {
        "city": "Toronto",
        "neighbourhood": "Etobicoke",
        "bedroom_type": "1 Bedroom",
        "rent_price": 1800,
        "lat": 43.6205,
        "lon": -79.5132,
    }

    test_result = score_housing(test_profile, test_job, test_home)

    print("\nHousing Recommendation Test")
    print("-" * 40)
    print("Job:", test_result["job_title"])
    print("Housing:", test_result["neighbourhood"])
    print("Monthly rent:", test_result["monthly_rent"])
    print("Maximum affordable rent:", test_result["max_affordable_rent"])
    print("Affordable:", test_result["affordable"])
    print("Commute:", test_result["commute_km"], "km")
    print("Housing score:", test_result["housing_score"])
    print("Combined score:", test_result["combined_score"])

    print("\nExplanation:")
    for reason in test_result["explanation"]:
        print("-", reason)