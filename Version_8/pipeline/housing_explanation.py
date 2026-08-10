"""Housing recommendation explanation utilities."""

from __future__ import annotations

from typing import Any


def generate_housing_explanation(
    profile: dict[str, Any],
    job: dict[str, Any],
    home: dict[str, Any],
    score_breakdown: dict[str, float],
) -> list[str]:
    """Generate plain-language reasons for one housing recommendation."""
    reasons = []

    monthly_rent = float(
        home["monthly_rent"]
    )
    affordable_rent = float(
        home["max_affordable_rent"]
    )
    commute_km = float(
        home["commute_km"]
    )

    if home["affordable"]:
        reasons.append(
            f"The monthly rent of "
            f"${monthly_rent:,.0f} is within the "
            f"estimated affordable limit of "
            f"${affordable_rent:,.0f}."
        )
    else:
        reasons.append(
            f"The monthly rent of "
            f"${monthly_rent:,.0f} is above the "
            f"estimated affordable limit of "
            f"${affordable_rent:,.0f}."
        )

    commute_minutes = home.get("commute_minutes")

    description = (
        "driving"
        if home.get("distance_source") == "road"
        else "approximate straight-line"
    )

    duration_text = (
        f" (about {float(commute_minutes):.0f} minutes by car)"
        if commute_minutes is not None
        else ""
    )

    if (
        commute_km
        <= float(
            profile.get(
                "max_commute_km",
                30,
            )
        )
    ):
        reasons.append(
            f"The {description} commute is "
            f"{commute_km:.1f} km{duration_text} "
            f"and is within the preferred range."
        )
    else:
        reasons.append(
            f"The {description} commute is "
            f"{commute_km:.1f} km{duration_text} "
            f"and is longer than the preferred range."
        )

    if score_breakdown["bedroom"] == 1.0:
        reasons.append(
            "The bedroom type matches the "
            "user's preference."
        )

    if score_breakdown["city"] == 1.0:
        reasons.append(
            f"The housing option is in a "
            f"preferred city: {home.get('city', '')}."
        )

    if not reasons:
        reasons.append(
            "This option received one of the "
            "highest overall housing scores."
        )

    return reasons
