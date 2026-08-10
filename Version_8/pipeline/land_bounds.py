"""
Land-boundary safety checks for housing coordinates.

Synthetic/geocoded housing points can occasionally land in water (lakes,
harbours, rivers) rather than on the city they claim to represent. This
module provides conservative, hand-tuned lat/lon bounding boxes per city,
plus two extra shoreline guards (Toronto, Hamilton) for cities where a
simple rectangle isn't tight enough.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

# (min_lat, max_lat, min_lon, max_lon) per city. Water-facing edges were
# tightened against real shoreline coordinates -- see inline notes.
CITY_LAND_BOUNDS = {
    "toronto": (43.59, 43.82, -79.65, -79.12),
    "mississauga": (43.50, 43.75, -79.82, -79.42),
    "burlington": (43.30, 43.50, -80.10, -79.65),
    # Hamilton: max_lat trimmed to exclude Hamilton Harbour / Lake Ontario
    # at the north end of the city (harbour sits ~43.29N).
    "hamilton": (43.15, 43.28, -80.15, -79.65),
    "oakville": (43.38, 43.55, -79.82, -79.48),
    "ottawa": (45.20, 45.65, -76.10, -75.40),
    # Oshawa: min_lat raised to exclude Lake Ontario to the south.
    "oshawa": (43.85, 44.05, -79.10, -78.65),
    "barrie": (44.25, 44.55, -79.90, -79.45),
    # Kingston: min_lat raised to exclude open Lake Ontario / the
    # St. Lawrence south of the downtown waterfront (~44.23N).
    "kingston": (44.19, 44.40, -76.75, -76.20),
    # Belleville: min_lat raised to exclude the Bay of Quinte.
    "belleville": (44.12, 44.32, -77.65, -77.15),
    # Windsor: tightened on all sides to exclude the Detroit River (west),
    # Lake St. Clair (northeast), and Lake Erie (south).
    "windsor": (42.24, 42.36, -83.15, -82.95),
    # Thunder Bay: max_lon trimmed to exclude Thunder Bay (the lake inlet)
    # to the east of the city.
    "thunder bay": (48.15, 48.60, -89.55, -89.15),
    # St. Catharines: max_lat trimmed to exclude Lake Ontario to the north
    # (Port Dalhousie / Port Weller shoreline).
    "st. catharines": (43.00, 43.22, -79.45, -79.00),
    "brantford": (43.00, 43.28, -80.45, -80.05),
    "greater sudbury": (46.30, 46.70, -81.30, -80.70),
    "guelph": (43.38, 43.70, -80.45, -80.05),
    "kitchener waterloo": (43.28, 43.62, -80.72, -80.28),
    "london": (42.80, 43.18, -81.50, -81.00),
    "peterborough": (44.12, 44.48, -78.55, -78.05),
}

_CITY_ALIASES = {
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


def _clean_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9. ]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_city(value: Any) -> str:
    """Map a raw city string to the canonical key used in CITY_LAND_BOUNDS."""
    text = _clean_text(value)
    return _CITY_ALIASES.get(text, text)


def _valid_coordinates(lat: Any, lon: Any) -> bool:
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    if pd.isna(lat_f) or pd.isna(lon_f):
        return False
    # Broad Canada sanity check.
    return 40.0 <= lat_f <= 60.0 and -95.0 <= lon_f <= -70.0


def _toronto_shoreline_lat(lon: float) -> float:
    """Approximate mainland shoreline. Points below this are offshore."""
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


def _hamilton_harbour_guard(lat: float, lon: float) -> bool:
    """Extra safeguard for Hamilton Harbour / western Lake Ontario points."""
    if 43.275 <= lat <= 43.325 and -79.885 <= lon <= -79.755:
        return False
    if 43.305 <= lat <= 43.365 and -79.790 <= lon <= -79.675:
        return False
    return True


def is_land_safe(city: Any, lat: Any, lon: Any) -> bool:
    """
    Return True when a (city, lat, lon) triple looks like it's on land
    and inside its claimed city, rather than out in a lake or river.
    """
    if not _valid_coordinates(lat, lon):
        return False

    lat_f, lon_f = float(lat), float(lon)
    city_key = normalize_city(city)

    bounds = CITY_LAND_BOUNDS.get(city_key)
    if bounds is not None:
        min_lat, max_lat, min_lon, max_lon = bounds
        if not (min_lat <= lat_f <= max_lat and min_lon <= lon_f <= max_lon):
            return False

    if city_key == "toronto" and lat_f < _toronto_shoreline_lat(lon_f):
        return False

    if city_key == "hamilton" and not _hamilton_harbour_guard(lat_f, lon_f):
        return False

    return True


def filter_land_safe_records(
    records: list[dict[str, Any]],
    city_key: str = "city",
    lat_key: str = "lat",
    lon_key: str = "lon",
) -> list[dict[str, Any]]:
    """
    Filter a list of housing dicts down to ones that look land-safe.

    Accepts either 'lat'/'lon' or 'latitude'/'longitude' keys per record.
    """
    kept = []
    for record in records:
        lat = record.get(lat_key, record.get("latitude"))
        lon = record.get(lon_key, record.get("longitude"))
        city = record.get(city_key, "")
        if is_land_safe(city, lat, lon):
            kept.append(record)
    return kept
