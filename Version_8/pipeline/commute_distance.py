"""
Commute distance providers.

Two providers are available:

    StraightLineDistanceProvider  offline haversine distance
    OSRMDistanceProvider          road distance and duration from OSRM

Both expose the same two methods, so they can be swapped freely:

    distances(origin, destinations)  -> list of commute dictionaries
    route_geometry(origin, target)   -> list of [lat, lon] points, or None

Every commute dictionary carries a "source" of either "road" or
"straight_line", so a caller can always tell how a number was produced.
"""

from __future__ import annotations

import json
import time
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path
from typing import Any, Sequence

import requests


STRAIGHT_LINE = "straight_line"
ROAD = "road"

DEFAULT_OSRM_URL = "https://router.project-osrm.org"

# The public OSRM demo server rejects very large coordinate lists.
MAX_TABLE_DESTINATIONS = 90


def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """
    Calculate straight-line distance between two coordinates in kilometres.
    """
    earth_radius_km = 6371.0

    lat1_rad = radians(float(lat1))
    lon1_rad = radians(float(lon1))
    lat2_rad = radians(float(lat2))
    lon2_rad = radians(float(lon2))

    delta_lat = lat2_rad - lat1_rad
    delta_lon = lon2_rad - lon1_rad

    a = (
        sin(delta_lat / 2) ** 2
        + cos(lat1_rad)
        * cos(lat2_rad)
        * sin(delta_lon / 2) ** 2
    )

    c = 2 * atan2(
        sqrt(a),
        sqrt(1 - a),
    )

    return earth_radius_km * c


def straight_line_commute(
    origin: tuple[float, float],
    destination: tuple[float, float],
) -> dict[str, Any]:
    """Build a straight-line commute result for one coordinate pair."""
    return {
        "distance_km": haversine_distance(
            origin[0],
            origin[1],
            destination[0],
            destination[1],
        ),
        "duration_minutes": None,
        "source": STRAIGHT_LINE,
    }


class StraightLineDistanceProvider:
    """Offline provider that measures straight-line distance only."""

    def distances(
        self,
        origin: tuple[float, float],
        destinations: Sequence[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """Return one straight-line commute result per destination."""
        return [
            straight_line_commute(origin, destination)
            for destination in destinations
        ]

    def route_geometry(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> list[list[float]] | None:
        """Straight-line distance has no road geometry to draw."""
        return None


class OSRMDistanceProvider:
    """
    Road distance and travel time from an OSRM routing server.

    The default server is the public OSRM demo, which needs no API key but is
    rate limited and unsuitable for heavy use. Point base_url at a self-hosted
    OSRM instance for production work.

    Results are cached in memory, and optionally on disk via cache_path, so
    repeated runs over the same coordinates do not re-query the server.

    Any network failure, timeout, or unroutable pair falls back to
    straight-line distance rather than raising, and the affected results are
    marked with a "straight_line" source.
    """

    def __init__(
        self,
        base_url: str = DEFAULT_OSRM_URL,
        profile: str = "driving",
        timeout: float = 10.0,
        max_retries: int = 2,
        min_request_interval: float = 1.0,
        cache_path: str | Path | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.profile = profile
        self.timeout = float(timeout)
        self.max_retries = int(max_retries)
        self.min_request_interval = float(min_request_interval)
        self.cache_path = (
            Path(cache_path)
            if cache_path is not None
            else None
        )

        self._cache: dict[str, Any] = {}
        self._cache_dirty = False
        self._last_request_time = 0.0

        self._load_cache()

    # ---------------------------------------------------------------- cache

    def _load_cache(self) -> None:
        """Load a previously saved cache file when one exists."""
        if self.cache_path is None or not self.cache_path.exists():
            return

        try:
            self._cache = json.loads(
                self.cache_path.read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            self._cache = {}

    def save_cache(self) -> None:
        """Write the cache to disk when a cache_path was supplied."""
        if self.cache_path is None or not self._cache_dirty:
            return

        try:
            self.cache_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )
            self.cache_path.write_text(
                json.dumps(self._cache),
                encoding="utf-8",
            )
            self._cache_dirty = False
        except OSError:
            pass

    def _cache_key(
        self,
        kind: str,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> str:
        """Build a cache key from rounded coordinates."""
        return (
            f"{kind}|{self.profile}"
            f"|{origin[0]:.5f},{origin[1]:.5f}"
            f"|{destination[0]:.5f},{destination[1]:.5f}"
        )

    # -------------------------------------------------------------- requests

    def _wait_for_rate_limit(self) -> None:
        """Pause so requests stay within the configured interval."""
        if self.min_request_interval <= 0:
            return

        elapsed = time.monotonic() - self._last_request_time
        remaining = self.min_request_interval - elapsed

        if remaining > 0:
            time.sleep(remaining)

    def _get(
        self,
        url: str,
        params: dict[str, str],
    ) -> dict[str, Any] | None:
        """Send a GET request, retrying briefly, and return parsed JSON."""
        for attempt in range(self.max_retries + 1):
            self._wait_for_rate_limit()

            try:
                response = requests.get(
                    url,
                    params=params,
                    timeout=self.timeout,
                )
                self._last_request_time = time.monotonic()

                response.raise_for_status()
                payload = response.json()
            except (requests.RequestException, ValueError):
                self._last_request_time = time.monotonic()

                if attempt == self.max_retries:
                    return None

                time.sleep(2**attempt)
                continue

            if payload.get("code") != "Ok":
                return None

            return payload

        return None

    @staticmethod
    def _coordinate_string(
        coordinates: Sequence[tuple[float, float]],
    ) -> str:
        """Format coordinates as the lon,lat pairs OSRM expects."""
        return ";".join(
            f"{lon:.6f},{lat:.6f}"
            for lat, lon in coordinates
        )

    # --------------------------------------------------------- public API

    def distances(
        self,
        origin: tuple[float, float],
        destinations: Sequence[tuple[float, float]],
    ) -> list[dict[str, Any]]:
        """
        Return one commute result per destination, in the order given.

        A single table request covers up to MAX_TABLE_DESTINATIONS
        destinations, so scoring many homes against one job stays cheap.
        """
        results: list[dict[str, Any] | None] = [None] * len(destinations)
        pending: list[int] = []

        for index, destination in enumerate(destinations):
            cached = self._cache.get(
                self._cache_key(
                    "table",
                    origin,
                    destination,
                )
            )

            if cached is None:
                pending.append(index)
            else:
                results[index] = dict(cached)

        for start in range(0, len(pending), MAX_TABLE_DESTINATIONS):
            chunk = pending[start:start + MAX_TABLE_DESTINATIONS]

            self._fill_chunk(
                origin,
                destinations,
                chunk,
                results,
            )

        self.save_cache()

        return [
            result
            if result is not None
            else straight_line_commute(origin, destinations[index])
            for index, result in enumerate(results)
        ]

    def _fill_chunk(
        self,
        origin: tuple[float, float],
        destinations: Sequence[tuple[float, float]],
        chunk: list[int],
        results: list[dict[str, Any] | None],
    ) -> None:
        """Resolve one batch of destinations into the results list."""
        chunk_destinations = [destinations[index] for index in chunk]

        payload = self._get(
            f"{self.base_url}/table/v1/{self.profile}/"
            + self._coordinate_string(
                [origin, *chunk_destinations]
            ),
            {
                "sources": "0",
                "destinations": ";".join(
                    str(position)
                    for position in range(
                        1,
                        len(chunk_destinations) + 1,
                    )
                ),
                "annotations": "distance,duration",
            },
        )

        if payload is None:
            for index in chunk:
                results[index] = straight_line_commute(
                    origin,
                    destinations[index],
                )
            return

        distance_row = (payload.get("distances") or [[]])[0]
        duration_row = (payload.get("durations") or [[]])[0]

        for position, index in enumerate(chunk):
            metres = (
                distance_row[position]
                if position < len(distance_row)
                else None
            )

            if metres is None:
                results[index] = straight_line_commute(
                    origin,
                    destinations[index],
                )
                continue

            seconds = (
                duration_row[position]
                if position < len(duration_row)
                else None
            )

            result = {
                "distance_km": float(metres) / 1000.0,
                "duration_minutes": (
                    float(seconds) / 60.0
                    if seconds is not None
                    else None
                ),
                "source": ROAD,
            }

            self._cache[
                self._cache_key(
                    "table",
                    origin,
                    destinations[index],
                )
            ] = result
            self._cache_dirty = True

            results[index] = dict(result)

    def route_geometry(
        self,
        origin: tuple[float, float],
        destination: tuple[float, float],
    ) -> list[list[float]] | None:
        """
        Return the driving route as [lat, lon] points for drawing on a map.

        Returns None when the route cannot be retrieved, so the caller can
        fall back to a straight line.
        """
        key = self._cache_key(
            "route",
            origin,
            destination,
        )

        if key in self._cache:
            return self._cache[key]

        payload = self._get(
            f"{self.base_url}/route/v1/{self.profile}/"
            + self._coordinate_string([origin, destination]),
            {
                "overview": "full",
                "geometries": "geojson",
            },
        )

        if payload is None:
            return None

        routes = payload.get("routes") or []

        if not routes:
            return None

        coordinates = (
            routes[0]
            .get("geometry", {})
            .get("coordinates", [])
        )

        if not coordinates:
            return None

        geometry = [
            [float(lat), float(lon)]
            for lon, lat in coordinates
        ]

        self._cache[key] = geometry
        self._cache_dirty = True
        self.save_cache()

        return geometry
