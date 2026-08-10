"""Interactive settlement map for housing and commute recommendations."""

from __future__ import annotations

from typing import Any

import folium
import pandas as pd

from .commute_distance import ROAD, OSRMDistanceProvider


def build_settlement_map(
    recommendations_df: pd.DataFrame,
    max_items: int = 15,
    zoom: int = 9,
    distance_provider: Any | None = None,
) -> folium.Map:
    """
    Build a Folium map from recommend_housing() output.

    Commute lines follow the driving route when the recommendations carry road distances and the provider can return route geometry. Otherwise they are
    drawn as straight lines.

    distance_provider defaults to OSRMDistanceProvider, and is only consulted when at least one row was scored with a road distance.
    """
    if recommendations_df.empty:
        raise ValueError(
            "The recommendations DataFrame is empty."
        )

    required_columns = {
        "job_lat",
        "job_lon",
        "lat",
        "lon",
        "job_title",
        "job_company",
        "job_location",
        "job_score",
        "city",
        "neighbourhood",
        "monthly_rent",
        "max_affordable_rent",
        "commute_km",
        "affordable",
        "housing_score",
        "combined_score",
    }

    missing = required_columns.difference(
        recommendations_df.columns
    )

    if missing:
        raise ValueError(
            "Recommendations are missing required columns: "
            + ", ".join(sorted(missing))
        )

    top_items = recommendations_df.head(
        max_items
    )

    has_road_distances = (
        "distance_source" in top_items.columns
        and (top_items["distance_source"] == ROAD).any()
    )

    if has_road_distances and distance_provider is None:
        distance_provider = OSRMDistanceProvider()

    if not has_road_distances:
        distance_provider = None

    all_latitudes = pd.concat(
        [
            top_items["job_lat"],
            top_items["lat"],
        ]
    )
    all_longitudes = pd.concat(
        [
            top_items["job_lon"],
            top_items["lon"],
        ]
    )

    settlement_map = folium.Map(
        location=[
            all_latitudes.mean(),
            all_longitudes.mean(),
        ],
        zoom_start=zoom,
    )

    jobs_layer = folium.FeatureGroup(
        name="Recommended Jobs",
        show=True,
    )
    homes_layer = folium.FeatureGroup(
        name="Recommended Housing",
        show=True,
    )
    commute_layer = folium.FeatureGroup(
        name="Driving Commute" if has_road_distances else "Straight-line Commute",
        show=True,
    )

    plotted_jobs = set()
    plotted_homes = set()
    all_coordinates = []

    for _, row in top_items.iterrows():
        job_coordinate = [
            row["job_lat"],
            row["job_lon"],
        ]
        home_coordinate = [
            row["lat"],
            row["lon"],
        ]

        all_coordinates.extend(
            [
                job_coordinate,
                home_coordinate,
            ]
        )

        job_key = (
            str(row["job_title"]),
            str(row["job_company"]),
            float(row["job_lat"]),
            float(row["job_lon"]),
        )

        if job_key not in plotted_jobs:
            job_popup = (
                f"<b>{row['job_title']}</b><br>"
                f"Company: {row['job_company']}<br>"
                f"Location: {row['job_location']}<br>"
                f"Job score: {row['job_score']:.1f}"
            )

            folium.Marker(
                job_coordinate,
                popup=job_popup,
                tooltip=folium.Tooltip(
                    job_popup,
                    sticky=True,
                ),
                icon=folium.Icon(
                    color="blue",
                    icon="briefcase",
                    prefix="fa",
                ),
            ).add_to(jobs_layer)

            plotted_jobs.add(job_key)

        home_key = (
            str(row["city"]),
            str(row["neighbourhood"]),
            float(row["lat"]),
            float(row["lon"]),
        )

        commute_label = (
            "Driving commute"
            if row.get("distance_source") == ROAD
            else "Straight-line commute"
        )

        commute_minutes = row.get("commute_minutes")

        commute_text = f"{row['commute_km']:.1f} km"

        if commute_minutes is not None and pd.notna(commute_minutes):
            commute_text += f" / {float(commute_minutes):.0f} min"

        if home_key not in plotted_homes:
            home_popup = (
                f"<b>{row['neighbourhood']}, "
                f"{row['city']}</b><br>"
                f"Rent: ${row['monthly_rent']:,.0f}<br>"
                f"Affordable limit: "
                f"${row['max_affordable_rent']:,.0f}<br>"
                f"{commute_label}: "
                f"{commute_text}<br>"
                f"Affordable: "
                f"{'Yes' if row['affordable'] else 'No'}<br>"
                f"Housing score: "
                f"{row['housing_score']:.1f}<br>"
                f"Combined score: "
                f"{row['combined_score']:.1f}"
            )

            marker_colour = (
                "green"
                if row["affordable"]
                else "orange"
            )

            folium.Marker(
                home_coordinate,
                popup=home_popup,
                tooltip=folium.Tooltip(
                    home_popup,
                    sticky=True,
                ),
                icon=folium.Icon(
                    color=marker_colour,
                    icon="home",
                    prefix="fa",
                ),
            ).add_to(homes_layer)

            plotted_homes.add(home_key)

        route_points = None

        if distance_provider is not None:
            route_points = distance_provider.route_geometry(
                (
                    float(row["job_lat"]),
                    float(row["job_lon"]),
                ),
                (
                    float(row["lat"]),
                    float(row["lon"]),
                ),
            )

        folium.PolyLine(
            locations=route_points
            or [
                job_coordinate,
                home_coordinate,
            ],
            weight=(
                4
                if route_points
                else 2
            ),
            opacity=0.7,
            tooltip=folium.Tooltip(
                f"{commute_label}: {commute_text}",
                sticky=True,
            ),
        ).add_to(commute_layer)

    jobs_layer.add_to(settlement_map)
    homes_layer.add_to(settlement_map)
    commute_layer.add_to(settlement_map)

    folium.LayerControl(
        collapsed=False
    ).add_to(settlement_map)

    if all_coordinates:
        settlement_map.fit_bounds(
            all_coordinates
        )

    return settlement_map