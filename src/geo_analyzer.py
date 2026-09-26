"""
Module: geo_analyzer.py
Spatial GIS mapping engine using Folium and MarkerClusters.
Renders interactive disaster heatmaps, marker clusters, and shelter locations.
"""

import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple


def create_gis_heatmap(
    df: pd.DataFrame,
    center_lat: float = 20.5937,
    center_lon: float = 78.9629,
    zoom_start: int = 5,
    map_tile: str = "CartoDB positron"
) -> folium.Map:
    """Renders a spatial Folium HeatMap based on disaster location coordinates and risk scores."""
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron" if map_tile == "CartoDB positron" else "OpenStreetMap"
    )
    MiniMap(toggle_display=True, position="bottomright").add_to(m)

    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        heat_data = []
        for _, r in df.iterrows():
            lat, lon = r["latitude"], r["longitude"]
            risk = r.get("risk_score", 50.0)
            heat_data.append([lat, lon, risk / 100.0])

        if heat_data:
            HeatMap(heat_data, name="🔥 Hazard Density Heatmap", min_opacity=0.35, radius=22, blur=18).add_to(m)

    folium.LayerControl().add_to(m)
    return m


def create_marker_cluster_map(
    df: pd.DataFrame,
    shelter_df: Optional[pd.DataFrame] = None,
    center_lat: float = 20.5937,
    center_lon: float = 78.9629,
    zoom_start: int = 5
) -> folium.Map:
    """Renders a Folium MarkerCluster map with dynamic HTML popups, risk colors, and shelter pins."""
    m = folium.Map(location=[center_lat, center_lon], zoom_start=zoom_start, tiles="CartoDB positron")

    if not df.empty and "latitude" in df.columns and "longitude" in df.columns:
        disaster_group = folium.FeatureGroup(name="🚨 Disaster Incidents")
        cluster = MarkerCluster().add_to(disaster_group)

        for _, r in df.iterrows():
            lat, lon = r["latitude"], r["longitude"]
            loc_name = r.get("location_name", "Incident")
            risk = r.get("risk_score", 50.0)
            cas = r.get("casualties", 0)

            color = "#EF4444" if risk >= 75 else ("#F59E0B" if risk >= 50 else "#10B981")

            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 220px; padding: 6px;'>
                <div style='background: {color}; color: white; padding: 6px; border-radius: 6px; font-weight: bold;'>
                    🚨 {loc_name}
                </div>
                <div style='margin-top: 8px; font-size: 12px; color: #1E293B;'>
                    <b>Risk Severity Score:</b> <b style='color: {color};'>{risk} / 100</b><br/>
                    <b>Casualties Logged:</b> {cas:,}<br/>
                    <b>Timestamp:</b> {r.get('timestamp', 'N/A')}
                </div>
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(22, risk / 4.5)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"{loc_name} - Risk Score: {risk}/100"
            ).add_to(cluster)

        disaster_group.add_to(m)

    # Add Emergency Shelters Layer if provided
    if shelter_df is not None and not shelter_df.empty:
        shelter_group = folium.FeatureGroup(name="🏥 Emergency Shelters")
        for _, s in shelter_df.iterrows():
            lat, lon, name = s["latitude"], s["longitude"], s["name"]
            cap, occ = s["capacity"], s["current_occupancy"]
            avail = cap - occ
            status_color = "green" if avail > 200 else ("orange" if avail > 0 else "red")

            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 230px;'>
                <b style='color: #10B981; font-size: 14px;'>🏥 {name}</b><br/>
                <b>Capacity:</b> {cap:,} | <b>Available:</b> <b style='color: {status_color};'>{avail:,} seats</b><br/>
                <b>Phone:</b> {s.get('contact_number', 'N/A')}
            </div>
            """
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="green", icon="home", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"Shelter: {name} (Avail: {avail})"
            ).add_to(shelter_group)
        shelter_group.add_to(m)

    folium.LayerControl().add_to(m)
    return m
