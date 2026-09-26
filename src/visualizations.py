"""
Module 4: Visualization Components (Vibrant & Interactive Design Engine)
Provides functions for multi-layer Folium GIS risk maps, Plotly animated trend charts,
severity histograms, risk gauge charts, comparative bar charts, and impact breakdown cards.
Uses rich color palettes (Neon Red, Ocean Blue, Electric Purple, Emerald Green, Amber Gold).
"""

import folium
from folium.plugins import HeatMap, MarkerCluster, MiniMap
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

from config import DEFAULT_LAT, DEFAULT_LON, DEFAULT_ZOOM, RISK_LEVELS

# Color Palette Constants
COLOR_EARTHQUAKE = "#FF3366"  # Electric Crimson
COLOR_FLOOD = "#00D2FF"       # Ocean Blue
COLOR_CYCLONE = "#9D50BB"     # Radiant Purple
COLOR_SHELTER = "#10B981"     # Emerald Green
COLOR_WARNING = "#F59E0B"     # Amber Gold
COLOR_CRITICAL = "#EF4444"    # Bright Red


def create_interactive_risk_map(
    eq_df: pd.DataFrame,
    flood_df: pd.DataFrame,
    cyc_df: pd.DataFrame,
    shelter_df: pd.DataFrame,
    selected_disaster: str = "All",
    center_lat: float = DEFAULT_LAT,
    center_lon: float = DEFAULT_LON,
    zoom_start: int = DEFAULT_ZOOM,
    map_tile: str = "CartoDB positron"
) -> folium.Map:
    """
    Create a highly styled, multi-layer Folium GIS map with HeatMap, pulse markers,
    and shelter occupancy progress meters inside popups.
    """
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles="CartoDB positron" if map_tile == "CartoDB positron" else "OpenStreetMap",
        control_scale=True
    )

    # Add MiniMap inset for spatial navigation
    MiniMap(toggle_display=True, tile_layer="CartoDB positron", position="bottomright").add_to(m)

    # 1. Earthquakes Layer
    if selected_disaster in ["All", "Earthquake"] and not eq_df.empty:
        eq_group = folium.FeatureGroup(name="🌋 Earthquakes (USGS Live Feed)")
        eq_heat_data = []

        for _, row in eq_df.iterrows():
            lat, lon, mag, depth, risk = row["latitude"], row["longitude"], row["magnitude"], row["depth"], row["risk_score"]
            eq_heat_data.append([lat, lon, risk / 100.0])

            # Vibrant color gradient based on magnitude/risk
            if risk >= 75:
                color = "#DC2626"
                badge_bg = "#FEE2E2"
            elif risk >= 50:
                color = "#F59E0B"
                badge_bg = "#FEF3C7"
            else:
                color = "#10B981"
                badge_bg = "#D1FAE5"

            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 230px; padding: 6px; border-radius: 8px;'>
                <div style='background: {color}; color: white; padding: 6px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; display: flex; justify-space: space-between;'>
                    <span>🌋 Earthquake</span>
                    <span>M{mag}</span>
                </div>
                <div style='margin-top: 8px; font-size: 12px; color: #1E293B; line-height: 1.6;'>
                    <b>Location:</b> {row.get('location', 'N/A')}<br/>
                    <b>Focal Depth:</b> <span style='color: #475569; font-weight: 600;'>{depth} km</span><br/>
                    <b>Hazard Score:</b> <span style='background: {badge_bg}; color: {color}; padding: 2px 6px; border-radius: 4px; font-weight: bold;'>{risk} / 100</span><br/>
                    <b>Tsunami Status:</b> {'⚠️ YES - ALERT' if row.get('tsunami_warning', 0) == 1 else 'None'}<br/>
                    <b>Timestamp:</b> {row.get('timestamp', 'N/A')}
                </div>
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(5, min(22, mag * 2.5)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🌋 M{mag} Earthquake - Risk: {risk}/100"
            ).add_to(eq_group)

        if eq_heat_data:
            HeatMap(eq_heat_data, name="🔥 Seismic Heatmap Density", min_opacity=0.35, radius=22, blur=18).add_to(eq_group)
        eq_group.add_to(m)

    # 2. Floods Layer
    if selected_disaster in ["All", "Flood"] and not flood_df.empty:
        flood_group = folium.FeatureGroup(name="🌊 Floods & Inundations")
        for _, row in flood_df.iterrows():
            lat, lon, rain, river, risk = row["latitude"], row["longitude"], row["rainfall_mm"], row["river_level_m"], row["risk_score"]
            color = "#0284C7" if risk >= 60 else "#06B6D4"

            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 230px; padding: 6px;'>
                <div style='background: linear-gradient(135deg, #0284C7, #06B6D4); color: white; padding: 6px 10px; border-radius: 6px; font-weight: bold; font-size: 13px;'>
                    🌊 Flood Inundation Hazard
                </div>
                <div style='margin-top: 8px; font-size: 12px; color: #1E293B; line-height: 1.6;'>
                    <b>Region:</b> {row.get('region', 'N/A')}<br/>
                    <b>24h Rainfall:</b> <b style='color: #0284C7;'>{rain} mm</b><br/>
                    <b>River Gauge:</b> <b>{river} m</b> (Thresh: {row.get('threshold_m', 5.0)}m)<br/>
                    <b>Risk Score:</b> <b style='color: #0284C7;'>{risk} / 100</b>
                </div>
            </div>
            """
            folium.CircleMarker(
                location=[lat, lon],
                radius=max(6, min(20, river * 2.0)),
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"🌊 Flood Hazard - Risk: {risk}/100"
            ).add_to(flood_group)
        flood_group.add_to(m)

    # 3. Cyclones Layer
    if selected_disaster in ["All", "Cyclone"] and not cyc_df.empty:
        cyc_group = folium.FeatureGroup(name="🌀 Cyclones & Storm Systems")
        for _, row in cyc_df.iterrows():
            lat, lon, wind, cat, risk = row["latitude"], row["longitude"], row["wind_speed_kmh"], row["category"], row["risk_score"]
            
            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 240px; padding: 6px;'>
                <div style='background: linear-gradient(135deg, #7C3AED, #C084FC); color: white; padding: 6px 10px; border-radius: 6px; font-weight: bold; font-size: 13px;'>
                    🌀 Cyclone {row.get('cyclone_name', '')}
                </div>
                <div style='margin-top: 8px; font-size: 12px; color: #1E293B; line-height: 1.6;'>
                    <b>Category:</b> {cat}<br/>
                    <b>Wind Speed:</b> <b style='color: #7C3AED;'>{wind} km/h</b><br/>
                    <b>Pressure:</b> {row.get('pressure_hpa', 1000)} hPa<br/>
                    <b>Risk Score:</b> <b style='color: #7C3AED;'>{risk} / 100</b>
                </div>
            </div>
            """
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="purple", icon="cloud", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=260),
                tooltip=f"🌀 {row.get('cyclone_name', 'Cyclone')} ({wind} km/h)"
            ).add_to(cyc_group)
        cyc_group.add_to(m)

    # 4. Emergency Shelters Layer with Occupancy Progress Meter inside Popup
    if not shelter_df.empty:
        shelter_group = folium.FeatureGroup(name="🏥 Emergency Shelters & Relief Hubs")
        cluster = MarkerCluster().add_to(shelter_group)
        for _, row in shelter_df.iterrows():
            lat, lon, name, cap, occ = row["latitude"], row["longitude"], row["name"], row["capacity"], row["current_occupancy"]
            avail = cap - occ
            pct_occupied = int((occ / max(1, cap)) * 100)
            
            status_color = "#10B981" if pct_occupied < 70 else ("#F59E0B" if pct_occupied < 90 else "#EF4444")
            status_label = "Available" if pct_occupied < 70 else ("Near Capacity" if pct_occupied < 90 else "FULL")

            popup_html = f"""
            <div style='font-family: system-ui, sans-serif; width: 250px; padding: 6px;'>
                <div style='background: #0F172A; color: white; padding: 8px 10px; border-radius: 6px; font-weight: bold; font-size: 13px; display: flex; justify-content: space-between; align-items: center;'>
                    <span>🏥 {name}</span>
                    <span style='background: {status_color}; padding: 2px 6px; border-radius: 4px; font-size: 10px;'>{status_label}</span>
                </div>
                <div style='margin-top: 10px; font-size: 12px; color: #1E293B;'>
                    <b>Region:</b> {row.get('region', 'N/A')}<br/>
                    <b>Capacity:</b> {cap:,} seats | <b>Occupied:</b> {occ:,}<br/>
                    <b>Available Seats:</b> <b style='color: {status_color}; font-size: 14px;'>{avail:,}</b><br/>
                    
                    <!-- Occupancy Progress Bar -->
                    <div style='margin-top: 6px; background: #E2E8F0; border-radius: 6px; height: 10px; overflow: hidden;'>
                        <div style='background: {status_color}; width: {pct_occupied}%; height: 100%; border-radius: 6px;'></div>
                    </div>
                    <div style='text-align: right; font-size: 10px; color: #64748B; margin-top: 2px;'>{pct_occupied}% Occupied</div>
                    
                    <div style='margin-top: 8px; font-size: 11px; background: #F8FAFC; padding: 6px; border-radius: 4px;'>
                        🍚 <b>Food Ration:</b> {row.get('supplies_food_days', 7)} days | 🩺 <b>Kits:</b> {row.get('medical_kits', 50)}<br/>
                        📞 <b>Phone:</b> {row.get('contact_number', 'N/A')}
                    </div>
                </div>
            </div>
            """
            folium.Marker(
                location=[lat, lon],
                icon=folium.Icon(color="green" if pct_occupied < 80 else "red", icon="home", prefix="fa"),
                popup=folium.Popup(popup_html, max_width=270),
                tooltip=f"🏥 Shelter: {name} ({avail} seats available)"
            ).add_to(cluster)
        shelter_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def create_multiline_trend_chart(
    eq_df: pd.DataFrame,
    flood_df: pd.DataFrame,
    cyc_df: pd.DataFrame
) -> go.Figure:
    """
    Vibrant Plotly multi-line chart with glowing curves, filled gradients, and unified hover template.
    """
    fig = go.Figure()

    disasters = [
        (eq_df, "Earthquakes 🌋", COLOR_EARTHQUAKE, "rgba(255, 51, 102, 0.1)"),
        (flood_df, "Floods 🌊", COLOR_FLOOD, "rgba(0, 210, 255, 0.1)"),
        (cyc_df, "Cyclones 🌀", COLOR_CYCLONE, "rgba(157, 80, 187, 0.1)")
    ]

    for df, name, color, fill_color in disasters:
        if not df.empty and "timestamp" in df.columns:
            df_ts = df.copy()
            df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
            monthly = df_ts.set_index("timestamp").resample("ME").size().reset_index(name="count")

            fig.add_trace(go.Scatter(
                x=monthly["timestamp"],
                y=monthly["count"],
                mode="lines+markers",
                name=name,
                line=dict(color=color, width=3, shape="spline"),
                marker=dict(size=7, color=color, symbol="circle", line=dict(width=1, color="white")),
                fill="tozeroy",
                fillcolor=fill_color
            ))

    fig.update_layout(
        title="<b>📈 Monthly Disaster Incident Frequency Trends</b>",
        xaxis_title="Timeline",
        yaxis_title="Incidents Logged",
        template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=12, color="#1E293B")),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    return fig


def create_risk_gauge_chart(score: float, title: str = "Composite Hazard Index") -> go.Figure:
    """
    Creates an interactive gauge chart with vibrant color bands and dynamic indicator needle.
    """
    score = float(max(0.0, min(100.0, score)))

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=score,
        title={'text': f"<b>{title}</b>", 'font': {'size': 18, 'color': '#0F172A'}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 2, 'tickcolor': "#334155"},
            'bar': {'color': "#0F172A", 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0, 30], 'color': '#10B981'},    # Emerald Green
                {'range': [30, 60], 'color': '#F59E0B'},   # Amber Gold
                {'range': [60, 80], 'color': '#F97316'},   # Vibrant Orange
                {'range': [80, 100], 'color': '#EF4444'}   # Crimson Red
            ],
            'threshold': {
                'line': {'color': "#DC2626", 'width': 4},
                'thickness': 0.8,
                'value': score
            }
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=50, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


def create_severity_histogram(
    eq_df: pd.DataFrame,
    flood_df: pd.DataFrame,
    cyc_df: pd.DataFrame
) -> go.Figure:
    """
    Histogram distribution comparing risk score distribution across hazard types with distinct vibrant colors.
    """
    combined = []
    if not eq_df.empty:
        df1 = eq_df[["risk_score"]].copy()
        df1["Disaster Type"] = "Earthquake 🌋"
        combined.append(df1)
    if not flood_df.empty:
        df2 = flood_df[["risk_score"]].copy()
        df2["Disaster Type"] = "Flood 🌊"
        combined.append(df2)
    if not cyc_df.empty:
        df3 = cyc_df[["risk_score"]].copy()
        df3["Disaster Type"] = "Cyclone 🌀"
        combined.append(df3)

    if not combined:
        return go.Figure()

    df_all = pd.concat(combined, ignore_index=True)

    fig = px.histogram(
        df_all,
        x="risk_score",
        color="Disaster Type",
        barmode="overlay",
        nbins=22,
        color_discrete_map={
            "Earthquake 🌋": COLOR_EARTHQUAKE,
            "Flood 🌊": COLOR_FLOOD,
            "Cyclone 🌀": COLOR_CYCLONE
        },
        title="<b>📊 Risk Severity Score Distribution across Hazards</b>",
        labels={"risk_score": "Risk Score (0 - 100)"}
    )
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    fig.update_traces(opacity=0.8)
    return fig


def create_impact_analysis_chart(
    eq_df: pd.DataFrame,
    flood_df: pd.DataFrame,
    cyc_df: pd.DataFrame
) -> go.Figure:
    """
    Grouped bar chart for estimated economic loss ($M) and direct casualties.
    """
    eq_loss = float(eq_df["economic_loss_millions"].sum()) if not eq_df.empty and "economic_loss_millions" in eq_df.columns else 0.0
    fl_loss = float((flood_df["affected_population"].sum() * 0.002)) if not flood_df.empty and "affected_population" in flood_df.columns else 0.0
    cy_loss = float(len(cyc_df) * 12.5) if not cyc_df.empty else 0.0

    eq_cas = int(eq_df["casualties"].sum()) if not eq_df.empty and "casualties" in eq_df.columns else 0
    fl_cas = int(flood_df["affected_population"].sum() * 0.0005) if not flood_df.empty and "affected_population" in flood_df.columns else 0
    cy_cas = int(len(cyc_df) * 15) if not cyc_df.empty else 0

    categories = ["Earthquake 🌋", "Flood 🌊", "Cyclone 🌀"]
    losses = [eq_loss, fl_loss, cy_loss]
    casualties = [eq_cas, fl_cas, cy_cas]

    fig = go.Figure(data=[
        go.Bar(
            name="Economic Loss ($ Million)",
            x=categories,
            y=losses,
            marker=dict(color=["#EF4444", "#3B82F6", "#8B5CF6"], line=dict(color="#0F172A", width=1.5)),
            text=[f"${v:,.1f}M" for v in losses],
            textposition="auto"
        ),
        go.Bar(
            name="Casualties / Direct Impact",
            x=categories,
            y=casualties,
            marker=dict(color=["#F59E0B", "#06B6D4", "#C084FC"], line=dict(color="#0F172A", width=1.5)),
            text=[f"{v:,}" for v in casualties],
            textposition="auto"
        )
    ])

    fig.update_layout(
        barmode="group",
        title="<b>💰 Socio-Economic Impact Assessment by Hazard Category</b>",
        xaxis_title="Disaster Category",
        yaxis_title="Magnitude",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.8)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    return fig


def create_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    """
    Time-Series forecasting chart with a shaded 95% Confidence Band (`fill='toself'`) and projection curve.
    """
    fig = go.Figure()

    # 95% Confidence Interval Band
    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["upper_ci_95"], forecast_df["lower_ci_95"][::-1]]),
        fill="toself",
        fillcolor="rgba(59, 130, 246, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        showlegend=True,
        name="95% Confidence Interval Band"
    ))

    # Forecast Mean Curve
    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["forecast_mean"],
        mode="lines+markers",
        name="Predicted Disaster Event Volume",
        line=dict(color="#2563EB", width=3.5, shape="spline"),
        marker=dict(size=8, color="#1D4ED8", symbol="diamond", line=dict(width=1.5, color="white"))
    ))

    fig.update_layout(
        title="<b>🔮 12-Month Predictive Disaster Event Volume Projection (Holt-Winters)</b>",
        xaxis_title="Forecast Month",
        yaxis_title="Predicted Event Count",
        template="plotly_white",
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248, 250, 252, 0.8)"
    )
    return fig
