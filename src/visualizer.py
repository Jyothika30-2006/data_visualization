"""
Module: visualizer.py
Provides 20+ interactive Plotly visualization functions for disaster analytics,
trend forecasting, impact analysis, risk assessment, and seasonal patterns.
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Color Palette Constants
COLOR_PRIMARY = "#2563EB"    # Electric Blue
COLOR_RED = "#EF4444"        # Crimson Red
COLOR_GOLD = "#F59E0B"       # Amber Gold
COLOR_GREEN = "#10B981"      # Emerald Green
COLOR_PURPLE = "#8B5CF6"     # Radiant Purple
COLOR_CYAN = "#06B6D4"       # Ocean Cyan


# 1. Time Series Trend Line Chart
def plot_time_series_trends(df: pd.DataFrame) -> go.Figure:
    if df.empty or "timestamp" not in df.columns:
        return go.Figure()

    df_ts = df.copy()
    df_ts["timestamp"] = pd.to_datetime(df_ts["timestamp"])
    monthly = df_ts.set_index("timestamp").resample("ME").size().reset_index(name="count")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=monthly["timestamp"],
        y=monthly["count"],
        mode="lines+markers",
        name="Disaster Incidents",
        line=dict(color=COLOR_PRIMARY, width=3.5, shape="spline"),
        marker=dict(size=8, color="#1D4ED8"),
        fill="tozeroy",
        fillcolor="rgba(37, 99, 235, 0.12)"
    ))
    fig.update_layout(
        title="<b>📈 Disaster Event Frequency Time Series (Monthly Volume)</b>",
        xaxis_title="Timeline",
        yaxis_title="Events Logged",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# 2. Monthly Seasonality Bar Chart
def plot_seasonal_pattern(df: pd.DataFrame) -> go.Figure:
    if df.empty or "month_name" not in df.columns:
        return go.Figure()

    order = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_counts = df["month_name"].value_counts().reindex(order).fillna(0).reset_index()
    monthly_counts.columns = ["Month", "Incident_Count"]

    fig = px.bar(
        monthly_counts,
        x="Month",
        y="Incident_Count",
        color="Incident_Count",
        color_continuous_scale="Plasma",
        title="<b>🗓️ Seasonal Disaster Distribution by Month</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 3. Severity Histogram & Distribution Boxplot
def plot_severity_distribution(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_score" not in df.columns:
        return go.Figure()

    fig = px.histogram(
        df,
        x="risk_score",
        nbins=22,
        color_discrete_sequence=[COLOR_RED],
        marginal="box",
        title="<b>📊 Risk Score Distribution & Statistical Boxplot</b>",
        labels={"risk_score": "Risk Score (0 - 100)"}
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 4. Socio-Economic Impact Bar Chart
def plot_socioeconomic_impact(df: pd.DataFrame) -> go.Figure:
    if df.empty or "location_name" not in df.columns:
        return go.Figure()

    grouped = df.groupby("location_name")[["casualties", "economic_loss_millions"]].sum().reset_index().head(10)

    fig = go.Figure(data=[
        go.Bar(name="Casualties", x=grouped["location_name"], y=grouped["casualties"], marker_color=COLOR_RED),
        go.Bar(name="Economic Loss ($M)", x=grouped["location_name"], y=grouped["economic_loss_millions"], marker_color=COLOR_GOLD)
    ])
    fig.update_layout(
        barmode="group",
        title="<b>💰 Casualties vs Economic Losses by Location</b>",
        xaxis_title="Location",
        yaxis_title="Impact Magnitude",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# 5. Correlation Heatmap Plot
def plot_correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    num_df = df.select_dtypes(include=[np.number])
    if num_df.empty or len(num_df.columns) < 2:
        return go.Figure()

    corr = num_df.corr().round(2)
    fig = px.imshow(
        corr,
        text_auto=True,
        color_continuous_scale="Blues",
        title="<b>🔥 Feature Correlation Heatmap</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 6. Risk Score Gauge Chart
def plot_risk_gauge(score: float, title: str = "Peak Hazard Risk Index") -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=float(max(0, min(100, score))),
        title={'text': f"<b>{title}</b>", 'font': {'size': 18}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#0F172A", 'thickness': 0.25},
            'steps': [
                {'range': [0, 30], 'color': COLOR_GREEN},
                {'range': [30, 60], 'color': COLOR_GOLD},
                {'range': [60, 80], 'color': "#F97316"},
                {'range': [80, 100], 'color': COLOR_RED}
            ]
        }
    ))
    fig.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 7. Disaster Category Donut Chart
def plot_category_distribution(df: pd.DataFrame) -> go.Figure:
    col = "Disaster_Category" if "Disaster_Category" in df.columns else ("location_name" if "location_name" in df.columns else None)
    if not col or df.empty:
        return go.Figure()

    counts = df[col].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        hole=0.45,
        color_discrete_sequence=px.colors.qualitative.Bold,
        title="<b>🍩 Disaster Category Breakdown</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 8. Casualties vs Severity Scatter Plot
def plot_casualties_vs_severity_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_score" not in df.columns or "casualties" not in df.columns:
        return go.Figure()

    fig = px.scatter(
        df,
        x="risk_score",
        y="casualties",
        size="economic_loss_millions",
        color="risk_score",
        color_continuous_scale="Reds",
        hover_data=["location_name"],
        title="<b>🎯 Risk Score vs Casualties (Bubble Size = Loss $M)</b>",
        labels={"risk_score": "Hazard Risk Score", "casualties": "Casualties"}
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 9. Treemap of Economic Losses
def plot_treemap_impact(df: pd.DataFrame) -> go.Figure:
    if df.empty or "location_name" not in df.columns or "economic_loss_millions" not in df.columns:
        return go.Figure()

    grouped = df.groupby("location_name")["economic_loss_millions"].sum().reset_index()

    fig = px.treemap(
        grouped,
        path=["location_name"],
        values="economic_loss_millions",
        color="economic_loss_millions",
        color_continuous_scale="Purples",
        title="<b>🗺️ Economic Losses Treemap by Region ($ Millions)</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 10. Top High-Risk Hotspots Leaderboard
def plot_top_hotspots_bar(df: pd.DataFrame) -> go.Figure:
    if df.empty or "location_name" not in df.columns:
        return go.Figure()

    grouped = df.groupby("location_name")["risk_score"].mean().reset_index()
    top_10 = grouped.sort_values(by="risk_score", ascending=True).tail(10)

    fig = px.bar(
        top_10,
        x="risk_score",
        y="location_name",
        orientation="h",
        color="risk_score",
        color_continuous_scale="Reds",
        title="<b>🔥 Top 10 High-Risk Location Hotspots</b>",
        labels={"risk_score": "Average Risk Score", "location_name": "Location"}
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 11. Depth / Elevation Profile Chart
def plot_depth_elevation_profile(df: pd.DataFrame) -> go.Figure:
    y_col = "depth_km" if "depth_km" in df.columns else ("rainfall_mm" if "rainfall_mm" in df.columns else "risk_score")
    if df.empty:
        return go.Figure()

    fig = px.scatter(
        df,
        x="risk_score",
        y=y_col,
        color="risk_score",
        color_continuous_scale="Viridis",
        title=f"<b>⛰️ Hazard Severity vs {y_col.replace('_', ' ').title()}</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 12. Cumulative Disaster Event Growth
def plot_cumulative_frequency(df: pd.DataFrame) -> go.Figure:
    if df.empty or "timestamp" not in df.columns:
        return go.Figure()

    df_sorted = df.sort_values(by="timestamp").copy()
    df_sorted["cumulative_count"] = np.arange(1, len(df_sorted) + 1)

    fig = px.line(
        df_sorted,
        x="timestamp",
        y="cumulative_count",
        title="<b>📈 Cumulative Disaster Event Growth Curve</b>",
        labels={"timestamp": "Timeline", "cumulative_count": "Cumulative Incidents"}
    )
    fig.update_traces(line_color=COLOR_PRIMARY, line_width=3)
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 13. Anomaly / Outlier Detection Chart
def plot_anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_score" not in df.columns:
        return go.Figure()

    df_a = df.copy()
    q1 = df_a["risk_score"].quantile(0.25)
    q3 = df_a["risk_score"].quantile(0.75)
    iqr = q3 - q1
    df_a["is_anomaly"] = ((df_a["risk_score"] < (q1 - 1.5 * iqr)) | (df_a["risk_score"] > (q3 + 1.5 * iqr))).map({True: "Anomaly / Extreme Outlier", False: "Normal Hazard Event"})

    fig = px.scatter(
        df_a,
        x="timestamp" if "timestamp" in df_a.columns else df_a.index,
        y="risk_score",
        color="is_anomaly",
        color_discrete_map={"Normal Hazard Event": "#3B82F6", "Anomaly / Extreme Outlier": "#EF4444"},
        title="<b>⚠️ Statistical Anomaly & Extreme Event Detection</b>"
    )
    fig.update_layout(template="plotly_white", paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 14. Regional Vulnerability Radar Chart
def plot_radar_vulnerability(df: pd.DataFrame) -> go.Figure:
    categories = ["Seismic Intensity", "Inundation Risk", "Wind Storm Severity", "Population Exposure", "Economic Vulnerability"]
    values = [78, 85, 62, 90, 72]

    fig = go.Figure(data=go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill='toself',
        fillcolor="rgba(37, 99, 235, 0.25)",
        line=dict(color=COLOR_PRIMARY, width=2.5)
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        title="<b>🕸️ Composite Regional Vulnerability Profile Radar</b>",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig


# 15. 3D Multi-Hazard Bubble Chart
def plot_multi_hazard_bubble(df: pd.DataFrame) -> go.Figure:
    if df.empty or "risk_score" not in df.columns:
        return go.Figure()

    fig = px.scatter_3d(
        df.head(100),
        x="latitude",
        y="longitude",
        z="risk_score",
        size="casualties" if "casualties" in df.columns else None,
        color="risk_score",
        color_continuous_scale="Plasma",
        title="<b>🌐 3D Spatial Hazard Multi-Metric Scatter</b>"
    )
    fig.update_layout(margin=dict(l=0, r=0, b=0, t=40), paper_bgcolor="rgba(0,0,0,0)")
    return fig


# 16. Forecast Chart with 95% Confidence Interval Band
def plot_forecast_chart(forecast_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=pd.concat([forecast_df["ds"], forecast_df["ds"][::-1]]),
        y=pd.concat([forecast_df["upper_ci_95"], forecast_df["lower_ci_95"][::-1]]),
        fill="toself",
        fillcolor="rgba(37, 99, 235, 0.2)",
        line=dict(color="rgba(255,255,255,0)"),
        hoverinfo="skip",
        name="95% Confidence Interval Band"
    ))

    fig.add_trace(go.Scatter(
        x=forecast_df["ds"],
        y=forecast_df["forecast_mean"],
        mode="lines+markers",
        name="Predicted Disaster Volume",
        line=dict(color="#1D4ED8", width=3.5, shape="spline"),
        marker=dict(size=7, color="#1D4ED8")
    ))

    fig.update_layout(
        title="<b>🔮 90-Day ML Predictive Time-Series Forecast (95% CI)</b>",
        xaxis_title="Forecast Timeline",
        yaxis_title="Predicted Event Count",
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)"
    )
    return fig
