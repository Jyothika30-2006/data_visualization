"""
Natural Disaster Intelligence & Emergency Response Dashboard
Main Streamlit Application File (app.py)

Course Project: B.Tech Data Science
Features: Auto-detects disaster category from uploaded datasets, flexible column auto-mapping,
20+ Plotly & Folium interactive visualizations, 90-day ML time-series forecasting (95% CI),
composite risk index scoring, and citizen safety portal.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys
import folium

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import APP_TITLE, APP_SUBTITLE, DEFAULT_LAT, DEFAULT_LON
from src.disaster_detector import auto_detect_disaster_type, auto_map_columns
from src.data_processor import clean_and_preprocess_dataset
from src.risk_calculator import classify_risk_level, calculate_top_hotspots
from src.predictor import forecast_90_day_disaster_events, train_disaster_severity_regressor
from src.geo_analyzer import create_gis_heatmap, create_marker_cluster_map
from src.citizen_features import find_nearest_shelters, submit_citizen_report, geocode_location_name
from src.utils import generate_json_summary_report, get_emergency_contacts_directory, get_safety_guidelines
from src.visualizer import (
    plot_time_series_trends, plot_seasonal_pattern, plot_severity_distribution,
    plot_socioeconomic_impact, plot_correlation_heatmap, plot_risk_gauge,
    plot_category_distribution, plot_casualties_vs_severity_scatter,
    plot_treemap_impact, plot_top_hotspots_bar, plot_depth_elevation_profile,
    plot_cumulative_frequency, plot_anomaly_scatter, plot_radar_vulnerability,
    plot_multi_hazard_bubble, plot_forecast_chart
)

from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# Streamlit Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Disaster-Themed CSS
st.markdown("""
<style>
    .stApp {
        background-color: #F8FAFC;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #2563EB 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 1.8rem;
    }
    .hero-title {
        font-size: 2.3rem;
        font-weight: 900;
        letter-spacing: -0.5px;
        margin-bottom: 0.4rem;
        background: linear-gradient(90deg, #FFFFFF, #93C5FD);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #E2E8F0;
        margin-bottom: 0.8rem;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        background: rgba(16, 185, 129, 0.2);
        border: 1px solid rgba(16, 185, 129, 0.5);
        color: #34D399;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        margin-right: 8px;
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
        animation: pulse 1.8s infinite;
    }
    @keyframes pulse {
        0% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { transform: scale(1); box-shadow: 0 0 0 8px rgba(16, 185, 129, 0); }
        100% { transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    .kpi-card {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        border-top: 5px solid #2563EB;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease;
    }
    .kpi-card:hover { transform: translateY(-4px); }
    .kpi-red { border-top-color: #EF4444; }
    .kpi-gold { border-top-color: #F59E0B; }
    .kpi-green { border-top-color: #10B981; }
    .kpi-purple { border-top-color: #8B5CF6; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        border-radius: 8px;
        font-weight: 700;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Sidebar & Dataset Loader
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/siren.png", width=65)
    st.markdown("### 📤 Dataset Input & Settings")

    uploaded_file = st.file_uploader(
        "Upload Custom Disaster Dataset (.csv, .xlsx)",
        type=["csv", "xlsx"]
    )

    st.write("---")
    st.markdown("#### 📁 Fast Sample Datasets")
    sample_choice = st.selectbox(
        "Or load pre-seeded dataset:",
        [
            "None (Use Uploaded)",
            "Earthquake Dataset (1,050 records)",
            "Flood Dataset (520 records)",
            "Cyclone Dataset (220 records)",
            "Tsunami Dataset (150 records)",
            "Multi-Disaster Combined (1,500 records)"
        ]
    )

    map_tile_choice = st.selectbox("🗺️ Map Style", ["CartoDB positron", "OpenStreetMap"])
    min_risk_threshold = st.slider("⚡ Risk Score Filter Threshold", 0, 100, 0, step=5)


# Determine Dataset Source
raw_df = None
dataset_name = "Custom Upload"

if uploaded_file is not None:
    dataset_name = uploaded_file.name
    if uploaded_file.name.endswith(".csv"):
        raw_df = pd.read_csv(uploaded_file)
    else:
        raw_df = pd.read_excel(uploaded_file)
elif sample_choice != "None (Use Uploaded)":
    sample_file_map = {
        "Earthquake Dataset (1,050 records)": "sample_datasets/earthquake_dataset.csv",
        "Flood Dataset (520 records)": "sample_datasets/flood_dataset.csv",
        "Cyclone Dataset (220 records)": "sample_datasets/cyclone_dataset.csv",
        "Tsunami Dataset (150 records)": "sample_datasets/tsunami_dataset.csv",
        "Multi-Disaster Combined (1,500 records)": "sample_datasets/multidisaster_dataset.csv"
    }
    file_path = sample_file_map[sample_choice]
    if os.path.exists(file_path):
        raw_df = pd.read_csv(file_path)
        dataset_name = sample_choice

if raw_df is None or raw_df.empty:
    # Default fallback to multi-disaster
    raw_df = pd.read_csv("sample_datasets/multidisaster_dataset.csv")
    dataset_name = "Default Multi-Disaster Stream"


# Auto-Detection & Processing Engine
mapped_cols = auto_map_columns(raw_df)
detected_disaster_type, confidence = auto_detect_disaster_type(raw_df, mapped_cols)
clean_df, meta_summary = clean_and_preprocess_dataset(raw_df, mapped_cols, detected_disaster_type)


# Apply Sidebar Threshold Filter
if min_risk_threshold > 0:
    clean_df = clean_df[clean_df["risk_score"] >= min_risk_threshold]


# Hero Header Banner
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")
st.markdown(f"""
<div class='hero-banner'>
    <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;'>
        <div>
            <div class='hero-title'>🚨 Natural Disaster Intelligence Dashboard</div>
            <div class='hero-subtitle'>Automated Dataset Intelligence, Predictive Hazard Modeling & Citizen Emergency Response</div>
            <div style='font-size: 0.9rem; background: rgba(255,255,255,0.15); padding: 4px 12px; border-radius: 6px; display: inline-block;'>
                Active File: <b>{dataset_name}</b> | Auto-Detected Category: <b style='color: #FDE047;'>{detected_disaster_type}</b> (Confidence: {int(confidence*100)}%)
            </div>
        </div>
        <div>
            <div class='status-badge'>
                <span class='pulse-dot'></span> INTELLIGENCE ENGINE ACTIVE
            </div>
            <div style='font-size: 0.8rem; color: #94A3B8; text-align: right; margin-top: 6px;'>{current_time}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Main Application Navigation (7 Tabs)
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📊 Overview",
    "🗺️ Geographic Analysis",
    "📈 Trends & Seasonality",
    "💰 Impact & Correlation",
    "⚡ Risk Assessment",
    "🔮 ML Predictions & Forecasting",
    "👤 For Citizens & Safety"
])


# =====================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =====================================================================
with tab1:
    st.markdown("### 📊 Executive Summary & Dataset Telemetry")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(f"""
        <div class='kpi-card kpi-blue'>
            <div style='font-size: 0.85rem; color: #64748B;'>Total Processed Events</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #0284C7;'>{meta_summary.get('clean_rows', 0):,}</div>
            <div style='font-size: 0.75rem; color: #10B981;'>Cleaned ({meta_summary.get('duplicates_removed', 0)} duplicates dropped)</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        risk_label, risk_color = classify_risk_level(meta_summary.get('peak_risk_score', 0.0))
        st.markdown(f"""
        <div class='kpi-card kpi-red'>
            <div style='font-size: 0.85rem; color: #64748B;'>Peak Hazard Risk Score</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: {risk_color};'>{meta_summary.get('peak_risk_score', 0.0)} / 100</div>
            <div style='font-size: 0.75rem; color: {risk_color}; font-weight: 600;'>Status: {risk_label}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card kpi-gold'>
            <div style='font-size: 0.85rem; color: #64748B;'>Average Risk Severity</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #D97706;'>{meta_summary.get('avg_risk_score', 0.0)} / 100</div>
            <div style='font-size: 0.75rem; color: #64748B;'>Across all locations</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card kpi-purple'>
            <div style='font-size: 0.85rem; color: #64748B;'>Total Direct Casualties</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #8B5CF6;'>{meta_summary.get('total_casualties', 0):,}</div>
            <div style='font-size: 0.75rem; color: #8B5CF6;'>Logged across dataset</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='kpi-card kpi-green'>
            <div style='font-size: 0.85rem; color: #64748B;'>Economic Damage ($M)</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #10B981;'>${meta_summary.get('total_economic_loss_m', 0.0):,}M</div>
            <div style='font-size: 0.75rem; color: #10B981;'>Estimated infrastructure loss</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("📋 Cleaned Data Log Preview & Export")
        st.dataframe(clean_df[["timestamp", "location_name", "risk_score", "casualties", "economic_loss_millions"]].head(15), use_container_width=True)

        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            csv_bytes = clean_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Cleaned Dataset (CSV)", data=csv_bytes, file_name="cleaned_disaster_data.csv", mime="text/csv", type="primary")
        with col_ex2:
            json_report = generate_json_summary_report(clean_df, meta_summary)
            st.download_button("📄 Export Summary Intelligence Report (JSON)", data=json_report, file_name="disaster_summary_report.json", mime="text/json")

    with col_right:
        st.subheader(" Peak Risk Gauge")
        gauge_fig = plot_risk_gauge(meta_summary.get('peak_risk_score', 0.0), "Peak Hazard Index")
        st.plotly_chart(gauge_fig, use_container_width=True)


# =====================================================================
# TAB 2: GEOGRAPHIC ANALYSIS
# =====================================================================
with tab2:
    st.subheader("🗺️ Spatial GIS Geographic Risk Heatmaps & Clusters")
    st.caption("Interactive spatial visualization auto-detecting dataset coordinates or geocoding location landmarks.")

    col_m1, col_m2 = st.columns([1, 1])

    with col_m1:
        st.markdown("### 🔥 Spatial Hazard Density Heatmap")
        m_heat = create_gis_heatmap(clean_df, map_tile=map_tile_choice)
        st_folium(m_heat, width=600, height=450)

    with col_m2:
        st.markdown("### 📍 Interactive Incident Cluster Map")
        m_cluster = create_marker_cluster_map(clean_df)
        st_folium(m_cluster, width=600, height=450)


# =====================================================================
# TAB 3: TRENDS & SEASONALITY
# =====================================================================
with tab3:
    st.subheader("📈 Time Series Incident Trends & Seasonal Behavior")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        fig_ts = plot_time_series_trends(clean_df)
        st.plotly_chart(fig_ts, use_container_width=True)

    with col_t2:
        fig_season = plot_seasonal_pattern(clean_df)
        st.plotly_chart(fig_season, use_container_width=True)

    st.divider()

    col_t3, col_t4 = st.columns(2)
    with col_t3:
        fig_cum = plot_cumulative_frequency(clean_df)
        st.plotly_chart(fig_cum, use_container_width=True)

    with col_t4:
        fig_depth = plot_depth_elevation_profile(clean_df)
        st.plotly_chart(fig_depth, use_container_width=True)


# =====================================================================
# TAB 4: IMPACT & CORRELATION ANALYSIS
# =====================================================================
with tab4:
    st.subheader("💰 Socio-Economic Impact Analysis & Correlations")

    col_i1, col_i2 = st.columns(2)
    with col_i1:
        fig_impact = plot_socioeconomic_impact(clean_df)
        st.plotly_chart(fig_impact, use_container_width=True)

    with col_i2:
        fig_corr = plot_correlation_heatmap(clean_df)
        st.plotly_chart(fig_corr, use_container_width=True)

    st.divider()

    col_i3, col_i4 = st.columns(2)
    with col_i3:
        fig_tree = plot_treemap_impact(clean_df)
        st.plotly_chart(fig_tree, use_container_width=True)

    with col_i4:
        fig_anom = plot_anomaly_scatter(clean_df)
        st.plotly_chart(fig_anom, use_container_width=True)


# =====================================================================
# TAB 5: RISK ASSESSMENT
# =====================================================================
with tab5:
    st.subheader("⚡ Regional Risk Assessment & Hotspot Rankings")

    col_r1, col_r2 = st.columns(2)
    with col_r1:
        fig_hotspots = plot_top_hotspots_bar(clean_df)
        st.plotly_chart(fig_hotspots, use_container_width=True)

    with col_r2:
        fig_cat = plot_category_distribution(clean_df)
        st.plotly_chart(fig_cat, use_container_width=True)

    st.divider()

    col_r3, col_r4 = st.columns(2)
    with col_r3:
        fig_radar = plot_radar_vulnerability(clean_df)
        st.plotly_chart(fig_radar, use_container_width=True)

    with col_r4:
        st.markdown("### 🔥 Top High-Risk Hotspots Leaderboard")
        top_df = calculate_top_hotspots(clean_df, top_n=10)
        st.dataframe(top_df, use_container_width=True)


# =====================================================================
# TAB 6: ML PREDICTIONS & FORECASTING
# =====================================================================
with tab6:
    st.subheader("🔮 90-Day ML Predictive Forecasting & Machine Learning Models")

    col_p1, col_p2 = st.columns([1.2, 1])

    with col_p1:
        st.markdown("### 🔮 90-Day Time-Series Event Volume Forecast")
        forecast_df = forecast_90_day_disaster_events(clean_df, days_ahead=90)
        fig_forecast = plot_forecast_chart(forecast_df)
        st.plotly_chart(fig_forecast, use_container_width=True)

    with col_p2:
        st.markdown("### 🤖 Train Machine Learning Risk Predictor")
        rf_model, metrics, avail_features = train_disaster_severity_regressor(clean_df)

        if metrics:
            st.success("✅ Random Forest Regressor trained on uploaded dataset features!")
            st.json(metrics)
            st.write(f"Model Feature Inputs: `{', '.join(avail_features)}`")

            # Interactive Predictor Test
            st.markdown("#### Test Risk Prediction:")
            test_val = st.slider("Adjust Primary Feature Magnitude/Scale", 1.0, 10.0, 6.0)
            if rf_model and avail_features:
                sample_input = pd.DataFrame([{f: test_val if f in ["magnitude", "rainfall_mm", "wind_speed_kmh"] else 10.0 for f in avail_features}])
                pred_risk = round(float(rf_model.predict(sample_input)[0]), 1)
                st.metric("Predicted Hazard Risk Score", f"{pred_risk} / 100")


# =====================================================================
# TAB 7: FOR CITIZENS & SAFETY PORTAL
# =====================================================================
with tab7:
    st.subheader("👤 Citizen Emergency Safety Portal")

    col_c1, col_c2 = st.columns([1, 1])

    with col_c1:
        st.markdown("### 📍 Location Risk Checker & Shelter Finder")
        c_loc_input = st.text_input("Enter City, Landmark, or District", value="Mumbai Kurla")

        if st.button("🔍 Check Nearby Risk & Emergency Shelters", type="primary", use_container_width=True):
            shelter_df = query_to_df("SELECT * FROM shelters")
            nearest_df, resolved_name = find_nearest_shelters(c_loc_input, top_n=5)
            st.success(f"Emergency Status for **{resolved_name}**:")
            if not nearest_df.empty:
                st.dataframe(nearest_df[["name", "region", "capacity", "available_space", "distance_km", "contact_number"]], use_container_width=True)

        st.divider()

        st.markdown("### 📢 Report Emergency Disaster Incident")
        with st.form("citizen_report_form"):
            rep_name = st.text_input("Reporter Name")
            rep_phone = st.text_input("Contact Phone")
            rep_disaster = st.selectbox("Hazard Type", ["Earthquake", "Flood", "Cyclone", "Tsunami", "Landslide"])
            rep_loc = st.text_input("Location Landmark", "Kurla West, Mumbai")
            rep_desc = st.text_area("Incident Description")
            
            if st.form_submit_button("Submit Emergency Report", type="primary"):
                submit_citizen_report(rep_name, rep_phone, rep_disaster, rep_loc, description=rep_desc)
                st.success("Emergency report dispatched to Control Room!")

    with col_c2:
        st.markdown("### 📞 Emergency Helpline Directory")
        st.dataframe(get_emergency_contacts_directory(), use_container_width=True)

        st.divider()

        st.markdown("### 📋 Citizen Disaster Survival Guidelines")
        guidelines = get_safety_guidelines()
        for g_title, g_items in guidelines.items():
            with st.expander(f"🛡️ {g_title}"):
                for item in g_items:
                    st.write(item)
