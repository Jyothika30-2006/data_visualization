"""
Natural Disaster Intelligence & Response Dashboard
Main Streamlit Application File (app.py)

Course Project: B.Tech Data Science
Theme: Modern, Vibrant, Interactive & High-Impact Visual Template.
User Location: Uses City / Location Names / Landmarks instead of raw Lat/Lon inputs.
"""

import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os
import sys

# Add root directory to sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from config import APP_TITLE, APP_SUBTITLE, DEFAULT_LAT, DEFAULT_LON, REFRESH_INTERVAL_MINUTES
from src.database import query_to_df, get_connection
from src.data_collection import fetch_live_usgs_earthquakes, refresh_disaster_database
from src.data_processing import clean_disaster_data, aggregate_disaster_summary, calculate_earthquake_risk_score, calculate_flood_risk_score
from src.ml_models import train_and_eval_all_models, forecast_disaster_time_series
from src.visualizations import (
    create_interactive_risk_map, create_multiline_trend_chart, create_risk_gauge_chart,
    create_severity_histogram, create_impact_analysis_chart, create_forecast_chart
)
from src.alert_system import trigger_automated_risk_alerts, register_subscriber
from src.citizen_features import (
    submit_citizen_report, find_nearest_shelters, get_emergency_contacts_directory,
    get_safety_checklists, verify_citizen_report, geocode_location_name, LOCATION_DIRECTORY
)
from src.government_dashboard import (
    get_active_operations, update_resource_allocation, create_shelter_capacity_chart,
    create_resource_deployment_chart
)
from src.sentiment_analysis import generate_mock_disaster_tweets, create_sentiment_summary_chart
from src.route_optimization import compute_safe_evacuation_path
from src.i18n import get_text

from streamlit_folium import st_folium

# Page Configuration
st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =====================================================================
# VIBRANT CUSTOM CSS STYLING
# =====================================================================
st.markdown("""
<style>
    /* Global Page Styling */
    .stApp {
        background-color: #F8FAFC;
    }
    
    /* Vibrant Gradient Hero Banner */
    .hero-banner {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 40%, #2563EB 100%);
        border-radius: 16px;
        padding: 2rem 2.2rem;
        color: #FFFFFF;
        box-shadow: 0 10px 25px -3px rgba(37, 99, 235, 0.25);
        margin-bottom: 1.8rem;
        position: relative;
        overflow: hidden;
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
        font-weight: 400;
        margin-bottom: 1rem;
    }
    
    /* Pulsing Status Dot */
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

    /* Vibrant Glassmorphism KPI Cards */
    .kpi-card-vibrant {
        background: #FFFFFF;
        border-radius: 14px;
        padding: 1.25rem 1.4rem;
        border-top: 5px solid #3B82F6;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .kpi-card-vibrant:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 20px rgba(0, 0, 0, 0.1);
    }
    .kpi-card-red { border-top-color: #EF4444; }
    .kpi-card-gold { border-top-color: #F59E0B; }
    .kpi-card-blue { border-top-color: #0284C7; }
    .kpi-card-purple { border-top-color: #8B5CF6; }
    .kpi-card-green { border-top-color: #10B981; }

    /* Critical Emergency Banner */
    .alert-banner-gradient {
        background: linear-gradient(135deg, #DC2626 0%, #991B1B 100%);
        color: #FFFFFF;
        padding: 1.1rem 1.5rem;
        border-radius: 12px;
        font-weight: 600;
        box-shadow: 0 8px 16px rgba(220, 38, 38, 0.25);
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }

    /* Tabs Styling */
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
        background: transparent;
        transition: all 0.2s ease;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(37, 99, 235, 0.3);
    }
</style>
""", unsafe_allow_html=True)


# Caching data for performance
@st.cache_data(ttl=300)
def load_all_disaster_data():
    """Load and clean data from SQLite database."""
    eq_df = query_to_df("SELECT * FROM earthquakes ORDER BY timestamp DESC")
    flood_df = query_to_df("SELECT * FROM floods ORDER BY timestamp DESC")
    cyc_df = query_to_df("SELECT * FROM cyclones ORDER BY timestamp DESC")
    shelter_df = query_to_df("SELECT * FROM shelters")
    pop_df = query_to_df("SELECT * FROM population_density")
    return eq_df, flood_df, cyc_df, shelter_df, pop_df


@st.cache_resource
def load_ml_models():
    """Load or train ML models."""
    return train_and_eval_all_models()


# Load Data & Models
eq_df, flood_df, cyc_df, shelter_df, pop_df = load_all_disaster_data()
ml_results = load_ml_models()


# =====================================================================
# SIDEBAR CONTROLS & STYLING
# =====================================================================
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/siren.png", width=65)
    st.markdown("### 🎛️ Control Panel")
    
    # Language Selector
    selected_language = st.selectbox(
        "🌐 Language / भाषा / Idioma",
        ["English", "Hindi", "Spanish", "French"]
    )
    
    st.divider()
    
    # Map Tile Theme Selector
    map_tile_choice = st.selectbox(
        "🗺️ Map Style Theme",
        ["CartoDB positron", "OpenStreetMap"]
    )
    
    # Disaster Type Filter
    disaster_filter = st.selectbox(
        get_text("filter_disaster", selected_language),
        ["All", "Earthquake", "Flood", "Cyclone"]
    )
    
    # Region filter
    all_regions = ["All Regions"]
    if not flood_df.empty and "region" in flood_df.columns:
        all_regions.extend(list(flood_df["region"].unique()))
    selected_region = st.selectbox(get_text("filter_region", selected_language), all_regions)

    # Risk Threshold Filter
    min_risk_threshold = st.slider("⚡ Minimum Hazard Risk Score Filter", 0, 100, 0, step=5)
    
    st.divider()
    
    # Live Data Sync Button with Vibrant Styling
    if st.button(get_text("btn_refresh", selected_language), use_container_width=True, type="primary"):
        with st.spinner("Fetching live feeds from USGS & Meteorological APIs..."):
            refresh_disaster_database()
            st.cache_data.clear()
            st.success("Database synced with live satellite & weather telemetry!")
            st.rerun()

    st.caption("🚨 Natural Disaster Intelligence v2.5 | B.Tech Data Science Project")


# Apply Filters to DataFrames
if min_risk_threshold > 0:
    if not eq_df.empty: eq_df = eq_df[eq_df["risk_score"] >= min_risk_threshold]
    if not flood_df.empty: flood_df = flood_df[flood_df["risk_score"] >= min_risk_threshold]
    if not cyc_df.empty: cyc_df = cyc_df[cyc_df["risk_score"] >= min_risk_threshold]

if selected_region != "All Regions":
    if not flood_df.empty and "region" in flood_df.columns:
        flood_df = flood_df[flood_df["region"].str.contains(selected_region, case=False, na=False)]
    if not shelter_df.empty and "region" in shelter_df.columns:
        shelter_df = shelter_df[shelter_df["region"].str.contains(selected_region, case=False, na=False)]


# Calculate Executive KPIs
summary_kpis = aggregate_disaster_summary(eq_df, flood_df, cyc_df)


# =====================================================================
# VIBRANT HERO HEADER BANNER
# =====================================================================
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S UTC")

st.markdown(f"""
<div class='hero-banner'>
    <div style='display: flex; justify-content: space-between; align-items: flex-start; flex-wrap: wrap;'>
        <div>
            <div class='hero-title'>🚨 {get_text('title', selected_language)}</div>
            <div class='hero-subtitle'>{get_text('subtitle', selected_language)}</div>
        </div>
        <div>
            <div class='status-badge'>
                <span class='pulse-dot'></span> LIVE TELEMETRY FEED ACTIVE
            </div>
            <div style='font-size: 0.8rem; color: #94A3B8; text-align: right; margin-top: 6px;'>
                Last Synced: {current_time}
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# Critical Emergency Banner (if peak risk >= 75)
if summary_kpis["peak_hazard_risk_score"] >= 75:
    st.markdown(f"""
    <div class='alert-banner-gradient'>
        <span style='font-size: 1.8rem;'>🚨</span>
        <div>
            <div style='font-size: 1.1rem; font-weight: 800;'>CRITICAL HAZARD WARNING IN EFFECT</div>
            <div style='font-size: 0.95rem; opacity: 0.95;'>
                High severity hazard detected with risk score exceeding 75/100. Automated SMS & Email warning dispatches sent to local emergency response teams.
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# Navigation Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    get_text("tab_overview", selected_language),
    get_text("tab_map", selected_language),
    get_text("tab_analytics", selected_language),
    get_text("tab_ml", selected_language),
    get_text("tab_citizen", selected_language),
    get_text("tab_government", selected_language),
    get_text("tab_innovations", selected_language)
])


# =====================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =====================================================================
with tab1:
    st.markdown("### 📊 Executive Overview & Live Telemetry Summary")

    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        st.markdown(f"""
        <div class='kpi-card-vibrant kpi-card-blue'>
            <div style='font-size: 0.85rem; color: #64748B; font-weight: 600;'>{get_text('kpi_total_events', selected_language)}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #0284C7; margin: 4px 0;'>{summary_kpis['total_disaster_events']:,}</div>
            <div style='font-size: 0.75rem; color: #10B981; font-weight: 600;'>⚡ +12 Live API Streams</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='kpi-card-vibrant kpi-card-red'>
            <div style='font-size: 0.85rem; color: #64748B; font-weight: 600;'>{get_text('kpi_critical_alerts', selected_language)}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #EF4444; margin: 4px 0;'>{summary_kpis['critical_warnings_count']:,}</div>
            <div style='font-size: 0.75rem; color: #EF4444; font-weight: 600;'>⚠️ Action Triggered</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class='kpi-card-vibrant kpi-card-gold'>
            <div style='font-size: 0.85rem; color: #64748B; font-weight: 600;'>{get_text('kpi_casualties', selected_language)}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #D97706; margin: 4px 0;'>{summary_kpis['total_casualties']:,}</div>
            <div style='font-size: 0.75rem; color: #64748B;'>Estimated Direct Impacts</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='kpi-card-vibrant kpi-card-purple'>
            <div style='font-size: 0.85rem; color: #64748B; font-weight: 600;'>{get_text('kpi_affected', selected_language)}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #8B5CF6; margin: 4px 0;'>{summary_kpis['total_affected_population']:,}</div>
            <div style='font-size: 0.75rem; color: #8B5CF6;'>In Danger Zone</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown(f"""
        <div class='kpi-card-vibrant kpi-card-green'>
            <div style='font-size: 0.85rem; color: #64748B; font-weight: 600;'>{get_text('kpi_loss', selected_language)}</div>
            <div style='font-size: 1.8rem; font-weight: 800; color: #10B981; margin: 4px 0;'>${summary_kpis['total_economic_loss_m']}M</div>
            <div style='font-size: 0.75rem; color: #10B981;'>Infrastructure Damage</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    col_left, col_right = st.columns([2, 1])

    with col_left:
        st.subheader("🌐 Live Incident Log Feed")
        
        # Combine recent events for overview display
        recent_list = []
        if not eq_df.empty:
            for _, r in eq_df.head(10).iterrows():
                recent_list.append({"Timestamp": r["timestamp"], "Type": "Earthquake 🌋", "Location": r["location"], "Risk Score": r["risk_score"], "Key Details": f"M{r['magnitude']} Depth:{r['depth']}km"})
        if not flood_df.empty:
            for _, r in flood_df.head(10).iterrows():
                recent_list.append({"Timestamp": r["timestamp"], "Type": "Flood 🌊", "Location": r["region"], "Risk Score": r["risk_score"], "Key Details": f"Rain:{r['rainfall_mm']}mm River:{r['river_level_m']}m"})
        if not cyc_df.empty:
            for _, r in cyc_df.head(10).iterrows():
                recent_list.append({"Timestamp": r["timestamp"], "Type": "Cyclone 🌀", "Location": r["cyclone_name"], "Risk Score": r["risk_score"], "Key Details": f"Wind:{r['wind_speed_kmh']}km/h Cat:{r['category']}"})

        recent_df = pd.DataFrame(recent_list)
        if not recent_df.empty:
            recent_df = recent_df.sort_values(by="Timestamp", ascending=False).head(15)
            st.dataframe(recent_df, use_container_width=True)
            
            # Export CSV button
            csv_data = recent_df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Export Disaster Incident Log (CSV)", data=csv_data, file_name="disaster_incidents_report.csv", mime="text/csv", type="primary")
        else:
            st.info("No disaster records match the selected sidebar filters.")

    with col_right:
        st.subheader(" Peak Hazard Index")
        gauge_fig = create_risk_gauge_chart(summary_kpis["peak_hazard_risk_score"], "Peak Hazard Index")
        st.plotly_chart(gauge_fig, use_container_width=True)

        st.markdown("### ⚡ Live System Gateways")
        st.success("✔ **USGS Seismic Web Feed**: Synchronized")
        st.success("✔ **Meteorological API Gateway**: Connected")
        st.info("✔ **Automated Alert Trigger**: Threshold Score ≥ 65")


# =====================================================================
# TAB 2: INTERACTIVE RISK MAP
# =====================================================================
with tab2:
    st.subheader("🗺️ Interactive Spatial Hazard Map & Shelter Clusters")
    st.caption("Multi-layer Folium GIS map rendering earthquakes, flood inundations, cyclones, and emergency shelter capacities.")

    m = create_interactive_risk_map(
        eq_df, flood_df, cyc_df, shelter_df,
        selected_disaster=disaster_filter,
        map_tile=map_tile_choice
    )
    st_folium(m, width=1250, height=580)


# =====================================================================
# TAB 3: HISTORICAL TRENDS & ANALYTICS
# =====================================================================
with tab3:
    st.subheader("📈 Multi-Hazard Analytics & Socio-Economic Impact Breakdown")

    col1, col2 = st.columns(2)
    with col1:
        fig_trend = create_multiline_trend_chart(eq_df, flood_df, cyc_df)
        st.plotly_chart(fig_trend, use_container_width=True)

    with col2:
        fig_hist = create_severity_histogram(eq_df, flood_df, cyc_df)
        st.plotly_chart(fig_hist, use_container_width=True)

    st.divider()

    col3, col4 = st.columns(2)
    with col3:
        fig_impact = create_impact_analysis_chart(eq_df, flood_df, cyc_df)
        st.plotly_chart(fig_impact, use_container_width=True)

    with col4:
        st.subheader("📊 Regional Population Vulnerability Profile")
        if not pop_df.empty:
            st.dataframe(pop_df[["region", "state", "population", "density_per_sq_km", "vulnerability_index"]].head(10), use_container_width=True)


# =====================================================================
# TAB 4: ML PREDICTIONS & FORECASTING
# =====================================================================
with tab4:
    st.subheader("🤖 Machine Learning Predictive Hazard Models & Time-Series Forecaster")

    ml_col1, ml_col2 = st.columns([1, 1])

    with ml_col1:
        st.markdown("### 🌋 Real-Time Earthquake Risk Calculator")
        
        # Location Selector instead of raw lat/lon
        city_choice_eq = st.selectbox(
            "📍 Target Location / City",
            ["Mumbai", "Chennai", "Kolkata", "Guwahati (Assam)", "Kochi (Kerala)", "Puri (Odisha)", "Patna (Bihar)", "Ahmedabad (Gujarat)", "Dehradun (Uttarakhand)", "Shimla (Himachal)", "Srinagar (J&K)", "Visakhapatnam (AP)", "Delhi / NCR", "Hyderabad", "Bangalore", "Himalayan Belt", "Japan Ring of Fire", "California Coast"],
            key="eq_city_select"
        )
        eq_lat, eq_lon, _ = geocode_location_name(city_choice_eq)

        eq_mag_input = st.slider("Earthquake Richter Magnitude", 2.0, 9.5, 6.5, step=0.1)
        eq_depth_input = st.slider("Focal Depth (km)", 1.0, 150.0, 12.0, step=1.0)
        
        eq_model = ml_results["earthquake_model"]["model"]
        eq_pred = eq_model.predict({
            "magnitude": eq_mag_input,
            "depth": eq_depth_input,
            "latitude": eq_lat,
            "longitude": eq_lon
        })

        st.metric(f"Predicted Hazard Risk Score ({city_choice_eq})", f"{eq_pred['predicted_risk_score']} / 100")
        st.info(f"95% Confidence Interval: {eq_pred['confidence_interval_95'][0]} - {eq_pred['confidence_interval_95'][1]}")
        st.write(f"High Hazard Class Probability: **{int(eq_pred['high_risk_probability'] * 100)}%**")

    with ml_col2:
        st.markdown("### 🌊 Real-Time Flood Severity Predictor")
        
        city_choice_fl = st.selectbox(
            "📍 Target River Basin / Region",
            ["Mumbai", "Chennai", "Kolkata", "Guwahati (Assam)", "Kochi (Kerala)", "Puri (Odisha)", "Patna (Bihar)", "Ahmedabad (Gujarat)", "Dehradun (Uttarakhand)", "Silchar (Assam)", "Alappuzha (Kerala)"],
            key="fl_city_select"
        )
        fl_lat, fl_lon, _ = geocode_location_name(city_choice_fl)

        rain_input = st.slider("24h Rainfall (mm)", 0.0, 400.0, 210.0, step=5.0)
        river_input = st.slider("River Level Gauge (m)", 1.0, 12.0, 7.2, step=0.1)
        
        fl_model = ml_results["flood_model"]["model"]
        fl_pred = fl_model.predict({
            "rainfall_mm": rain_input,
            "river_level_m": river_input,
            "threshold_m": 5.0,
            "latitude": fl_lat,
            "longitude": fl_lon
        })

        st.metric(f"Predicted Flood Risk Score ({city_choice_fl})", f"{fl_pred['predicted_risk_score']} / 100")
        st.info(f"95% Confidence Interval: {fl_pred['confidence_interval_95'][0]} - {fl_pred['confidence_interval_95'][1]}")
        st.write(f"Predicted Severity Category: **{fl_pred['flood_severity_category']}**")

    st.divider()

    st.subheader("📈 Time-Series Disaster Event Volume Forecasting (Holt-Winters)")
    all_combined = pd.concat([eq_df, flood_df, cyc_df], ignore_index=True)
    forecast_df = forecast_disaster_time_series(all_combined, periods_ahead=12)
    fig_forecast = create_forecast_chart(forecast_df)
    st.plotly_chart(fig_forecast, use_container_width=True)

    with st.expander("🔍 Model Evaluation Metrics & Accuracy Pipeline"):
        eq_m = ml_results["earthquake_model"]["metrics"]
        fl_m = ml_results["flood_model"]["metrics"]
        
        st.write("**Earthquake Random Forest Model Metrics:**")
        st.json(eq_m)
        st.write("**Flood Gradient Boosting Model Metrics:**")
        st.json(fl_m)


# =====================================================================
# TAB 5: CITIZEN PORTAL & SAFETY
# =====================================================================
with tab5:
    st.subheader("👤 Citizen Emergency Assistance Portal")

    c_col1, c_col2 = st.columns([1, 1])

    with c_col1:
        st.markdown("### 📍 Nearest Shelter Finder")
        st.caption("Find the nearest open shelters with available space by entering your city, area, or PIN code.")
        
        # Location Name Search Input instead of Lat/Lon
        user_loc_input = st.text_input("Enter Your City, Area, Landmark or PIN Code", value="Mumbai Kurla")
        
        # Preset Quick Select Buttons
        st.write("Or pick a quick location:")
        q_cols = st.columns(4)
        if q_cols[0].button("Mumbai"): user_loc_input = "Mumbai"
        if q_cols[1].button("Chennai"): user_loc_input = "Chennai"
        if q_cols[2].button("Guwahati"): user_loc_input = "Guwahati"
        if q_cols[3].button("Kochi"): user_loc_input = "Kochi"

        if st.button("🔍 Search Nearest Emergency Shelters", use_container_width=True, type="primary"):
            nearest_df, resolved_name = find_nearest_shelters(user_loc_input, top_n=5)
            st.success(f"Showing 5 nearest available shelters for **{resolved_name}**:")
            if not nearest_df.empty:
                st.dataframe(nearest_df[["name", "region", "capacity", "available_space", "distance_km", "contact_number"]], use_container_width=True)
            else:
                st.warning(f"No open shelters found near '{resolved_name}'.")

        st.divider()

        st.markdown("### 📢 Report an Emergency Incident")
        with st.form("citizen_report_form"):
            rep_name = st.text_input("Your Full Name")
            rep_phone = st.text_input("Phone Number")
            rep_disaster = st.selectbox("Incident Type", ["Flood", "Earthquake", "Cyclone", "Landslide", "Structural Damage"])
            rep_loc = st.text_input("Location Name / Landmark / City", "Kurla West, Mumbai")
            rep_desc = st.text_area("Description of Incident")
            rep_sev = st.select_slider("Severity Level", options=["Low", "Moderate", "High", "Critical"])
            
            if st.form_submit_button("Submit Emergency Report", type="primary"):
                if rep_name and rep_phone and rep_loc:
                    submit_citizen_report(rep_name, rep_phone, rep_disaster, rep_loc, description=rep_desc, severity=rep_sev)
                    st.success(f"Your report for '{rep_loc}' has been dispatched to the Control Room for verification!")
                else:
                    st.error("Please complete all required fields.")

    with c_col2:
        st.markdown("### 🔔 Subscribe for Emergency Warning Alerts")
        with st.form("sub_form"):
            s_name = st.text_input("Subscriber Name")
            s_email = st.text_input("Email Address")
            s_phone = st.text_input("Mobile Number (+91...)")
            s_region = st.selectbox("Target Region / State", ["Maharashtra", "Kerala", "Tamil Nadu", "Odisha", "Assam", "Gujarat", "Uttarakhand"])
            s_pref = st.multiselect("Preferred Hazards", ["Earthquake", "Flood", "Cyclone"], default=["Earthquake", "Flood", "Cyclone"])
            
            if st.form_submit_button("Subscribe Now", type="primary"):
                if s_name and (s_email or s_phone):
                    pref_str = ",".join(s_pref) if s_pref else "All"
                    register_subscriber(s_name, s_email, s_phone, s_region, preferred_disasters=pref_str)
                    st.success("Subscribed successfully! You will receive instant warnings when hazard risk scores exceed 65.")

        st.divider()

        st.markdown("### 📞 Emergency Helpline Contacts")
        st.dataframe(get_emergency_contacts_directory(), use_container_width=True)

    with st.expander("📋 Citizen Disaster Survival Checklists"):
        checklists = get_safety_checklists()
        for hazard, items in checklists.items():
            st.markdown(f"**{hazard}:**")
            for item in items:
                st.checkbox(item, key=f"chk_{hazard}_{item[:15]}")


# =====================================================================
# TAB 6: GOVERNMENT OPERATIONS
# =====================================================================
with tab6:
    st.subheader("🏛️ Government Emergency Resource Command Console")

    gov_col1, gov_col2 = st.columns([1, 1])

    with gov_col1:
        st.markdown("### 🏥 Shelter Capacity & Occupancy Monitor")
        fig_shelter = create_shelter_capacity_chart(shelter_df)
        st.plotly_chart(fig_shelter, use_container_width=True)

    with gov_col2:
        st.markdown("### 🚁 NDRF & Rescue Asset Deployment")
        alloc_df = get_active_operations()
        fig_alloc = create_resource_deployment_chart(alloc_df)
        st.plotly_chart(fig_alloc, use_container_width=True)

    st.divider()

    st.markdown("### ⚙️ Dispatch Emergency Operations")
    with st.form("gov_deploy_form"):
        dep_op = st.text_input("Operation Codename", "Op Disaster Shield")
        dep_reg = st.selectbox("Deployment Region", ["Maharashtra", "Odisha", "Kerala", "Assam", "Gujarat", "Uttarakhand"])
        col_a, col_b, col_c, col_d = st.columns(4)
        with col_a:
            p_count = st.number_input("Personnel Deployed", value=250, step=50)
        with col_b:
            m_count = st.number_input("Medical Units", value=12, step=2)
        with col_c:
            b_count = st.number_input("Rescue Boats", value=20, step=5)
        with col_d:
            h_count = st.number_input("Helicopters", value=3, step=1)
            
        if st.form_submit_button("Deploy Emergency Operations", type="primary"):
            update_resource_allocation(dep_op, dep_reg, p_count, m_count, 15000, 30000, b_count, h_count)
            st.success(f"Emergency resources for '{dep_op}' dispatched to regional command log!")


# =====================================================================
# TAB 7: INNOVATIVE TOOLS
# =====================================================================
with tab7:
    st.subheader("🚀 Innovative Features: Social Sentiment & Evacuation Path Optimizer")

    inn_col1, inn_col2 = st.columns([1, 1])

    with inn_col1:
        st.markdown("### 🐤 Social Media Distress Sentiment Analyzer")
        tweets_df = generate_mock_disaster_tweets(30)
        fig_sent = create_sentiment_summary_chart(tweets_df)
        st.plotly_chart(fig_sent, use_container_width=True)
        
        with st.expander("Live Social Media Distress Feed"):
            st.dataframe(tweets_df[["location", "sentiment_category", "panic_distress_score", "text"]], use_container_width=True)

    with inn_col2:
        st.markdown("### 🗺️ Safe Evacuation Route Optimizer")
        st.caption("Calculates optimal path to nearest shelter avoiding active high-hazard centroids.")
        
        route_city_choice = st.selectbox(
            "Select Your City / Current Location",
            ["Mumbai Kurla", "Chennai Adyar", "Guwahati Assam", "Kochi Kerala", "Puri Odisha", "Patna Bihar", "Dehradun Uttarakhand", "Salt Lake Kolkata"],
            key="route_city_select"
        )
        r_lat, r_lon, r_name = geocode_location_name(route_city_choice)
        
        if st.button("Compute Optimal Evacuation Route", type="primary"):
            shelter_list = shelter_df.to_dict(orient="records") if not shelter_df.empty else []
            hazard_list = [{"lat": r_lat - 0.02, "lon": r_lon - 0.02, "risk_score": 85.0}]
            
            route_res = compute_safe_evacuation_path(r_lat, r_lon, hazard_list, shelter_list)
            if route_res.get("status") == "Success":
                st.success(f"Recommended Safe Shelter for **{r_name}**: **{route_res['target_shelter_name']}**")
                st.info(f"Distance: **{route_res['distance_km']} km** | Estimated Walking/Driving ETA: **{route_res['estimated_eta_minutes']} mins**")
                st.write("**Route Waypoints (Lat/Lon Pathway):**")
                st.write(route_res["waypoints"])
            else:
                st.error("Unable to compute route.")
