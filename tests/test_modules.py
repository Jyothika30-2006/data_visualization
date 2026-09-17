"""
Unit Test Suite for Natural Disaster Intelligence Dashboard.
Tests API fetching, risk algorithms, ML models, alert system, and citizen features.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import init_db, query_to_df, get_connection
from src.data_collection import fetch_live_usgs_earthquakes, fetch_realtime_weather
from src.data_processing import (
    calculate_earthquake_risk_score, calculate_flood_risk_score,
    calculate_cyclone_risk_score, clean_disaster_data
)
from src.ml_models import EarthquakePredictionModel, FloodPredictionModel, forecast_disaster_time_series
from src.citizen_features import haversine_distance, find_nearest_shelters
from src.alert_system import generate_disaster_alert_template, send_sms_alert, send_email_alert
from src.route_optimization import compute_safe_evacuation_path


def test_database_initialization(tmp_path):
    test_db = str(tmp_path / "test_disaster.db")
    init_db(test_db)
    assert os.path.exists(test_db)
    conn = get_connection(test_db)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    assert "earthquakes" in tables
    assert "floods" in tables
    assert "shelters" in tables


def test_risk_calculation_algorithms():
    eq_risk = calculate_earthquake_risk_score(magnitude=7.5, depth_km=10.0, population_density=2000.0)
    assert 0 <= eq_risk <= 100
    assert eq_risk > 50.0  # High magnitude + shallow depth should yield high score

    fl_risk = calculate_flood_risk_score(rainfall_mm=250.0, river_level_m=7.0, threshold_m=5.0)
    assert 0 <= fl_risk <= 100
    assert fl_risk > 40.0

    cy_risk = calculate_cyclone_risk_score(wind_speed_kmh=210.0, pressure_hpa=940.0)
    assert 0 <= cy_risk <= 100
    assert cy_risk > 60.0


def test_haversine_distance():
    # Mumbai to Pune (~120 km)
    dist = haversine_distance(19.0760, 72.8777, 18.5204, 73.8567)
    assert 110.0 <= dist <= 140.0


def test_ml_models_training_and_prediction():
    # Generate dummy training dataframe
    eq_data = pd.DataFrame([{
        "event_id": f"EQ-{i}",
        "magnitude": float(np.random.uniform(3.0, 8.0)),
        "depth": float(np.random.uniform(5.0, 80.0)),
        "latitude": 20.0 + i*0.01,
        "longitude": 78.0 + i*0.01,
        "location": "Test Loc",
        "timestamp": "2024-01-01 00:00:00",
        "risk_score": float(np.random.uniform(20.0, 95.0)),
        "tsunami_warning": 0,
        "casualties": 10,
        "economic_loss_millions": 5.0
    } for i in range(50)])

    model = EarthquakePredictionModel()
    metrics = model.train(eq_data)
    assert "rmse" in metrics
    assert "r2_score" in metrics

    pred = model.predict({"magnitude": 7.0, "depth": 15.0})
    assert "predicted_risk_score" in pred
    assert "confidence_interval_95" in pred
    assert pred["predicted_risk_score"] >= 0.0


def test_alert_template_and_dispatch():
    template = generate_disaster_alert_template("Flood", "Assam", 85.0, "Heavy rain 250mm")
    assert "EMERGENCY ALERT" in template["subject"]
    assert "85.0" in template["sms_body"]

    assert send_sms_alert("+919876543210", template["sms_body"]) is True
    assert send_email_alert("test@example.com", template["subject"], template["email_body"]) is True


def test_evacuation_route_optimization():
    origin_lat, origin_lon = 19.0760, 72.8777
    shelters = [
        {"name": "Shelter A", "latitude": 19.1000, "longitude": 72.9000, "address": "Addr A"},
        {"name": "Shelter B", "latitude": 19.2000, "longitude": 72.9500, "address": "Addr B"}
    ]
    hazard_zones = [{"lat": 19.0800, "lon": 72.8800, "risk_score": 90.0}]

    route = compute_safe_evacuation_path(origin_lat, origin_lon, hazard_zones, shelters)
    assert route["status"] == "Success"
    assert route["target_shelter_name"] in ["Shelter A", "Shelter B"]
    assert route["distance_km"] > 0
