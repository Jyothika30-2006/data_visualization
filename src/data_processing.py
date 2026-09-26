"""
Module 2: Data Processing Engine
Handles data cleaning, preprocessing, multi-hazard risk scoring algorithms,
feature engineering for ML pipelines, spatial/temporal aggregations,
and dynamic user-uploaded dataset cleaning & visualization pipelines.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.citizen_features import geocode_location_name

logger = logging.getLogger(__name__)


def clean_disaster_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and normalize disaster datasets.
    Handles missing values, drops duplicate IDs, standardizes dates, and handles numeric outliers.
    """
    if df.empty:
        return df

    df = df.copy()

    # Drop duplicates if event_id is present
    if "event_id" in df.columns:
        df = df.drop_duplicates(subset=["event_id"])

    # Ensure timestamp conversion
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())

    # Numeric cleaning & fill missing values with median/0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    return df


def process_and_clean_user_uploaded_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Dynamic dataset cleaner for arbitrary uploaded CSV/Excel/JSON disaster datasets.
    Automatically detects column semantics (magnitude, rainfall, wind, location, timestamp, casualties),
    imputes missing values, geocodes locations if lat/lon are missing, and computes risk scores.
    
    Returns:
        Tuple[pd.DataFrame, Dict[str, Any]]: Cleaned dataframe and metadata summary.
    """
    if df.empty:
        return pd.DataFrame(), {}

    df = df.copy()
    col_map = {str(c).lower().strip(): c for c in df.columns}

    # 1. Detect and standardize timestamp column
    time_col = None
    for candidate in ["timestamp", "date", "datetime", "time", "event_date", "year", "created_at"]:
        if candidate in col_map:
            time_col = col_map[candidate]
            break

    if time_col:
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
        df["timestamp"] = df["timestamp"].fillna(pd.Timestamp.now())
    else:
        df["timestamp"] = pd.Timestamp.now()

    # 2. Detect location / region column
    loc_col = None
    for candidate in ["location", "region", "city", "place", "district", "state", "area", "landmark", "address", "site"]:
        if candidate in col_map:
            loc_col = col_map[candidate]
            break

    if loc_col:
        df["location_name"] = df[loc_col].astype(str)
    else:
        df["location_name"] = "Region " + (df.index + 1).astype(str)

    # 3. Detect or geocode latitude / longitude
    lat_col = col_map.get("latitude", col_map.get("lat"))
    lon_col = col_map.get("longitude", col_map.get("lon", col_map.get("lng")))

    if lat_col and lon_col:
        df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce").fillna(20.5937)
        df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce").fillna(78.9629)
    else:
        lats, lons = [], []
        for loc_str in df["location_name"]:
            lt, ln, _ = geocode_location_name(loc_str)
            lats.append(lt)
            lons.append(ln)
        df["latitude"] = lats
        df["longitude"] = lons

    # 4. Detect disaster hazard metrics & calculate Risk Score if missing
    mag_col = col_map.get("magnitude", col_map.get("mag"))
    rain_col = col_map.get("rainfall_mm", col_map.get("rainfall", col_map.get("rain")))
    wind_col = col_map.get("wind_speed_kmh", col_map.get("wind_speed", col_map.get("wind")))
    risk_col = col_map.get("risk_score", col_map.get("risk", col_map.get("severity_score")))

    if risk_col:
        df["risk_score"] = pd.to_numeric(df[risk_col], errors="coerce").fillna(50.0)
    else:
        # Calculate heuristic risk score based on detected physical metric
        scores = []
        for _, row in df.iterrows():
            if mag_col and pd.notnull(row.get(mag_col)):
                s = calculate_earthquake_risk_score(float(row[mag_col]), 20.0)
            elif rain_col and pd.notnull(row.get(rain_col)):
                s = calculate_flood_risk_score(float(row[rain_col]), 5.5)
            elif wind_col and pd.notnull(row.get(wind_col)):
                s = calculate_cyclone_risk_score(float(row[wind_col]))
            else:
                s = 50.0
            scores.append(s)
        df["risk_score"] = scores

    # 5. Casualties & Loss detection
    cas_col = col_map.get("casualties", col_map.get("deaths", col_map.get("affected")))
    loss_col = col_map.get("economic_loss_millions", col_map.get("loss", col_map.get("damage_millions")))

    df["casualties"] = pd.to_numeric(df[cas_col], errors="coerce").fillna(0).astype(int) if cas_col else 0
    df["economic_loss_millions"] = pd.to_numeric(df[loss_col], errors="coerce").fillna(0.0).astype(float) if loss_col else 0.0

    # Categorize Risk Level
    risk_cats = []
    for r in df["risk_score"]:
        if r >= 75:
            risk_cats.append("Critical Hazard")
        elif r >= 50:
            risk_cats.append("High Warning")
        elif r >= 25:
            risk_cats.append("Moderate Risk")
        else:
            risk_cats.append("Low Alert")
    df["risk_category"] = risk_cats

    # Clean missing values for all numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median() if not np.isnan(df[c].median()) else 0)

    summary_meta = {
        "total_rows": len(df),
        "columns_detected": list(df.columns),
        "peak_risk_score": float(df["risk_score"].max()) if not df.empty else 0.0,
        "avg_risk_score": float(round(df["risk_score"].mean(), 1)) if not df.empty else 0.0,
        "critical_events_count": int((df["risk_score"] >= 75).sum()),
        "total_casualties": int(df["casualties"].sum()),
        "total_economic_loss_m": float(round(df["economic_loss_millions"].sum(), 2))
    }

    return df, summary_meta


def calculate_earthquake_risk_score(
    magnitude: float,
    depth_km: float,
    population_density: float = 500.0,
    vulnerability_index: float = 0.5
) -> float:
    """
    Calculate composite Earthquake Risk Score (0 - 100).
    Formula incorporates Richter magnitude energy, focal depth penalty (shallow = higher damage),
    and local exposure (population density & vulnerability).
    """
    mag = max(0.0, float(magnitude))
    depth = max(1.0, float(depth_km))
    pop_dens = max(10.0, float(population_density))
    vuln = max(0.1, min(1.0, float(vulnerability_index)))

    # Magnitude component (Scale 0-80 for magnitude <= 9.0)
    mag_score = min(80.0, (mag / 9.0) ** 2.2 * 80.0)

    # Focal depth factor (shallow depth < 50 km drastically increases surface shaking)
    depth_factor = np.exp(-depth / 50.0) * 15.0

    # Population exposure factor (log scale up to 15 points)
    pop_factor = min(15.0, (np.log10(pop_dens + 1.0) / 4.0) * 15.0 * vuln)

    composite_score = mag_score + depth_factor + pop_factor
    return float(round(min(100.0, max(0.0, composite_score)), 1))


def calculate_flood_risk_score(
    rainfall_mm: float,
    river_level_m: float,
    threshold_m: float = 5.0,
    population_density: float = 500.0,
    vulnerability_index: float = 0.5
) -> float:
    """
    Calculate composite Flood Risk Score (0 - 100).
    Evaluates 24h rainfall, river stage ratio, and regional exposure.
    """
    rain = max(0.0, float(rainfall_mm))
    river = max(0.0, float(river_level_m))
    thresh = max(1.0, float(threshold_m))
    pop_dens = max(10.0, float(population_density))
    vuln = max(0.1, min(1.0, float(vulnerability_index)))

    # Rainfall score (100mm = 25 points, 300mm = 60 points)
    rain_score = min(50.0, (rain / 300.0) * 50.0)

    # River overflow ratio (stage / threshold)
    stage_ratio = river / thresh
    river_score = min(50.0, max(0.0, (stage_ratio - 0.7) * 50.0))

    # Exposure multiplier
    pop_factor = min(1.3, np.log10(pop_dens + 1.0) / 3.0)

    total_score = (rain_score + river_score) * (0.8 + 0.4 * pop_factor * vuln)
    return float(round(min(100.0, max(0.0, total_score)), 1))


def calculate_cyclone_risk_score(
    wind_speed_kmh: float,
    pressure_hpa: float = 1013.0,
    distance_to_coast_km: float = 10.0,
    population_density: float = 500.0
) -> float:
    """
    Calculate composite Cyclone Risk Score (0 - 100).
    Based on Saffir-Simpson wind scale logic, pressure deficit, and coastal proximity.
    """
    wind = max(0.0, float(wind_speed_kmh))
    pressure = float(pressure_hpa)
    dist = max(1.0, float(distance_to_coast_km))
    pop_dens = max(10.0, float(population_density))

    # Wind speed component (260 km/h max baseline)
    wind_score = min(70.0, (wind / 260.0) * 70.0)

    # Pressure drop score (Normal sea level pressure ~1013 hPa)
    pressure_drop = max(0.0, 1013.0 - pressure)
    pressure_score = min(30.0, (pressure_drop / 90.0) * 30.0)

    # Coastal decay (risk drops further inland)
    coastal_factor = np.exp(-dist / 150.0)

    pop_factor = min(1.2, np.log10(pop_dens + 1.0) / 3.0)

    score = (wind_score + pressure_score) * (0.7 + 0.3 * coastal_factor) * pop_factor
    return float(round(min(100.0, max(0.0, score)), 1))


def extract_ml_features(
    df: pd.DataFrame,
    disaster_type: str = "earthquake"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Feature engineering pipeline for machine learning prediction models.
    """
    df = clean_disaster_data(df)
    
    if df.empty:
        return pd.DataFrame(), pd.Series()

    if "timestamp" in df.columns:
        df["month"] = df["timestamp"].dt.month
        df["day_of_year"] = df["timestamp"].dt.dayofyear
        df["year"] = df["timestamp"].dt.year
    else:
        df["month"] = 6
        df["day_of_year"] = 180
        df["year"] = 2024

    if disaster_type.lower() == "earthquake":
        features = ["magnitude", "depth", "latitude", "longitude", "month", "day_of_year"]
        
        df["mag_depth_ratio"] = df["magnitude"] / (df["depth"] + 1.0)
        df["shallow_flag"] = (df["depth"] < 30.0).astype(int)
        df["high_mag_flag"] = (df["magnitude"] >= 5.5).astype(int)
        
        feature_cols = features + ["mag_depth_ratio", "shallow_flag", "high_mag_flag"]
        target_col = "risk_score"
        
    elif disaster_type.lower() == "flood":
        df["rain_river_ratio"] = df["rainfall_mm"] / (df["river_level_m"] + 0.1)
        df["over_threshold"] = (df["river_level_m"] > df["threshold_m"]).astype(int)
        
        feature_cols = ["rainfall_mm", "river_level_m", "threshold_m", "latitude", "longitude",
                        "month", "day_of_year", "rain_river_ratio", "over_threshold"]
        target_col = "risk_score"
        
    else:  # Cyclone
        df["pressure_drop"] = 1013.0 - df["pressure_hpa"]
        df["wind_pressure_ratio"] = df["wind_speed_kmh"] / (df["pressure_hpa"] + 1.0)
        
        feature_cols = ["wind_speed_kmh", "pressure_hpa", "latitude", "longitude",
                        "month", "pressure_drop", "wind_pressure_ratio"]
        target_col = "risk_score"

    X = df[feature_cols].copy()
    y = df[target_col].copy()
    return X, y


def aggregate_disaster_summary(
    eq_df: pd.DataFrame,
    flood_df: pd.DataFrame,
    cyc_df: pd.DataFrame
) -> Dict[str, Any]:
    """
    Generate overall summary metrics for the executive dashboard KPI cards.
    """
    total_events = len(eq_df) + len(flood_df) + len(cyc_df)
    
    critical_eq = len(eq_df[eq_df["risk_score"] >= 75]) if not eq_df.empty else 0
    critical_fl = len(flood_df[flood_df["risk_score"] >= 75]) if not flood_df.empty else 0
    critical_cy = len(cyc_df[cyc_df["risk_score"] >= 75]) if not cyc_df.empty else 0
    total_critical = critical_eq + critical_fl + critical_cy
    
    total_casualties = int(eq_df["casualties"].sum() if not eq_df.empty and "casualties" in eq_df.columns else 0)
    total_affected = int(flood_df["affected_population"].sum() if not flood_df.empty and "affected_population" in flood_df.columns else 0)
    total_loss_m = float(eq_df["economic_loss_millions"].sum() if not eq_df.empty and "economic_loss_millions" in eq_df.columns else 0.0)

    max_risk_eq = float(eq_df["risk_score"].max()) if not eq_df.empty else 0.0
    max_risk_fl = float(flood_df["risk_score"].max()) if not flood_df.empty else 0.0
    max_risk_cy = float(cyc_df["risk_score"].max()) if not cyc_df.empty else 0.0
    max_risk = max(max_risk_eq, max_risk_fl, max_risk_cy)

    return {
        "total_disaster_events": total_events,
        "critical_warnings_count": total_critical,
        "total_casualties": total_casualties,
        "total_affected_population": total_affected,
        "total_economic_loss_m": round(total_loss_m, 2),
        "peak_hazard_risk_score": max_risk,
        "earthquake_count": len(eq_df),
        "flood_count": len(flood_df),
        "cyclone_count": len(cyc_df)
    }
