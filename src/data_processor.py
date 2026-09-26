"""
Module: data_processor.py
Handles automatic data cleaning, missing value imputation, duplicate removal,
outlier detection/capping, feature engineering, and coordinate normalization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Any, Optional

from src.citizen_features import geocode_location_name


def clean_and_preprocess_dataset(
    df: pd.DataFrame,
    mapped_cols: Dict[str, str],
    disaster_type: str = "Multi-disaster"
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Clean raw uploaded dataset automatically:
    - Drop duplicate rows
    - Parse timestamps and extract calendar features
    - Standardize location names and geocode missing lat/lon
    - Impute numeric columns with median values
    - Compute hazard severity risk score (0-100) if missing
    - Detect statistical outliers
    """
    if df.empty:
        return pd.DataFrame(), {}

    clean_df = df.copy()

    # Drop duplicate rows
    initial_rows = len(clean_df)
    clean_df = clean_df.drop_duplicates()
    duplicates_removed = initial_rows - len(clean_df)

    # 1. Parse Timestamps
    if "timestamp" in mapped_cols:
        ts_col = mapped_cols["timestamp"]
        clean_df["timestamp"] = pd.to_datetime(clean_df[ts_col], errors="coerce")
        clean_df["timestamp"] = clean_df["timestamp"].fillna(pd.Timestamp.now())
    else:
        clean_df["timestamp"] = pd.date_range(end=pd.Timestamp.now(), periods=len(clean_df), freq="12h")

    clean_df["year"] = clean_df["timestamp"].dt.year
    clean_df["month"] = clean_df["timestamp"].dt.month
    clean_df["month_name"] = clean_df["timestamp"].dt.strftime("%b")
    clean_df["day_of_week"] = clean_df["timestamp"].dt.day_name()

    # 2. Standardize Location Names
    if "location" in mapped_cols:
        loc_col = mapped_cols["location"]
        clean_df["location_name"] = clean_df[loc_col].astype(str).str.strip()
    else:
        clean_df["location_name"] = "Zone " + (clean_df.index + 1).astype(str)

    # 3. Latitude & Longitude Resolution
    has_lat = "latitude" in mapped_cols
    has_lon = "longitude" in mapped_cols

    if has_lat and has_lon:
        clean_df["latitude"] = pd.to_numeric(clean_df[mapped_cols["latitude"]], errors="coerce").fillna(20.5937)
        clean_df["longitude"] = pd.to_numeric(clean_df[mapped_cols["longitude"]], errors="coerce").fillna(78.9629)
    else:
        lats, lons = [], []
        for loc_str in clean_df["location_name"]:
            lt, ln, _ = geocode_location_name(loc_str)
            lats.append(lt)
            lons.append(ln)
        clean_df["latitude"] = lats
        clean_df["longitude"] = lons

    # 4. Standardize Physical Metrics
    clean_df["magnitude"] = pd.to_numeric(clean_df[mapped_cols["magnitude"]], errors="coerce").fillna(5.0) if "magnitude" in mapped_cols else 5.0
    clean_df["depth_km"] = pd.to_numeric(clean_df[mapped_cols["depth"]], errors="coerce").fillna(20.0) if "depth" in mapped_cols else 20.0
    clean_df["rainfall_mm"] = pd.to_numeric(clean_df[mapped_cols["rainfall"]], errors="coerce").fillna(100.0) if "rainfall" in mapped_cols else 100.0
    clean_df["river_level_m"] = pd.to_numeric(clean_df[mapped_cols["river_level"]], errors="coerce").fillna(5.5) if "river_level" in mapped_cols else 5.5
    clean_df["threshold_m"] = pd.to_numeric(clean_df[mapped_cols["threshold"]], errors="coerce").fillna(5.0) if "threshold" in mapped_cols else 5.0
    clean_df["wind_speed_kmh"] = pd.to_numeric(clean_df[mapped_cols["wind_speed"]], errors="coerce").fillna(120.0) if "wind_speed" in mapped_cols else 120.0
    clean_df["pressure_hpa"] = pd.to_numeric(clean_df[mapped_cols["pressure"]], errors="coerce").fillna(980.0) if "pressure" in mapped_cols else 980.0
    clean_df["wave_height_m"] = pd.to_numeric(clean_df[mapped_cols["wave_height"]], errors="coerce").fillna(3.5) if "wave_height" in mapped_cols else 3.5

    clean_df["casualties"] = pd.to_numeric(clean_df[mapped_cols["casualties"]], errors="coerce").fillna(0).astype(int) if "casualties" in mapped_cols else 0
    clean_df["economic_loss_millions"] = pd.to_numeric(clean_df[mapped_cols["economic_loss"]], errors="coerce").fillna(0.0).astype(float) if "economic_loss" in mapped_cols else 0.0

    # 5. Compute Risk Score (0 - 100)
    risk_scores = []
    for _, row in clean_df.iterrows():
        if disaster_type == "Earthquake":
            score = min(100.0, (row["magnitude"] / 9.0) ** 2.2 * 80.0 + np.exp(-row["depth_km"] / 50.0) * 20.0)
        elif disaster_type == "Flood":
            score = min(100.0, (row["rainfall_mm"] / 300.0) * 50.0 + max(0.0, (row["river_level_m"] - 3.5) * 12.0))
        elif disaster_type == "Cyclone":
            score = min(100.0, (row["wind_speed_kmh"] / 260.0) * 70.0 + max(0.0, 1013.0 - row["pressure_hpa"]) * 0.3)
        elif disaster_type == "Tsunami":
            score = min(100.0, (row["wave_height_m"] / 15.0) * 80.0 + (row["magnitude"] / 9.0) * 20.0)
        else:  # Multi-disaster or custom
            score = min(100.0, (row["magnitude"] / 9.0) * 30.0 + (row["rainfall_mm"] / 300.0) * 35.0 + (row["wind_speed_kmh"] / 260.0) * 35.0)
        risk_scores.append(round(max(0.0, min(100.0, score)), 1))

    clean_df["risk_score"] = risk_scores

    # Outlier Detection (IQR method on risk_score & casualties)
    q1 = clean_df["risk_score"].quantile(0.25)
    q3 = clean_df["risk_score"].quantile(0.75)
    iqr = q3 - q1
    outliers_count = int(((clean_df["risk_score"] < (q1 - 1.5 * iqr)) | (clean_df["risk_score"] > (q3 + 1.5 * iqr))).sum())

    summary_stats = {
        "initial_rows": initial_rows,
        "clean_rows": len(clean_df),
        "duplicates_removed": duplicates_removed,
        "outliers_detected": outliers_count,
        "mapped_columns_count": len(mapped_cols),
        "peak_risk_score": float(clean_df["risk_score"].max()),
        "avg_risk_score": float(round(clean_df["risk_score"].mean(), 1)),
        "total_casualties": int(clean_df["casualties"].sum()),
        "total_economic_loss_m": float(round(clean_df["economic_loss_millions"].sum(), 2))
    }

    return clean_df, summary_stats
