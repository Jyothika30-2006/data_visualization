"""
Module: disaster_detector.py
Handles automatic detection of disaster category (Earthquake, Flood, Cyclone, Tsunami, or Multi-disaster)
from uploaded dataset columns and value semantics, plus flexible column mapping.
"""

import re
import pandas as pd
import numpy as np
from typing import Dict, Tuple, List, Any


KEYWORD_MAPPINGS = {
    "timestamp": ["timestamp", "date", "datetime", "time", "event_date", "observation_date", "landfall_date", "year", "created_at"],
    "location": ["location", "location_name", "region", "district", "city", "place", "site", "coastal_location", "affected_location", "address", "area"],
    "latitude": ["latitude", "lat", "lat_deg", "y"],
    "longitude": ["longitude", "lon", "lng", "lon_deg", "x"],
    "magnitude": ["magnitude", "richter_magnitude", "mag", "trigger_earthquake_mag", "eq_mag"],
    "depth": ["focal_depth_km", "depth_km", "depth", "focal_depth"],
    "rainfall": ["24h_rainfall_mm", "rainfall_mm", "rain_mm", "precipitation", "rainfall"],
    "river_level": ["river_gauge_level_m", "river_level_m", "river_level", "gauge_height"],
    "threshold": ["flood_threshold_m", "threshold_m", "threshold"],
    "wind_speed": ["wind_speed_kmh", "wind_speed", "wind_kmh", "max_sustained_wind"],
    "pressure": ["central_pressure_hpa", "pressure_hpa", "pressure", "min_pressure"],
    "wave_height": ["max_wave_height_m", "wave_height_m", "wave_height", "tsunami_height"],
    "category": ["disaster_category", "disaster_type", "storm_category", "category", "severity_class"],
    "casualties": ["casualties", "casualties_direct", "deaths", "fatalities", "injured", "affected_population"],
    "economic_loss": ["economic_loss_usd_millions", "economic_loss_millions", "economic_damage_millions", "loss", "damage_usd", "cost_m"]
}


def auto_map_columns(df: pd.DataFrame) -> Dict[str, str]:
    """
    Flexible column mapper. Scans uploaded dataframe headers and matches canonical keys.
    Returns dict: canonical_name -> original_df_column_header
    """
    mapped = {}
    cols_clean = {c: re.sub(r'[^a-z0-9_]', '', str(c).lower().strip()) for c in df.columns}

    for canonical, keywords in KEYWORD_MAPPINGS.items():
        for orig_col, clean_col in cols_clean.items():
            if orig_col in mapped.values():
                continue
            for kw in keywords:
                clean_kw = re.sub(r'[^a-z0-9_]', '', kw.lower())
                if clean_kw == clean_col or clean_kw in clean_col:
                    mapped[canonical] = orig_col
                    break
            if canonical in mapped:
                break

    return mapped


def auto_detect_disaster_type(df: pd.DataFrame, mapped_cols: Dict[str, str]) -> Tuple[str, float]:
    """
    Auto-detect disaster category from mapped columns and data content.
    Categories: 'Earthquake', 'Flood', 'Cyclone', 'Tsunami', 'Multi-disaster'
    Returns: (detected_disaster_type, confidence_score)
    """
    # 1. Check if explicit category / type column exists with multiple values
    if "category" in mapped_cols:
        cat_col = mapped_cols["category"]
        unique_vals = [str(v).lower() for v in df[cat_col].dropna().unique()]
        disaster_keywords_found = set()
        for v in unique_vals:
            if "earthquake" in v or "eq" in v: disaster_keywords_found.add("Earthquake")
            if "flood" in v or "rain" in v: disaster_keywords_found.add("Flood")
            if "cyclone" in v or "storm" in v: disaster_keywords_found.add("Cyclone")
            if "tsunami" in v or "wave" in v: disaster_keywords_found.add("Tsunami")
            if "landslide" in v: disaster_keywords_found.add("Landslide")

        if len(disaster_keywords_found) >= 2:
            return "Multi-disaster", 0.95
        elif len(disaster_keywords_found) == 1:
            return list(disaster_keywords_found)[0], 0.98

    # 2. Signature column presence heuristics
    scores = {
        "Earthquake": 0.0,
        "Flood": 0.0,
        "Cyclone": 0.0,
        "Tsunami": 0.0
    }

    if "magnitude" in mapped_cols: scores["Earthquake"] += 45.0
    if "depth" in mapped_cols: scores["Earthquake"] += 35.0
    
    if "rainfall" in mapped_cols: scores["Flood"] += 45.0
    if "river_level" in mapped_cols: scores["Flood"] += 35.0
    if "threshold" in mapped_cols: scores["Flood"] += 15.0

    if "wind_speed" in mapped_cols: scores["Cyclone"] += 45.0
    if "pressure" in mapped_cols: scores["Cyclone"] += 35.0

    if "wave_height" in mapped_cols: scores["Tsunami"] += 50.0

    # 3. Text search across text columns
    text_sample = " ".join(df.select_dtypes(include=['object']).head(50).fillna('').values.flatten()).lower()
    if "earthquake" in text_sample or "richter" in text_sample: scores["Earthquake"] += 20.0
    if "flood" in text_sample or "inundation" in text_sample: scores["Flood"] += 20.0
    if "cyclone" in text_sample or "typhoon" in text_sample or "hurricane" in text_sample: scores["Cyclone"] += 20.0
    if "tsunami" in text_sample or "wave height" in text_sample: scores["Tsunami"] += 20.0

    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]

    if best_score < 30.0:
        return "Multi-disaster", 0.60

    confidence = min(0.99, round(best_score / 100.0, 2))
    return best_type, max(0.65, confidence)
