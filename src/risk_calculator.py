"""
Module: risk_calculator.py
Calculates composite hazard risk scores (0-100), severity classifications,
and location hazard level rankings for disaster analytics.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any


RISK_CATEGORIES = {
    "LOW": {"min": 0, "max": 30, "color": "#10B981", "label": "Low Alert"},
    "MODERATE": {"min": 31, "max": 60, "color": "#F59E0B", "label": "Moderate Risk"},
    "HIGH": {"min": 61, "max": 80, "color": "#F97316", "label": "High Warning"},
    "CRITICAL": {"min": 81, "max": 100, "color": "#EF4444", "label": "Critical Hazard"}
}


def classify_risk_level(score: float) -> Tuple[str, str]:
    """Classify 0-100 risk score into category label and hex color."""
    val = float(max(0, min(100, score)))
    if val >= 81:
        return RISK_CATEGORIES["CRITICAL"]["label"], RISK_CATEGORIES["CRITICAL"]["color"]
    elif val >= 61:
        return RISK_CATEGORIES["HIGH"]["label"], RISK_CATEGORIES["HIGH"]["color"]
    elif val >= 31:
        return RISK_CATEGORIES["MODERATE"]["label"], RISK_CATEGORIES["MODERATE"]["color"]
    else:
        return RISK_CATEGORIES["LOW"]["label"], RISK_CATEGORIES["LOW"]["color"]


def calculate_top_hotspots(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Identify top N high-risk location hotspots sorted by average risk score and casualty impact."""
    if df.empty or "location_name" not in df.columns:
        return pd.DataFrame()

    grouped = df.groupby("location_name").agg(
        total_incidents=("risk_score", "count"),
        avg_risk_score=("risk_score", "mean"),
        peak_risk_score=("risk_score", "max"),
        total_casualties=("casualties", "sum"),
        total_economic_loss_m=("economic_loss_millions", "sum")
    ).reset_index()

    grouped["avg_risk_score"] = grouped["avg_risk_score"].round(1)
    grouped["total_economic_loss_m"] = grouped["total_economic_loss_m"].round(2)

    # Composite hotspot ranking
    grouped["hotspot_rank_score"] = (grouped["avg_risk_score"] * 0.6) + (np.log10(grouped["total_casualties"] + 1) * 15.0)
    sorted_hotspots = grouped.sort_values(by="hotspot_rank_score", ascending=False).head(top_n)

    return sorted_hotspots.drop(columns=["hotspot_rank_score"])
