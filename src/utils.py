"""
Module: utils.py
Provides report generation, CSV/JSON export helpers, emergency contacts,
and citizen safety guidelines directory.
"""

import json
import pandas as pd
from typing import Dict, List, Any


def generate_json_summary_report(df: pd.DataFrame, meta_summary: Dict[str, Any]) -> str:
    """Generate structured JSON summary report string for downloadable reports."""
    report = {
        "report_title": "Natural Disaster Intelligence & Response Executive Summary",
        "generated_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S UTC"),
        "analytics_metadata": meta_summary,
        "top_affected_locations": df.groupby("location_name")["risk_score"].mean().head(5).to_dict() if "location_name" in df.columns else {}
    }
    return json.dumps(report, indent=2)


def get_emergency_contacts_directory() -> pd.DataFrame:
    """Emergency Helplines & Rescue Agency Contact Directory."""
    data = [
        {"Agency / Authority": "National Disaster Response Force (NDRF)", "Helpline": "1078 / 011-24363260", "Role": "Specialized Search & Rescue Operations"},
        {"Agency / Authority": "National Emergency Hotline (Unified)", "Helpline": "112", "Role": "Police, Fire & Medical Unified Dispatch"},
        {"Agency / Authority": "State Disaster Management Authority (SDMA)", "Helpline": "1070", "Role": "State-level Emergency Relief Coordination"},
        {"Agency / Authority": "District Disaster Control Room", "Helpline": "1077", "Role": "District Shelters & Evacuation Logistics"},
        {"Agency / Authority": "Indian Meteorological Department (IMD)", "Helpline": "1800-180-1717", "Role": "Weather & Cyclone Warnings"},
        {"Agency / Authority": "Indian Coast Guard HQ", "Helpline": "1554", "Role": "Maritime Coastal & Maritime Flood Rescues"},
        {"Agency / Authority": "Emergency Medical Ambulance", "Helpline": "108 / 102", "Role": "Trauma Casualty Medical Dispatch"}
    ]
    return pd.DataFrame(data)


def get_safety_guidelines() -> Dict[str, List[str]]:
    """Citizen Disaster Safety Guidelines Directory."""
    return {
        "Earthquake Safety Guidelines": [
            "✅ Drop, Cover, and Hold On under heavy furniture or against interior walls.",
            "✅ Move away from glass, windows, unanchored heavy furniture, and power lines.",
            "✅ Keep an Emergency Survival Kit handy (water, flashlight, first-aid kit, radio).",
            "✅ Know how to shut off main gas, electricity, and water valves.",
            "✅ Expect aftershocks following major seismic shaking."
        ],
        "Flood Safety Guidelines": [
            "✅ Move immediately to higher ground when flood warnings are issued.",
            "✅ Never drive or walk through flowing floodwaters (6 inches of water can knock you down).",
            "✅ Disconnect electrical appliances and main switches before floodwaters enter.",
            "✅ Store bottled drinking water (at least 3 liters per person per day).",
            "✅ Avoid contact with waterlogged roads due to electrical hazard or open manholes."
        ],
        "Cyclone Safety Guidelines": [
            "✅ Board up or tape windows and secure loose outdoor objects and metal roof sheets.",
            "✅ Keep fully charged power banks, emergency radios, and battery lanterns.",
            "✅ Store 7-day non-perishable food rations and required prescription medicines.",
            "✅ Stay indoors throughout the storm, even during temporary lulls in the eye.",
            "✅ Follow official evacuation orders issued by local authorities immediately."
        ],
        "Tsunami Safety Guidelines": [
            "✅ If you feel strong coastal shaking or see ocean water receding, run inland to high ground.",
            "✅ Move at least 2 miles inland or 100 feet above sea level immediately.",
            "✅ Never stay near the shore to watch a tsunami wave.",
            "✅ Wait for official clear announcements from Coast Guard / SDMA before returning."
        ]
    }
