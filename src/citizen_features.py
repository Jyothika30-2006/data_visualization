"""
Module 7: Citizen Features
Implements citizen reporting, verification workflow, nearest emergency shelter finder
using location names / PIN codes / cities, emergency contacts directory, and interactive safety checklists.
"""

import math
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging

from src.database import execute_query, query_to_df

logger = logging.getLogger(__name__)

# Comprehensive Location Name & City Coordinate Directory
LOCATION_DIRECTORY = {
    "mumbai": (19.0760, 72.8777),
    "mumbai city": (19.0760, 72.8777),
    "kurla": (19.0657, 72.8797),
    "bandra": (19.0596, 72.8295),
    "thane": (19.2183, 72.9781),
    "chennai": (13.0827, 80.2707),
    "adyar": (13.0012, 80.2565),
    "marina": (13.0475, 80.2824),
    "kolkata": (22.5726, 88.4149),
    "salt lake": (22.5726, 88.4149),
    "howrah": (22.5851, 88.3413),
    "guwahati": (26.1445, 91.7362),
    "assam": (26.1445, 91.7362),
    "silchar": (24.8333, 92.7789),
    "kochi": (9.9312, 76.2673),
    "kerala": (9.9312, 76.2673),
    "alappuzha": (9.4981, 76.3388),
    "wayanad": (11.6854, 76.1320),
    "puri": (19.8135, 85.8312),
    "odisha": (20.2961, 85.8245),
    "paradip": (20.3165, 86.6114),
    "bhubaneswar": (20.2961, 85.8245),
    "patna": (25.5941, 85.1376),
    "bihar": (25.5941, 85.1376),
    "ahmedabad": (23.0225, 72.5714),
    "gujarat": (23.0225, 72.5714),
    "kutch": (23.2420, 69.6669),
    "bhuj": (23.2420, 69.6669),
    "dehradun": (30.3165, 78.0322),
    "uttarakhand": (30.3165, 78.0322),
    "shimla": (31.1048, 77.1734),
    "himachal": (31.1048, 77.1734),
    "srinagar": (34.0837, 74.7973),
    "jammu": (32.7266, 74.8570),
    "visakhapatnam": (17.6868, 83.2185),
    "vijayawada": (16.5062, 80.6480),
    "andhra": (16.5062, 80.6480),
    "port blair": (11.6234, 92.7265),
    "andaman": (11.6234, 92.7265),
    "delhi": (28.6139, 77.2090),
    "bangalore": (12.9716, 77.5946),
    "bengaluru": (12.9716, 77.5946),
    "hyderabad": (17.3850, 78.4867),
    "pune": (18.5204, 73.8567),
    "japan": (36.2048, 138.2529),
    "california": (36.7783, -119.4179),
    "nepal": (28.3949, 84.1240),
    "turkey": (38.9637, 35.2433)
}


def geocode_location_name(location_query: str) -> Tuple[float, float, str]:
    """
    Resolve a city name, landmark, region, or PIN code into latitude/longitude coordinates.
    Returns: (latitude, longitude, matched_canonical_name)
    """
    if not location_query:
        return 20.5937, 78.9629, "India Central"

    query_clean = location_query.strip().lower()

    # Direct matching
    for key, coords in LOCATION_DIRECTORY.items():
        if key in query_clean or query_clean in key:
            return coords[0], coords[1], key.title()

    # Try geopy if installed
    try:
        from geopy.geocoders import Nominatim
        geolocator = Nominatim(user_agent="disaster_intelligence_app")
        loc = geolocator.geocode(location_query, timeout=3)
        if loc:
            return float(loc.latitude), float(loc.longitude), loc.address.split(",")[0]
    except Exception as e:
        logger.debug(f"Geopy lookup fallback: {e}")

    # Default fallback
    return 20.5937, 78.9629, location_query.title()


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points on the earth in kilometers.
    """
    R = 6371.0  # Earth radius in kilometers

    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) ** 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return round(R * c, 2)


def find_nearest_shelters(location_query_or_lat: Any, lon: Optional[float] = None, top_n: int = 5) -> Tuple[pd.DataFrame, str]:
    """
    Find nearest available shelters using location name OR coordinates.
    Returns: (dataframe, resolved_location_name)
    """
    if isinstance(location_query_or_lat, str):
        user_lat, user_lon, resolved_name = geocode_location_name(location_query_or_lat)
    else:
        user_lat = float(location_query_or_lat)
        user_lon = float(lon) if lon is not None else 78.9629
        resolved_name = f"{user_lat:.2f}, {user_lon:.2f}"

    shelters_df = query_to_df("SELECT * FROM shelters WHERE status = 'Open' OR status = 'Active / Open'")
    if shelters_df.empty:
        shelters_df = query_to_df("SELECT * FROM shelters")

    if shelters_df.empty:
        return pd.DataFrame(), resolved_name

    shelters_df = shelters_df.copy()
    distances = []
    available_capacities = []

    for _, row in shelters_df.iterrows():
        dist = haversine_distance(user_lat, user_lon, row["latitude"], row["longitude"])
        avail = row["capacity"] - row["current_occupancy"]
        distances.append(dist)
        available_capacities.append(avail)

    shelters_df["distance_km"] = distances
    shelters_df["available_space"] = available_capacities

    sorted_df = shelters_df.sort_values(by="distance_km").head(top_n)
    return sorted_df, resolved_name


def submit_citizen_report(
    reporter_name: str,
    phone: str,
    disaster_type: str,
    location_name: str,
    latitude: Optional[float] = None,
    longitude: Optional[float] = None,
    description: str = "",
    severity: str = "Moderate"
) -> int:
    """
    Submit a crowdsourced disaster incident report. Geocodes location name automatically if coordinates are omitted.
    """
    if latitude is None or longitude is None or (latitude == 0.0 and longitude == 0.0):
        lat, lon, _ = geocode_location_name(location_name)
    else:
        lat, lon = latitude, longitude

    query = """
    INSERT INTO citizen_reports 
    (reporter_name, phone, disaster_type, location_name, latitude, longitude, description, severity, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'Pending Verification')
    """
    return execute_query(query, (
        reporter_name, phone, disaster_type, location_name,
        lat, lon, description, severity
    ))


def verify_citizen_report(report_id: int, verifier_name: str, status: str = "Verified") -> bool:
    """
    Control room admin verification function to approve or reject crowdsourced reports.
    """
    query = "UPDATE citizen_reports SET status = ?, verified_by = ? WHERE id = ?"
    execute_query(query, (status, verifier_name, report_id))
    return True


def get_emergency_contacts_directory() -> pd.DataFrame:
    """
    Emergency Helplines & Response Agency Contact Directory.
    """
    data = [
        {"Agency / Authority": "National Disaster Response Force (NDRF)", "Helpline Number": "1078 / 011-24363260", "Role": "Specialized Search & Rescue Operations"},
        {"Agency / Authority": "National Emergency Response System (Single Hotline)", "Helpline Number": "112", "Role": "Unified Police, Fire & Medical Dispatch"},
        {"Agency / Authority": "State Disaster Management Authority (SDMA)", "Helpline Number": "1070", "Role": "State-level Disaster Coordination"},
        {"Agency / Authority": "District Control Room Hotline", "Helpline Number": "1077", "Role": "District Evacuation & Shelter Logistics"},
        {"Agency / Authority": "Indian Meteorological Department (IMD)", "Helpline Number": "1800-180-1717", "Role": "Severe Weather & Cyclone Bulletins"},
        {"Agency / Authority": "Indian Coast Guard Rescue HQ", "Helpline Number": "1554", "Role": "Maritime & Coastal Flood Rescues"},
        {"Agency / Authority": "Ambulance Emergency Medical Services", "Helpline Number": "108 / 102", "Role": "Critical Casualty & Trauma Response"}
    ]
    return pd.DataFrame(data)


def get_safety_checklists() -> Dict[str, List[str]]:
    """
    Disaster Safety Checklists for Citizens.
    """
    return {
        "Earthquake Preparedness": [
            "✅ Create an Emergency Go-Bag (Water, canned food, flashlight, first-aid, radio).",
            "✅ Identify safe indoor spots: under heavy tables or against interior walls.",
            "✅ Secure heavy furniture, appliances, and wall hangings to studs.",
            "✅ Know how to shut off main gas, water, and electrical supply.",
            "✅ Drop, Cover, and Hold On during active shaking."
        ],
        "Flood Preparedness": [
            "✅ Monitor local river gauges and meteorological rainfall alerts.",
            "✅ Move vital documents and electrical appliances to upper floors.",
            "✅ Never drive through submerged bridges or waterlogged roads.",
            "✅ Keep bottled water supply (at least 3 liters per person per day).",
            "✅ Identify nearest high-ground emergency shelter route."
        ],
        "Cyclone Preparedness": [
            "✅ Board up or tape windows and secure outdoor items/roof sheets.",
            "✅ Keep battery-powered emergency radios and fully charged power banks.",
            "✅ Store 7-day non-perishable food ration and essential medications.",
            "✅ Remain indoors during the eye of the storm (winds can resume suddenly).",
            "✅ Disconnect non-essential electrical appliances before landfall."
        ]
    }
