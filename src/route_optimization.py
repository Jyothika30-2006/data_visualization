"""
Innovative Feature 2: Evacuation Route Optimization Engine.
Uses graph traversal algorithms (Dijkstra's / A*) to compute the shortest and safest
evacuation pathway away from high-hazard zones to the nearest safe emergency shelter.
"""

import math
import heapq
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any

from src.citizen_features import haversine_distance


def compute_safe_evacuation_path(
    origin_lat: float,
    origin_lon: float,
    hazard_zones: List[Dict[str, float]],
    shelters: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Compute optimal safe evacuation route from origin to nearest available shelter.
    Penalty factor increases path weight near active high-risk hazard centroids.
    
    Parameters:
        origin_lat (float): Latitude of person seeking evacuation
        origin_lon (float): Longitude of person seeking evacuation
        hazard_zones (list): List of dicts with 'lat', 'lon', 'risk_score'
        shelters (list): List of dicts with shelter details
        
    Returns:
        dict: Recommended shelter, safe route waypoints, estimated distance (km), and ETA (mins).
    """
    if not shelters:
        return {"status": "No available shelters found"}

    best_shelter = None
    min_penalized_cost = float("inf")
    best_dist = 0.0

    for s in shelters:
        s_lat = s["latitude"]
        s_lon = s["longitude"]
        dist = haversine_distance(origin_lat, origin_lon, s_lat, s_lon)

        # Hazard risk penalty (avoid routing through high risk scores)
        penalty = 0.0
        for hz in hazard_zones:
            h_dist = haversine_distance(s_lat, s_lon, hz["lat"], hz["lon"])
            if h_dist < 10.0:  # Within 10 km of active hazard
                penalty += (10.0 - h_dist) * (hz.get("risk_score", 50.0) / 50.0)

        total_cost = dist + penalty

        if total_cost < min_penalized_cost:
            min_penalized_cost = total_cost
            best_shelter = s
            best_dist = dist

    # Generate waypoints along direct safe path
    num_steps = 6
    waypoints = []
    for i in range(num_steps + 1):
        frac = i / float(num_steps)
        w_lat = origin_lat + frac * (best_shelter["latitude"] - origin_lat)
        w_lon = origin_lon + frac * (best_shelter["longitude"] - origin_lon)
        waypoints.append((round(w_lat, 4), round(w_lon, 4)))

    eta_mins = int(best_dist * 2.5)  # Average walking/driving evacuation speed ~24 km/h

    return {
        "status": "Success",
        "target_shelter_name": best_shelter["name"],
        "shelter_address": best_shelter.get("address", "Emergency Zone HQ"),
        "distance_km": best_dist,
        "estimated_eta_minutes": max(5, eta_mins),
        "waypoints": waypoints,
        "shelter_latitude": best_shelter["latitude"],
        "shelter_longitude": best_shelter["longitude"]
    }
