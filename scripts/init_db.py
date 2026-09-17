"""
Database Initialization & Sample Data Generator Script.
Populates the SQLite database with:
- 1000+ historical earthquake records
- 500+ flood incidents
- 200+ cyclone events
- 50+ shelter locations with real capacities & resources
- 50+ regional population density & vulnerability profiles
- Initial citizen reports, alert subscribers, and resource allocations
"""

import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Add parent dir to sys.path so config and src can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.database import init_db, insert_df, get_connection, query_to_df
from config import DB_PATH

def generate_sample_data():
    print("Initializing Database Schema...")
    init_db(DB_PATH)

    conn = get_connection(DB_PATH)
    cursor = conn.cursor()

    # Check if data already populated
    cursor.execute("SELECT COUNT(*) FROM earthquakes")
    eq_count = cursor.fetchone()[0]
    conn.close()

    if eq_count >= 1000:
        print(f"Database already populated ({eq_count} earthquake records found). Skipping generation.")
        return

    np.random.seed(42)
    start_date = datetime(2018, 1, 1)

    # -------------------------------------------------------------
    # 1. GENERATE 1050 HISTORICAL EARTHQUAKE RECORDS
    # -------------------------------------------------------------
    print("Generating 1,050 historical earthquake records...")
    regions = [
        {"name": "Himalayan Belt, India", "lat": 30.5, "lon": 79.0},
        {"name": "Gujarat / Kutch", "lat": 23.5, "lon": 70.0},
        {"name": "Andaman & Nicobar Islands", "lat": 11.7, "lon": 92.7},
        {"name": "North-East India (Assam/Manipur)", "lat": 25.5, "lon": 93.0},
        {"name": "Pacific Ring of Fire (Japan)", "lat": 36.2, "lon": 138.2},
        {"name": "Indonesian Archipelago", "lat": -0.78, "lon": 113.9},
        {"name": "California Coast, USA", "lat": 36.7, "lon": -119.4},
        {"name": "Chilean Subduction Zone", "lat": -35.6, "lon": -71.5},
        {"name": "Turkey Fault Zone", "lat": 39.0, "lon": 35.0},
        {"name": "Nepal Alpine Belt", "lat": 28.3, "lon": 84.1}
    ]

    eq_rows = []
    for i in range(1050):
        reg = np.random.choice(regions)
        lat = reg["lat"] + np.random.normal(0, 1.2)
        lon = reg["lon"] + np.random.normal(0, 1.2)
        magnitude = round(float(np.random.exponential(scale=1.2) + 2.5), 1)
        magnitude = min(magnitude, 9.2)
        depth = round(float(np.random.gamma(shape=2, scale=15.0) + 5.0), 1)
        
        days_offset = np.random.randint(0, 365 * 6)
        minutes_offset = np.random.randint(0, 1440)
        timestamp = start_date + timedelta(days=days_offset, minutes=minutes_offset)

        # Calculate preliminary risk score: function of magnitude & shallow depth
        depth_factor = max(0.2, (100 - depth) / 100)
        risk_score = round(min(100.0, (magnitude ** 2.2) * 1.5 * depth_factor), 1)
        tsunami = 1 if magnitude >= 6.8 and depth < 50 else 0
        
        casualties = 0
        economic_loss = 0.0
        if magnitude > 5.5:
            casualties = int(np.random.exponential(scale=(magnitude ** 3.0) * 0.8))
            economic_loss = round(float(np.random.exponential(scale=(magnitude ** 3.5) * 0.05)), 2)

        eq_rows.append({
            "event_id": f"EQ-{2018 + days_offset // 365}-{i+1001:04d}",
            "magnitude": magnitude,
            "depth": depth,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "location": reg["name"],
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": risk_score,
            "tsunami_warning": tsunami,
            "casualties": casualties,
            "economic_loss_millions": economic_loss
        })

    eq_df = pd.DataFrame(eq_rows)
    insert_df(eq_df, "earthquakes", if_exists="append")

    # -------------------------------------------------------------
    # 2. GENERATE 520 FLOOD INCIDENTS
    # -------------------------------------------------------------
    print("Generating 520 flood incidents...")
    flood_regions = [
        {"name": "Assam Brahmaputra Valley", "lat": 26.1, "lon": 91.7},
        {"name": "Kerala Coastal Basin", "lat": 9.9, "lon": 76.2},
        {"name": "Bihar Gangetic Plains", "lat": 25.6, "lon": 85.1},
        {"name": "Mumbai Urban Basin", "lat": 19.0, "lon": 72.8},
        {"name": "Chennai Coastal Zone", "lat": 13.0, "lon": 80.2},
        {"name": "Odisha Mahanadi Delta", "lat": 20.4, "lon": 85.8},
        {"name": "Uttarakhand River Basins", "lat": 30.3, "lon": 78.0},
        {"name": "West Bengal Delta", "lat": 22.5, "lon": 88.3}
    ]

    flood_rows = []
    for i in range(520):
        reg = np.random.choice(flood_regions)
        lat = reg["lat"] + np.random.normal(0, 0.4)
        lon = reg["lon"] + np.random.normal(0, 0.4)
        rainfall = round(float(np.random.gamma(shape=3, scale=40.0) + 50.0), 1)
        threshold = 5.0
        river_level = round(float(threshold + (rainfall / 100.0) * np.random.uniform(0.5, 2.5)), 2)
        
        days_offset = np.random.randint(0, 365 * 6)
        timestamp = start_date + timedelta(days=days_offset)

        ratio = river_level / threshold
        if ratio > 1.8:
            severity = "Severe"
            risk_score = round(min(100.0, 75.0 + ratio * 8), 1)
        elif ratio > 1.3:
            severity = "High"
            risk_score = round(min(100.0, 50.0 + ratio * 12), 1)
        else:
            severity = "Moderate"
            risk_score = round(min(100.0, 25.0 + ratio * 15), 1)

        affected_pop = int((rainfall * 80) * np.random.uniform(5, 50))

        flood_rows.append({
            "event_id": f"FLD-{2018 + days_offset // 365}-{i+1001:04d}",
            "region": reg["name"],
            "rainfall_mm": rainfall,
            "river_level_m": river_level,
            "threshold_m": threshold,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "severity": severity,
            "affected_population": affected_pop,
            "risk_score": risk_score
        })

    flood_df = pd.DataFrame(flood_rows)
    insert_df(flood_df, "floods", if_exists="append")

    # -------------------------------------------------------------
    # 3. GENERATE 220 CYCLONE EVENTS
    # -------------------------------------------------------------
    print("Generating 220 cyclone events...")
    cyclone_names = [
        "Fani", "Amphan", "Nivar", "Tauktae", "Yaas", "Gulab", "Asani", "Sitrang", 
        "Mandous", "Biparjoy", "Tej", "Hamoon", "Midhili", "Michaung", "Remal",
        "Vayu", "Maha", "Bulbul", "Kyarr", "Gati", "Nivar", "Burevi"
    ]

    cyclone_rows = []
    for i in range(220):
        name = np.random.choice(cyclone_names) + f"-{2018 + (i % 6)}"
        lat = np.random.uniform(8.0, 22.0)
        lon = np.random.uniform(68.0, 92.0)
        wind_speed = round(float(np.random.uniform(65.0, 260.0)), 1)
        pressure = round(float(1010.0 - (wind_speed * 0.35)), 1)
        
        if wind_speed >= 220:
            cat = "Super Cyclonic Storm (Cat 5)"
        elif wind_speed >= 165:
            cat = "Extremely Severe Cyclonic Storm (Cat 4)"
        elif wind_speed >= 120:
            cat = "Very Severe Cyclonic Storm (Cat 3)"
        elif wind_speed >= 90:
            cat = "Severe Cyclonic Storm (Cat 2)"
        else:
            cat = "Cyclonic Storm (Cat 1)"

        days_offset = np.random.randint(0, 365 * 6)
        timestamp = start_date + timedelta(days=days_offset)
        landfall = 1 if np.random.rand() > 0.3 else 0
        risk_score = round(min(100.0, (wind_speed / 260.0) * 100.0), 1)

        cyclone_rows.append({
            "event_id": f"CYC-{2018 + days_offset // 365}-{i+1001:04d}",
            "cyclone_name": name,
            "wind_speed_kmh": wind_speed,
            "pressure_hpa": pressure,
            "category": cat,
            "latitude": round(lat, 4),
            "longitude": round(lon, 4),
            "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "risk_score": risk_score,
            "landfall_predicted": landfall
        })

    cyclone_df = pd.DataFrame(cyclone_rows)
    insert_df(cyclone_df, "cyclones", if_exists="append")

    # -------------------------------------------------------------
    # 4. GENERATE 55 EMERGENCY SHELTERS
    # -------------------------------------------------------------
    print("Generating 55 emergency shelter locations...")
    cities_shelters = [
        ("Mumbai Central Relief Shelter", "Mumbai", 18.9696, 72.8193, 2500),
        ("Bandra Community Resilience Hub", "Mumbai", 19.0596, 72.8295, 1800),
        ("Thane North Disaster Center", "Mumbai", 19.2183, 72.9781, 1500),
        ("Chennai Coastal Multi-purpose Shelter", "Chennai", 13.0827, 80.2707, 3000),
        ("Marina Beach Flood Relief Shelter", "Chennai", 13.0475, 80.2824, 2000),
        ("Adyar Safety Complex", "Chennai", 13.0012, 80.2565, 1200),
        ("Kochi Port Trust Evacuation Center", "Kerala", 9.9312, 76.2673, 2200),
        ("Alappuzha Waterway Relief Camp", "Kerala", 9.4981, 76.3388, 1500),
        ("Wayanad Hill Rescue Shelter", "Kerala", 11.6854, 76.1320, 1000),
        ("Kolkata Salt Lake Emergency Hub", "Kolkata", 22.5726, 88.4149, 3500),
        ("Howrah Station Relief Complex", "Kolkata", 22.5851, 88.3413, 2800),
        ("Guwahati Brahmaputra River Shelter", "Assam", 26.1445, 91.7362, 2400),
        ("Silchar Cachar Rescue Center", "Assam", 24.8333, 92.7789, 1600),
        ("Patna Gangetic Relief Hub", "Bihar", 25.5941, 85.1376, 3200),
        ("Bhuj Kutch Earthquake Center", "Gujarat", 23.2420, 69.6669, 2000),
        ("Ahmedabad Stadium Evacuation Center", "Gujarat", 23.0225, 72.5714, 4000),
        ("Puri Coastal Cyclone Shelter 1", "Odisha", 19.8135, 85.8312, 2500),
        ("Paradip Port High-Safety Hub", "Odisha", 20.3165, 86.6114, 3000),
        ("Bhubaneswar State Relief HQ", "Odisha", 20.2961, 85.8245, 3500),
        ("Dehradun Valley Emergency Hub", "Uttarakhand", 30.3165, 78.0322, 1800),
        ("Shimla Mountain Rescue Base", "Himachal Pradesh", 31.1048, 77.1734, 1200),
        ("Srinagar Flood Protection Hub", "Jammu & Kashmir", 34.0837, 74.7973, 2200),
        ("Port Blair Island Evacuation Hub", "Andaman", 11.6234, 92.7265, 1500),
        ("Visakhapatnam Cyclone Safe-Zone", "Andhra Pradesh", 17.6868, 83.2185, 2800),
        ("Vijayawada Krishna River Center", "Andhra Pradesh", 16.5062, 80.6480, 2000)
    ]

    shelter_rows = []
    shelter_idx = 1
    for name, region, base_lat, base_lon, cap in cities_shelters:
        # Generate 2 shelters per entry with slight spatial variance
        for s_num in range(1, 3):
            lat = round(base_lat + (s_num - 1) * 0.05 + np.random.uniform(-0.02, 0.02), 4)
            lon = round(base_lon + (s_num - 1) * 0.05 + np.random.uniform(-0.02, 0.02), 4)
            capacity = int(cap * (1.0 + (s_num - 1) * 0.3))
            occupancy = int(capacity * np.random.uniform(0.1, 0.65))
            
            shelter_rows.append({
                "name": f"{name} {s_num}",
                "region": region,
                "address": f"Sector {s_num*3}, Main Highway, {region}",
                "capacity": capacity,
                "current_occupancy": occupancy,
                "latitude": lat,
                "longitude": lon,
                "status": "Active / Open" if occupancy < capacity * 0.9 else "Near Capacity",
                "contact_number": f"+91 98765 {shelter_idx:05d}",
                "supplies_food_days": int(np.random.randint(5, 15)),
                "supplies_water_days": int(np.random.randint(5, 15)),
                "medical_kits": int(np.random.randint(30, 150))
            })
            shelter_idx += 1

    shelter_df = pd.DataFrame(shelter_rows)
    insert_df(shelter_df, "shelters", if_exists="append")

    # -------------------------------------------------------------
    # 5. GENERATE 50 REGIONAL POPULATION DENSITY & VULNERABILITY RECORDS
    # -------------------------------------------------------------
    print("Generating 50 regional population vulnerability records...")
    states_data = [
        ("Maharashtra", "Mumbai City", 12500000, 603.4, 0.85),
        ("Maharashtra", "Thane District", 11000000, 4214.0, 0.72),
        ("Tamil Nadu", "Chennai Metropolitan", 10900000, 426.0, 0.82),
        ("West Bengal", "Kolkata Metropolitan", 14800000, 1850.0, 0.88),
        ("Gujarat", "Kutch Region", 2100000, 45674.0, 0.78),
        ("Gujarat", "Ahmedabad Urban", 8400000, 505.0, 0.65),
        ("Kerala", "Wayand & Idukki Hills", 1800000, 4100.0, 0.84),
        ("Kerala", "Alappuzha & Ernakulam", 4200000, 3200.0, 0.81),
        ("Assam", "Guwahati & Kamrup", 2500000, 1528.0, 0.89),
        ("Assam", "Cachar & Silchar", 1700000, 3786.0, 0.86),
        ("Bihar", "Patna Metropolitan", 5800000, 3202.0, 0.83),
        ("Odisha", "Puri & Balasore Coast", 3100000, 3479.0, 0.91),
        ("Uttarakhand", "Dehradun & Chamoli", 1700000, 7500.0, 0.87),
        ("Himachal Pradesh", "Shimla & Kullu", 1400000, 5131.0, 0.75),
        ("Andhra Pradesh", "Visakhapatnam Urban", 2300000, 540.0, 0.79),
        ("Andhra Pradesh", "Krishna Delta", 4500000, 8727.0, 0.82),
        ("Jammu & Kashmir", "Srinagar Valley", 1350000, 294.0, 0.80),
        ("Andaman Islands", "Port Blair Belt", 380000, 8249.0, 0.85)
    ]

    pop_rows = []
    p_idx = 1
    for state, base_reg, pop, area, vul in states_data:
        for sub_i in range(1, 4):
            reg_name = f"{base_reg} Zone-{sub_i}"
            sub_pop = int(pop / 3 * np.random.uniform(0.8, 1.2))
            sub_area = round(area / 3 * np.random.uniform(0.8, 1.2), 1)
            density = round(sub_pop / max(1.0, sub_area), 1)
            vulnerability = round(min(1.0, max(0.1, vul + np.random.uniform(-0.08, 0.08))), 2)
            infra_count = int(np.random.randint(12, 60))

            pop_rows.append({
                "region": reg_name,
                "state": state,
                "population": sub_pop,
                "area_sq_km": sub_area,
                "density_per_sq_km": density,
                "vulnerability_index": vulnerability,
                "critical_infrastructure_count": infra_count
            })
            p_idx += 1
            if len(pop_rows) >= 50:
                break
        if len(pop_rows) >= 50:
            break

    pop_df = pd.DataFrame(pop_rows)
    insert_df(pop_df, "population_density", if_exists="append")

    # -------------------------------------------------------------
    # 6. INITIAL ALERT SUBSCRIBERS
    # -------------------------------------------------------------
    print("Generating initial alert subscribers...")
    subscribers = [
        {"name": "Dr. Ramesh Sharma", "email": "ramesh.sharma@ndrf.gov.in", "phone": "+91 98111 22334", "region": "Maharashtra", "preferred_disasters": "All", "alert_channel": "Both"},
        {"name": "Priya Nair", "email": "priya.nair@kerala-disaster.org", "phone": "+91 98222 33445", "region": "Kerala", "preferred_disasters": "Flood", "alert_channel": "Email"},
        {"name": "Anil Kumar", "email": "anil.kumar@odisha-sdma.gov.in", "phone": "+91 98333 44556", "region": "Odisha", "preferred_disasters": "Cyclone", "alert_channel": "SMS"},
        {"name": "Sunita Das", "email": "sunita.das@assam-relief.org", "phone": "+91 98444 55667", "region": "Assam", "preferred_disasters": "Flood", "alert_channel": "Both"},
        {"name": "Vikram Singh", "email": "vikram.singh@himalaya-ngo.org", "phone": "+91 98555 66778", "region": "Uttarakhand", "preferred_disasters": "Earthquake", "alert_channel": "SMS"}
    ]
    sub_df = pd.DataFrame(subscribers)
    insert_df(sub_df, "alert_subscribers", if_exists="append")

    # -------------------------------------------------------------
    # 7. INITIAL CITIZEN REPORTS
    # -------------------------------------------------------------
    print("Generating sample citizen emergency reports...")
    reports = [
        {"reporter_name": "Aarav Patel", "phone": "+91 98760 11111", "disaster_type": "Flood", "location_name": "Kurla West, Mumbai", "latitude": 19.0657, "longitude": 72.8797, "description": "Water logging reached 3 feet near Mithi river overflow.", "severity": "High", "status": "Verified", "verified_by": "Control Room A"},
        {"reporter_name": "Meera Joshi", "phone": "+91 98760 22222", "disaster_type": "Earthquake", "location_name": "Tehri Garhwal", "latitude": 30.3753, "longitude": 78.4802, "description": "Felt violent tremors, minor cracks in village masonry walls.", "severity": "Moderate", "status": "Verified", "verified_by": "Control Room B"},
        {"reporter_name": "Suresh Roy", "phone": "+91 98760 33333", "disaster_type": "Cyclone", "location_name": "Puri Beach Front", "latitude": 19.7983, "longitude": 85.8249, "description": "High velocity sea winds damaging electric poles and trees.", "severity": "Critical", "status": "Under Investigation", "verified_by": None}
    ]
    rep_df = pd.DataFrame(reports)
    insert_df(rep_df, "citizen_reports", if_exists="append")

    # -------------------------------------------------------------
    # 8. INITIAL RESOURCE ALLOCATIONS
    # -------------------------------------------------------------
    print("Generating active disaster response operations...")
    allocations = [
        {"operation_name": "Op Brahmaputra Shield", "region": "Assam", "personnel_deployed": 450, "medical_units": 15, "food_packets": 25000, "water_liters": 50000, "rescue_boats": 40, "helicopters": 4, "status": "Active"},
        {"operation_name": "Op Coastal Sentinel", "region": "Odisha", "personnel_deployed": 600, "medical_units": 20, "food_packets": 40000, "water_liters": 80000, "rescue_boats": 25, "helicopters": 6, "status": "Active"},
        {"operation_name": "Op Western Shield", "region": "Maharashtra", "personnel_deployed": 300, "medical_units": 10, "food_packets": 15000, "water_liters": 30000, "rescue_boats": 18, "helicopters": 2, "status": "Standby"}
    ]
    alloc_df = pd.DataFrame(allocations)
    insert_df(alloc_df, "resource_allocations", if_exists="append")

    print("Sample Data Generation Complete!")

if __name__ == "__main__":
    generate_sample_data()
