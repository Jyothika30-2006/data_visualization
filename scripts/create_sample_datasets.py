"""
Script to generate 5 rich sample disaster datasets:
1. earthquake_dataset.csv (1000+ records)
2. flood_dataset.csv (500+ records)
3. cyclone_dataset.csv (200+ records)
4. tsunami_dataset.csv (150+ records)
5. multidisaster_dataset.csv (1500+ records)
"""

import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

os.makedirs("sample_datasets", exist_ok=True)
np.random.seed(42)
start_date = datetime(2018, 1, 1)

# 1. EARTHQUAKE DATASET (1050 records)
eq_rows = []
regions = [
    ("Himalayan Fault Belt", 30.5, 79.0), ("Gujarat Kutch Zone", 23.5, 70.0),
    ("Andaman Nicobar Basin", 11.7, 92.7), ("Assam North-East", 25.5, 93.0),
    ("Japan Ring of Fire", 36.2, 138.2), ("Indonesia Subduction Zone", -0.78, 113.9),
    ("California Fault", 36.7, -119.4), ("Turkey East Anatolian", 39.0, 35.0)
]
for i in range(1050):
    name, base_lat, base_lon = regions[i % len(regions)]
    lat = round(base_lat + np.random.normal(0, 1.2), 4)
    lon = round(base_lon + np.random.normal(0, 1.2), 4)
    mag = round(float(min(9.3, max(2.5, np.random.exponential(1.2) + 2.8))), 1)
    depth = round(float(np.random.gamma(2, 15) + 5), 1)
    ts = start_date + timedelta(days=np.random.randint(0, 2500), minutes=np.random.randint(0, 1440))
    tsunami = 1 if mag >= 6.8 and depth < 50 else 0
    cas = int(np.random.exponential(mag ** 2.8 * 0.5)) if mag > 5.2 else 0
    loss = round(float(np.random.exponential(mag ** 3.2 * 0.02)), 2) if mag > 5.2 else 0.0

    eq_rows.append({
        "Event_ID": f"EQ-20{i%8+18}-{i+1000}",
        "Date_Time": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Location_Name": name,
        "Latitude": lat,
        "Longitude": lon,
        "Richter_Magnitude": mag,
        "Focal_Depth_km": depth,
        "Tsunami_Warning_Flag": tsunami,
        "Casualties_Direct": cas,
        "Economic_Loss_USD_Millions": loss
    })
pd.DataFrame(eq_rows).to_csv("sample_datasets/earthquake_dataset.csv", index=False)
print("Generated sample_datasets/earthquake_dataset.csv")

# 2. FLOOD DATASET (520 records)
flood_rows = []
f_regions = [
    ("Brahmaputra Basin Guwahati", 26.1, 91.7), ("Kerala Coastal Wayanad", 9.9, 76.2),
    ("Gangetic Plains Bihar", 25.6, 85.1), ("Mumbai Urban Basin", 19.0, 72.8),
    ("Chennai Coastal Belt", 13.0, 80.2), ("Mahanadi Delta Odisha", 20.4, 85.8)
]
for i in range(520):
    name, base_lat, base_lon = f_regions[i % len(f_regions)]
    lat = round(base_lat + np.random.normal(0, 0.5), 4)
    lon = round(base_lon + np.random.normal(0, 0.5), 4)
    rain = round(float(np.random.gamma(3, 35) + 40), 1)
    thresh = 5.0
    river = round(float(thresh + (rain / 90.0) * np.random.uniform(0.6, 2.2)), 2)
    ts = start_date + timedelta(days=np.random.randint(0, 2500))
    aff_pop = int((rain * 120) * np.random.uniform(2, 35))
    cas = int(aff_pop * 0.0004)
    loss = round(float(aff_pop * 0.0018), 2)

    flood_rows.append({
        "Incident_ID": f"FLD-20{i%8+18}-{i+1000}",
        "Observation_Date": ts.strftime("%Y-%m-%d"),
        "Region_District": name,
        "Latitude": lat,
        "Longitude": lon,
        "24h_Rainfall_mm": rain,
        "River_Gauge_Level_m": river,
        "Flood_Threshold_m": thresh,
        "Inundated_Area_sqkm": round(rain * 1.5, 1),
        "Affected_Population": aff_pop,
        "Casualties": cas,
        "Economic_Damage_Millions": loss
    })
pd.DataFrame(flood_rows).to_csv("sample_datasets/flood_dataset.csv", index=False)
print("Generated sample_datasets/flood_dataset.csv")

# 3. CYCLONE DATASET (220 records)
cyclone_rows = []
c_names = ["Amphan", "Fani", "Tauktae", "Biparjoy", "Michaung", "Remal", "Yaas", "Nivar", "Gulab", "Sitrang"]
for i in range(220):
    c_name = c_names[i % len(c_names)] + f"-{2018 + (i % 8)}"
    lat = round(np.random.uniform(8.0, 22.0), 4)
    lon = round(np.random.uniform(68.0, 92.0), 4)
    wind = round(float(np.random.uniform(65.0, 255.0)), 1)
    pressure = round(float(1013.0 - (wind * 0.36)), 1)
    ts = start_date + timedelta(days=np.random.randint(0, 2500))
    cat = f"Cat {int(min(5, wind // 40))}"
    cas = int((wind / 10) ** 1.8)
    loss = round(float((wind / 10) ** 2.2 * 0.8), 2)

    cyclone_rows.append({
        "Cyclone_ID": f"CYC-20{i%8+18}-{i+1000}",
        "Landfall_Date": ts.strftime("%Y-%m-%d"),
        "Cyclone_System_Name": c_name,
        "Latitude": lat,
        "Longitude": lon,
        "Wind_Speed_kmh": wind,
        "Central_Pressure_hpa": pressure,
        "Storm_Category": cat,
        "Coastal_Landfall_Site": f"Zone-{i%12+1}",
        "Casualties": cas,
        "Economic_Loss_USD_Millions": loss
    })
pd.DataFrame(cyclone_rows).to_csv("sample_datasets/cyclone_dataset.csv", index=False)
print("Generated sample_datasets/cyclone_dataset.csv")

# 4. TSUNAMI DATASET (150 records)
tsunami_rows = []
for i in range(150):
    lat = round(np.random.uniform(-10.0, 35.0), 4)
    lon = round(np.random.uniform(70.0, 140.0), 4)
    wave_h = round(float(np.random.gamma(2, 2.5) + 0.5), 1)
    inund = round(float(wave_h * np.random.uniform(0.5, 2.5)), 1)
    mag = round(float(np.random.uniform(6.5, 9.2)), 1)
    ts = start_date + timedelta(days=np.random.randint(0, 2500))
    cas = int(wave_h ** 2.5 * 12)
    loss = round(float(wave_h ** 2.8 * 1.5), 2)

    tsunami_rows.append({
        "Tsunami_Event_ID": f"TSU-20{i%8+18}-{i+1000}",
        "Event_Date": ts.strftime("%Y-%m-%d"),
        "Coastal_Location": f"Coastal Zone-{i%10+1}",
        "Latitude": lat,
        "Longitude": lon,
        "Trigger_Earthquake_Mag": mag,
        "Max_Wave_Height_m": wave_h,
        "Inundation_Distance_km": inund,
        "Casualties": cas,
        "Economic_Loss_Millions": loss
    })
pd.DataFrame(tsunami_rows).to_csv("sample_datasets/tsunami_dataset.csv", index=False)
print("Generated sample_datasets/tsunami_dataset.csv")

# 5. MULTI-DISASTER DATASET (1500 records)
multi_rows = []
d_types = ["Earthquake", "Flood", "Cyclone", "Tsunami", "Landslide"]
m_locs = ["Mumbai", "Chennai", "Guwahati", "Kolkata", "Wayanad", "Puri", "Patna", "Dehradun", "Shimla", "Kutch"]
for i in range(1500):
    dtype = d_types[i % len(d_types)]
    loc = m_locs[i % len(m_locs)]
    lat = round(np.random.uniform(8.0, 32.0), 4)
    lon = round(np.random.uniform(68.0, 92.0), 4)
    sev = round(float(np.random.uniform(15.0, 98.0)), 1)
    ts = start_date + timedelta(days=np.random.randint(0, 2500))
    cas = int(sev * np.random.uniform(0.2, 5.0))
    loss = round(float(sev * np.random.uniform(0.1, 8.0)), 2)

    multi_rows.append({
        "Global_Disaster_ID": f"DIS-20{i%8+18}-{i+1000}",
        "Disaster_Category": dtype,
        "Event_Timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
        "Affected_Location": loc,
        "Latitude": lat,
        "Longitude": lon,
        "Calculated_Severity_Score": sev,
        "Casualties": cas,
        "Economic_Loss_Millions": loss
    })
pd.DataFrame(multi_rows).to_csv("sample_datasets/multidisaster_dataset.csv", index=False)
print("Generated sample_datasets/multidisaster_dataset.csv")
