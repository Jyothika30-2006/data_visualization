"""
Configuration settings for Natural Disaster Intelligence Dashboard.
Handles environment variables, database settings, API configurations, and default parameters.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).resolve().parent

# Database configuration
DB_DIR = BASE_DIR / "data"
DB_DIR.mkdir(exist_ok=True)
DB_PATH = os.getenv("DB_PATH", str(DB_DIR / "disaster_intelligence.db"))
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DB_PATH}")

# API Configurations (loaded from environment variables)
USGS_EARTHQUAKE_API_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query"
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "mock_key_demo_mode")
OPENWEATHER_API_URL = "https://api.openweathermap.org/data/2.5"
OPEN_METEO_API_URL = "https://api.open-meteo.com/v1/forecast"

# Twilio SMS API Configuration
TWILIO_ACCOUNT_SID = os.getenv("TWILIO_ACCOUNT_SID", "")
TWILIO_AUTH_TOKEN = os.getenv("TWILIO_AUTH_TOKEN", "")
TWILIO_PHONE_NUMBER = os.getenv("TWILIO_PHONE_NUMBER", "")

# Email SMTP Configuration
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_EMAIL = os.getenv("SMTP_EMAIL", "alerts@disaster-intel.org")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# Application Settings
APP_TITLE = "Natural Disaster Intelligence & Response Dashboard"
APP_SUBTITLE = "Real-Time Disaster Monitoring, Risk Prediction & Emergency Response Management"
REFRESH_INTERVAL_MINUTES = 5

# Default Map Center (Global / India focus for B.Tech project demo)
DEFAULT_LAT = 20.5937
DEFAULT_LON = 78.9629
DEFAULT_ZOOM = 5

# Disaster Risk Thresholds (0 - 100 score)
RISK_LEVELS = {
    "LOW": {"min": 0, "max": 30, "color": "#28a745", "label": "Low Risk"},
    "MODERATE": {"min": 31, "max": 60, "color": "#ffc107", "label": "Moderate Risk"},
    "HIGH": {"min": 61, "max": 80, "color": "#fd7e14", "label": "High Risk"},
    "CRITICAL": {"min": 81, "max": 100, "color": "#dc3545", "label": "Critical Warning"}
}

# Disaster Types
DISASTER_TYPES = ["Earthquake", "Flood", "Cyclone", "Tsunami", "Landslide"]

# Cache Settings
CACHE_TTL_SECONDS = 300  # 5 minutes cache for API calls
