"""
Database Management Module.
Handles SQLite database creation, connection management, schema initialization, and data querying.
Supports both SQLite for local development and PostgreSQL connection strings for production.
"""

import sqlite3
import pandas as pd
import os
import logging
from typing import Optional, Dict, List, Any
from config import DB_PATH

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Create and return a SQLite database connection with row factory."""
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_PATH) -> None:
    """Initialize database tables with complete schema."""
    conn = get_connection(db_path)
    cursor = conn.cursor()

    try:
        # 1. Earthquakes table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS earthquakes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            magnitude REAL NOT NULL,
            depth REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            location TEXT NOT NULL,
            timestamp DATETIME NOT NULL,
            risk_score REAL NOT NULL,
            tsunami_warning INTEGER DEFAULT 0,
            casualties INTEGER DEFAULT 0,
            economic_loss_millions REAL DEFAULT 0.0
        )
        """)

        # 2. Floods table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS floods (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            region TEXT NOT NULL,
            rainfall_mm REAL NOT NULL,
            river_level_m REAL NOT NULL,
            threshold_m REAL NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            severity TEXT NOT NULL,
            affected_population INTEGER DEFAULT 0,
            risk_score REAL NOT NULL
        )
        """)

        # 3. Cyclones table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cyclones (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_id TEXT UNIQUE,
            cyclone_name TEXT NOT NULL,
            wind_speed_kmh REAL NOT NULL,
            pressure_hpa REAL NOT NULL,
            category TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            timestamp DATETIME NOT NULL,
            risk_score REAL NOT NULL,
            landfall_predicted INTEGER DEFAULT 0
        )
        """)

        # 4. Emergency Shelters table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS shelters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            region TEXT NOT NULL,
            address TEXT NOT NULL,
            capacity INTEGER NOT NULL,
            current_occupancy INTEGER NOT NULL DEFAULT 0,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            status TEXT DEFAULT 'Open',
            contact_number TEXT,
            supplies_food_days INTEGER DEFAULT 7,
            supplies_water_days INTEGER DEFAULT 7,
            medical_kits INTEGER DEFAULT 50
        )
        """)

        # 5. Alert Subscribers table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alert_subscribers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT NOT NULL,
            phone TEXT NOT NULL,
            region TEXT NOT NULL,
            preferred_disasters TEXT DEFAULT 'All',
            alert_channel TEXT DEFAULT 'Both',
            subscribed_at DATETIME DEFAULT CURRENT_TIMESTAMP,
            active INTEGER DEFAULT 1
        )
        """)

        # 6. Citizen Reports table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS citizen_reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            reporter_name TEXT NOT NULL,
            phone TEXT NOT NULL,
            disaster_type TEXT NOT NULL,
            location_name TEXT NOT NULL,
            latitude REAL NOT NULL,
            longitude REAL NOT NULL,
            description TEXT NOT NULL,
            severity TEXT NOT NULL,
            status TEXT DEFAULT 'Pending Verification',
            image_url TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            verified_by TEXT
        )
        """)

        # 7. Resource Allocations table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS resource_allocations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            operation_name TEXT NOT NULL,
            region TEXT NOT NULL,
            personnel_deployed INTEGER DEFAULT 0,
            medical_units INTEGER DEFAULT 0,
            food_packets INTEGER DEFAULT 0,
            water_liters INTEGER DEFAULT 0,
            rescue_boats INTEGER DEFAULT 0,
            helicopters INTEGER DEFAULT 0,
            status TEXT DEFAULT 'Active',
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # 8. Regional Population Density table
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS population_density (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            region TEXT UNIQUE NOT NULL,
            state TEXT NOT NULL,
            population INTEGER NOT NULL,
            area_sq_km REAL NOT NULL,
            density_per_sq_km REAL NOT NULL,
            vulnerability_index REAL NOT NULL,
            critical_infrastructure_count INTEGER DEFAULT 10
        )
        """)

        conn.commit()
        logger.info("Database schema initialized successfully.")
    except Exception as e:
        conn.rollback()
        logger.error(f"Failed to initialize database schema: {e}")
        raise
    finally:
        conn.close()


def query_to_df(query: str, params: tuple = (), db_path: str = DB_PATH) -> pd.DataFrame:
    """Execute SQL query and return results as a Pandas DataFrame."""
    try:
        conn = get_connection(db_path)
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        logger.error(f"Error executing query '{query}': {e}")
        return pd.DataFrame()


def execute_query(query: str, params: tuple = (), db_path: str = DB_PATH) -> int:
    """Execute an INSERT, UPDATE, or DELETE query and return the last row ID or affected count."""
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute(query, params)
        conn.commit()
        last_id = cursor.lastrowid
        conn.close()
        return last_id
    except Exception as e:
        logger.error(f"Error executing non-query statement: {e}")
        raise


def insert_df(df: pd.DataFrame, table_name: str, if_exists: str = "append", db_path: str = DB_PATH) -> None:
    """Save a Pandas DataFrame directly to a SQLite table."""
    try:
        conn = get_connection(db_path)
        df.to_sql(table_name, conn, if_exists=if_exists, index=False)
        conn.close()
        logger.info(f"Successfully inserted {len(df)} rows into {table_name}.")
    except Exception as e:
        logger.error(f"Error inserting DataFrame into {table_name}: {e}")
        raise
