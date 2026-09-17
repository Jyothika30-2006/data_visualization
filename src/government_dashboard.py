"""
Module 8: Government & Emergency Resource Operations Dashboard
Provides authorities with active operations tracking, shelter capacity management,
resource allocation controls (boats, helicopters, medical units), and deployment status.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any

from src.database import execute_query, query_to_df


def get_active_operations() -> pd.DataFrame:
    """Retrieve all active disaster response operations."""
    return query_to_df("SELECT * FROM resource_allocations WHERE status = 'Active'")


def update_resource_allocation(
    operation_name: str,
    region: str,
    personnel: int,
    medical_units: int,
    food_packets: int,
    water_liters: int,
    boats: int,
    helicopters: int,
    status: str = "Active"
) -> int:
    """
    Create or update an official government emergency operation deployment.
    """
    query = """
    INSERT INTO resource_allocations 
    (operation_name, region, personnel_deployed, medical_units, food_packets, water_liters, rescue_boats, helicopters, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    return execute_query(query, (
        operation_name, region, personnel, medical_units,
        food_packets, water_liters, boats, helicopters, status
    ))


def create_shelter_capacity_chart(shelters_df: pd.DataFrame) -> go.Figure:
    """
    Bar chart visualization comparing capacity vs occupancy across emergency shelters.
    """
    if shelters_df.empty:
        return go.Figure()

    top_shelters = shelters_df.head(15).copy()

    fig = go.Figure(data=[
        go.Bar(name="Occupied Seats", x=top_shelters["name"], y=top_shelters["current_occupancy"], marker_color="#dc3545"),
        go.Bar(name="Total Capacity", x=top_shelters["name"], y=top_shelters["capacity"], marker_color="#28a745")
    ])

    fig.update_layout(
        barmode="overlay",
        title="<b>Emergency Shelter Capacity vs Current Occupancy</b>",
        xaxis_title="Shelter Name",
        yaxis_title="People Capacity",
        template="plotly_white",
        xaxis_tickangle=-45
    )
    return fig


def create_resource_deployment_chart(alloc_df: pd.DataFrame) -> go.Figure:
    """
    Grouped bar chart for deployed emergency assets across regions.
    """
    if alloc_df.empty:
        return go.Figure()

    fig = go.Figure(data=[
        go.Bar(name="NDRF Personnel", x=alloc_df["operation_name"], y=alloc_df["personnel_deployed"], marker_color="#007bff"),
        go.Bar(name="Rescue Boats", x=alloc_df["operation_name"], y=alloc_df["rescue_boats"], marker_color="#17a2b8"),
        go.Bar(name="Medical Units", x=alloc_df["operation_name"], y=alloc_df["medical_units"], marker_color="#28a745"),
        go.Bar(name="Helicopters", x=alloc_df["operation_name"], y=alloc_df["helicopters"], marker_color="#6f42c1")
    ])

    fig.update_layout(
        barmode="group",
        title="<b>Active Response Deployment Assets by Operation</b>",
        xaxis_title="Operation Name",
        yaxis_title="Units Deployed",
        template="plotly_white"
    )
    return fig
