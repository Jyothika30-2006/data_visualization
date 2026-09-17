"""
Innovative Feature 1: Social Media Disaster Sentiment & Panic Index Engine.
Analyzes crowdsourced social posts/tweets during active disaster events to detect distress levels,
urgent help requests, and trending disaster topics.
"""

import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, List, Any


DISASTER_KEYWORDS = {
    "CRITICAL_HELP": ["trapped", "help", "sos", "flooded", "collapsed", "rescue", "drowning", "bleeding", "stranded"],
    "MODERATE_ALERT": ["heavy rain", "wind", "shaking", "power cut", "damage", "shelter", "water level", "tremor"],
    "POSITIVE_RECOVERY": ["safe", "rescued", "food distributed", "shelter open", "clear", "evacuated safely", "ndrf arrived"]
}


def generate_mock_disaster_tweets(count: int = 30) -> pd.DataFrame:
    """Simulate real-time crowdsourced social media posts during disaster events."""
    locations = ["Mumbai Kurla", "Chennai Adyar", "Assam Guwahati", "Kerala Wayanad", "Puri Odisha", "Gujarat Kutch"]
    
    samples = [
        "SOS! Flood waters entering second floor at Kurla West! Need rescue boat immediately #MumbaiFloods #SOS",
        "Felt strong tremors for 15 seconds in Kutch! People running into open streets. #EarthquakeAlert",
        "NDRF team reached Adyar flooded colony. Distributing food packets and water bottles. #ChennaiRelief",
        "High speed cyclone winds tearing roof sheets off houses near Puri port! Stay indoors! #CycloneWarning",
        "Water logging cleared on main arterial highway. Traffic moving slowly now. #MumbaiUpdate",
        "Trapped in house with elderly family near Guwahati river embankment. Water rising fast! #AssamFloods",
        "Emergency shelter at Salt Lake complex is open and has food and medical supplies available.",
        "Power outage across entire town due to fallen electric poles in cyclone wind. #CycloneRelief"
    ]
    
    np.random.seed(42)
    rows = []
    for i in range(count):
        text = np.random.choice(samples)
        loc = np.random.choice(locations)
        
        # Simple sentiment / urgency calculation
        score = 0
        if any(k in text.lower() for k in DISASTER_KEYWORDS["CRITICAL_HELP"]):
            sentiment = "Critical / SOS"
            panic_score = round(float(np.random.uniform(75, 98)), 1)
        elif any(k in text.lower() for k in DISASTER_KEYWORDS["POSITIVE_RECOVERY"]):
            sentiment = "Positive / Relieved"
            panic_score = round(float(np.random.uniform(10, 35)), 1)
        else:
            sentiment = "Moderate Distress"
            panic_score = round(float(np.random.uniform(40, 70)), 1)

        rows.append({
            "post_id": f"TWEET-{1000+i}",
            "location": loc,
            "text": text,
            "sentiment_category": sentiment,
            "panic_distress_score": panic_score,
            "retweets": int(np.random.randint(5, 450))
        })
        
    return pd.DataFrame(rows)


def create_sentiment_summary_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart showing distribution of public distress categories."""
    if df.empty:
        return go.Figure()

    counts = df["sentiment_category"].value_counts().reset_index()
    counts.columns = ["Category", "Count"]

    fig = px.pie(
        counts,
        names="Category",
        values="Count",
        color="Category",
        color_discrete_map={
            "Critical / SOS": "#dc3545",
            "Moderate Distress": "#ffc107",
            "Positive / Relieved": "#28a745"
        },
        title="<b>Social Media Public Sentiment & Panic Analysis</b>"
    )
    return fig
