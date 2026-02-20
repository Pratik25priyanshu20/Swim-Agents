#swim/agents/homogen/utils/geo_utils.py

import pandas as pd

def compute_geo_bounds(df: pd.DataFrame) -> dict:
    """Compute geographic bounding box from lat/lon columns."""
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return {"error": "Latitude and Longitude columns not found."}

    min_lat = df['latitude'].min()
    max_lat = df['latitude'].max()
    min_lon = df['longitude'].min()
    max_lon = df['longitude'].max()

    return {
        "min_latitude": min_lat,
        "max_latitude": max_lat,
        "min_longitude": min_lon,
        "max_longitude": max_lon,
    }