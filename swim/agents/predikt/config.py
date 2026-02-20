# swim/agents/predikt/config.py

"""
Configuration file for PREDIKT agent system
"""

from pathlib import Path
from typing import Dict, Any
import os

# ============================================================
# PATHS
# ============================================================
BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "outputs"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"

# Create directories
for directory in [OUTPUT_DIR, LOGS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# MODEL CONFIGURATION
# ============================================================
MODEL_CONFIG = {
    "name": "PREDIKT HAB Forecasting Model",
    "version": "3.0",
    "architecture": "Hybrid LSTM-Transformer with Attention",
    "ensemble_size": 5,
    "base_accuracy": 0.917,
    "spatial_resolution_km": 1.0,
    "temporal_resolution_hours": 6,
    "training_epochs": 150,
    "batch_size": 32,
    "learning_rate": 0.001,
    "dropout_rate": 0.2,
    "sequence_length": 30,  # days of historical data
}


# ============================================================
# LAKE DATABASE
# ============================================================
GERMAN_LAKES = {
    "Bodensee": {
        "lat": 47.5923,
        "lon": 9.1881,
        "area_km2": 536,
        "depth_max_m": 251,
        "depth_mean_m": 90,
        "volume_km3": 48,
        "elevation_m": 395,
        "trophic_status": "oligotrophic",
        "region": "Baden-Württemberg/Bavaria"
    },
    "Chiemsee": {
        "lat": 47.8756,
        "lon": 12.4258,
        "area_km2": 80,
        "depth_max_m": 73,
        "depth_mean_m": 25,
        "volume_km3": 2.05,
        "elevation_m": 518,
        "trophic_status": "mesotrophic",
        "region": "Bavaria"
    },
    "Starnberger See": {
        "lat": 47.9039,
        "lon": 11.3253,
        "area_km2": 56,
        "depth_max_m": 128,
        "depth_mean_m": 53,
        "volume_km3": 3.0,
        "elevation_m": 584,
        "trophic_status": "oligotrophic",
        "region": "Bavaria"
    },
    "Ammersee": {
        "lat": 47.9975,
        "lon": 11.1244,
        "area_km2": 47,
        "depth_max_m": 81,
        "depth_mean_m": 37,
        "volume_km3": 1.75,
        "elevation_m": 533,
        "trophic_status": "mesotrophic",
        "region": "Bavaria"
    },
    "Müritz": {
        "lat": 53.4333,
        "lon": 12.7000,
        "area_km2": 117,
        "depth_max_m": 31,
        "depth_mean_m": 6.5,
        "volume_km3": 0.76,
        "elevation_m": 62,
        "trophic_status": "eutrophic",
        "region": "Mecklenburg-Vorpommern"
    }
}


# ============================================================
# FEATURE IMPORTANCE WEIGHTS
# ============================================================
FEATURE_WEIGHTS = {
    "chlorophyll_a": 0.22,
    "water_temperature": 0.19,
    "turbidity": 0.13,
    "solar_radiation": 0.11,
    "total_phosphorus": 0.09,
    "total_nitrogen": 0.08,
    "precipitation": 0.07,
    "dissolved_oxygen": 0.05,
    "ph": 0.03,
    "wind_speed": 0.03
}


# ============================================================
# RISK THRESHOLDS
# ============================================================
RISK_THRESHOLDS = {
    "low": {"min": 0.0, "max": 0.30, "color": "green", "emoji": "🟢"},
    "moderate": {"min": 0.30, "max": 0.60, "color": "yellow", "emoji": "🟡"},
    "high": {"min": 0.60, "max": 0.80, "color": "orange", "emoji": "🟠"},
    "critical": {"min": 0.80, "max": 1.00, "color": "red", "emoji": "🔴"}
}


# ============================================================
# FORECAST HORIZONS
# ============================================================
FORECAST_HORIZONS = {
    3: {
        "name": "Short-term",
        "description": "Immediate 3-day forecast with highest accuracy",
        "uncertainty_base": 0.08,
        "use_case": "Immediate decision-making and emergency response"
    },
    7: {
        "name": "Medium-term",
        "description": "Balanced 7-day forecast for planning",
        "uncertainty_base": 0.10,
        "use_case": "Weekly monitoring and preventive measures"
    },
    14: {
        "name": "Long-term",
        "description": "Strategic 14-day outlook with higher uncertainty",
        "uncertainty_base": 0.15,
        "use_case": "Strategic planning and resource allocation"
    }
}


# ============================================================
# ENVIRONMENTAL PARAMETER RANGES
# ============================================================
PARAMETER_RANGES = {
    "chlorophyll_a": {
        "unit": "µg/L",
        "normal": (2, 10),
        "elevated": (10, 20),
        "high": (20, 50),
        "critical": (50, 100)
    },
    "water_temperature": {
        "unit": "°C",
        "winter": (4, 10),
        "spring": (10, 18),
        "summer": (18, 28),
        "fall": (10, 18)
    },
    "turbidity": {
        "unit": "NTU",
        "clear": (0, 3),
        "moderate": (3, 8),
        "turbid": (8, 15),
        "very_turbid": (15, 50)
    },
    "total_phosphorus": {
        "unit": "mg/L",
        "oligotrophic": (0.001, 0.010),
        "mesotrophic": (0.010, 0.030),
        "eutrophic": (0.030, 0.100),
        "hypereutrophic": (0.100, 0.500)
    },
    "total_nitrogen": {
        "unit": "mg/L",
        "low": (0.1, 0.5),
        "moderate": (0.5, 1.0),
        "elevated": (1.0, 2.0),
        "high": (2.0, 5.0)
    },
    "dissolved_oxygen": {
        "unit": "mg/L",
        "critical": (0, 4),
        "low": (4, 6),
        "adequate": (6, 8),
        "good": (8, 12)
    },
    "ph": {
        "unit": "",
        "acidic": (5.0, 6.5),
        "neutral": (6.5, 7.5),
        "alkaline": (7.5, 9.0)
    },
    "wind_speed": {
        "unit": "m/s",
        "calm": (0, 2),
        "light": (2, 4),
        "moderate": (4, 7),
        "strong": (7, 12)
    },
    "solar_radiation": {
        "unit": "W/m²",
        "low": (0, 200),
        "moderate": (200, 400),
        "high": (400, 600),
        "very_high": (600, 1000)
    }
}


# ============================================================
# LLM CONFIGURATION
# ============================================================
LLM_CONFIG = {
    "provider": "google",
    "model": "models/gemini-2.5-flash",
    "temperature": 0.1,
    "max_tokens": 2048,
    "top_p": 0.95,
    "api_key_env": "GEMINI_API_KEY"
}


# ============================================================
# AGENT SETTINGS
# ============================================================
AGENT_CONFIG = {
    "name": "PREDIKT",
    "version": "3.0",
    "max_iterations": 5,
    "max_execution_time": 60,  # seconds
    "chat_history_length": 20,  # number of messages to keep
    "verbose": False,
    "handle_parsing_errors": True
}


# ============================================================
# LOGGING CONFIGURATION
# ============================================================
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "standard",
            "stream": "ext://sys.stdout"
        },
        "file": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "standard",
            "filename": str(LOGS_DIR / "predikt.log"),
            "mode": "a"
        }
    },
    "loggers": {
        "": {
            "handlers": ["console", "file"],
            "level": "INFO",
            "propagate": True
        }
    }
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_lake_info(lake_name: str) -> Dict[str, Any]:
    """Get detailed information about a lake"""
    return GERMAN_LAKES.get(lake_name, {})


def get_risk_level_info(probability: float) -> Dict[str, Any]:
    """Get risk level classification info"""
    for level, info in RISK_THRESHOLDS.items():
        if info["min"] <= probability < info["max"]:
            return {"level": level, **info}
    return {"level": "unknown", "color": "gray", "emoji": "⚪"}


def get_forecast_horizon_info(days: int) -> Dict[str, Any]:
    """Get forecast horizon information"""
    return FORECAST_HORIZONS.get(days, {})


def validate_coordinates(lat: float, lon: float) -> bool:
    """Validate geographic coordinates"""
    return -90 <= lat <= 90 and -180 <= lon <= 180


def get_parameter_status(parameter: str, value: float) -> str:
    """Get human-readable status for a parameter value"""
    if parameter not in PARAMETER_RANGES:
        return "unknown"
    
    ranges = PARAMETER_RANGES[parameter]
    
    for status, (min_val, max_val) in ranges.items():
        if isinstance(status, str) and min_val <= value <= max_val:
            return status
    
    return "out of range"