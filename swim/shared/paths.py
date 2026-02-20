# swim/shared/paths.py

"""Centralized path definitions for the SWIM platform."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
HARMONIZED_DIR = DATA_DIR / "harmonized"
PROCESSED_DIR = DATA_DIR / "processed"
INTERMEDIATE_DIR = DATA_DIR / "intermediate"
SATELLITE_DIR = RAW_DIR / "satellite"
VISIOS_IMAGE_DIR = DATA_DIR / "visios_images"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"
