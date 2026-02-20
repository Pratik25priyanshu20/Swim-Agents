# tests/conftest.py

import sys
from pathlib import Path

import pytest

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_location():
    return {"name": "Bodensee", "latitude": 47.5923, "longitude": 9.1881}


@pytest.fixture
def sample_location_custom():
    return {"name": "custom_location", "latitude": 48.0, "longitude": 11.0}
