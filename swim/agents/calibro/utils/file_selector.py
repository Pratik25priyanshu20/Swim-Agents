# swim/agents/calibro/utils/file_selector.py

from pathlib import Path
import pandas as pd
from typing import List, Optional

DATA_DIR = Path(__file__).parent.parent / "data"


def find_file_with_column(column_name: str) -> Optional[str]:
    """Return the first CSV file in data/ containing the given column."""
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=5)
            if column_name in df.columns:
                return csv_file.name
        except Exception:
            continue
    return None


def find_all_files_with_column(column_name: str) -> List[str]:
    """Return all files that contain a specific column."""
    matches = []
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=5)
            if column_name in df.columns:
                matches.append(csv_file.name)
        except Exception:
            continue
    return matches


def find_all_files_with_lake(lake_name: str) -> List[str]:
    """Return list of files that contain data for a given lake name."""
    matches = []
    for csv_file in DATA_DIR.glob("*.csv"):
        try:
            df = pd.read_csv(csv_file, nrows=100)
            if "lake_name" in df.columns and df["lake_name"].str.contains(lake_name, case=False, na=False).any():
                matches.append(csv_file.name)
        except Exception:
            continue
    return matches


def find_all_csv_files() -> List[str]:
    """Return all CSV files in the data/ folder."""
    return [f.name for f in DATA_DIR.glob("*.csv")]


def preview_csv(file_name: str) -> Optional[pd.DataFrame]:
    """Preview contents of a CSV (first 5 rows)."""
    file_path = DATA_DIR / file_name
    if not file_path.exists():
        return None
    try:
        return pd.read_csv(file_path).head(5)
    except Exception:
        return None