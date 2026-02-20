# swim/agents/calibro/utils/load_lake_data.py

import pandas as pd
from pathlib import Path

def find_lake_in_parquet(lake_name: str, file: Path):
    try:
        if file.exists():
            df = pd.read_parquet(file)
            if "lake" in df.columns:
                filtered = df[df["lake"].str.lower().str.strip() == lake_name.lower().strip()]
                if not filtered.empty:
                    return filtered, f"✅ Found lake in {file.name}"
    except Exception as e:
        return None, f"❌ Error reading {file.name}: {e}"
    return None, f"⚠️ No match in {file.name}"

def find_lake_in_satellite_csvs(lake_name: str, folder: Path):
    for file in folder.glob("*.csv"):
        if lake_name.lower() in file.stem.lower():
            try:
                df = pd.read_csv(file)
                if not df.empty:
                    return df, f"✅ Found satellite CSV: {file.name}"
            except Exception as e:
                return None, f"❌ Failed to read {file.name}: {e}"
    return None, f"⚠️ No satellite CSV found for {lake_name}"

def load_best_lake_data(lake_name: str, root_path: Path):
    # 1. Check harmonized parquet files
    harmonized_files = [
        root_path / "data/harmonized/lake_samples.parquet",
        root_path / "data/harmonized/unified_cube.parquet"
    ]
    
    for file in harmonized_files:
        df, msg = find_lake_in_parquet(lake_name, file)
        if df is not None:
            return df, msg

    # 2. Check processed satellite CSVs
    satellite_folder = root_path / "data/processed/satellite"
    df, msg = find_lake_in_satellite_csvs(lake_name, satellite_folder)
    if df is not None:
        return df, msg

    return None, f"❌ No data found for lake: {lake_name}"