# scripts/flatten_lake_json.py

import json
import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw/lake_data")
OUT_PARQUET = Path("data/harmonized/lake_samples.parquet")

def flatten_json_record(record):
    flat = {
        "lake": record.get("lake"),
        "date": record.get("date"),
        "source": record.get("source"),
        "quality_score": record.get("quality_score"),
        "data_version": record.get("data_version"),
        "year": record.get("year"),
        "week": record.get("week"),
        "day": record.get("day"),
    }

    # Flatten parameters
    parameters = record.get("parameters", {})
    for key, val in parameters.items():
        flat[key] = val

    # Flatten HABs indicators
    habs = record.get("habs_indicators", {})
    for key, val in habs.items():
        flat[key] = val

    return flat

def load_and_flatten_all():
    records = []
    files = sorted(RAW_DIR.glob("*.json"))
    print(f"🔍 Found {len(files)} JSON files in {RAW_DIR.name}: {[f.name for f in files]}")
    
    for file in files:
        with open(file) as f:
            try:
                data = json.load(f)
                if isinstance(data, list):
                    for record in data:
                        records.append(flatten_json_record(record))
                elif isinstance(data, dict):
                    records.append(flatten_json_record(data))
            except Exception as e:
                print(f"❌ Error loading {file.name}: {e}")
    
    df = pd.DataFrame(records)
    print(f"✅ Parsed {len(df)} records from all JSON files.")

    if not df.empty:
        df.to_parquet(OUT_PARQUET, index=False)
        print(f"✅ Saved to: {OUT_PARQUET}")
    else:
        print("⚠️ No data to save.")

if __name__ == "__main__":
    load_and_flatten_all()