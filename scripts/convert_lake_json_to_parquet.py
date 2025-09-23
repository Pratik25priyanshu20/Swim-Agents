import json
import pandas as pd
from pathlib import Path

# --------------------------
# Paths
# --------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data/raw/lake_data"
OUT_PARQUET = PROJECT_ROOT / "data/harmonized/lake_quality.parquet"

# --------------------------
# Load all JSON files
# --------------------------
json_files = sorted([f for f in RAW_DIR.glob("*.json") if f.is_file()])
print(f"🔍 Found {len(json_files)} files: {[f.name for f in json_files]}")

# --------------------------
# Flatten each JSON entry
# --------------------------
records = []

for file in json_files:
    with open(file) as f:
        data = json.load(f)

    for entry in data:
        flat = {
            "date": entry.get("date"),
            "lake": entry.get("lake"),
            "source": entry.get("source"),
            "quality_score": entry.get("quality_score"),
            "data_version": entry.get("data_version"),
            "year": entry.get("year"),
            "week": entry.get("week"),
            "day": entry.get("day"),
        }

        # Flatten parameters
        params = entry.get("parameters", {})
        for key, value in params.items():
            flat[f"param_{key}"] = value

        # Flatten HABs indicators
        habs = entry.get("habs_indicators", {})
        for key, value in habs.items():
            flat[f"habs_{key}"] = value

        records.append(flat)

# --------------------------
# Convert to DataFrame and Save
# --------------------------
df = pd.DataFrame(records)

# Show a few rows before saving
print("\n📌 Preview of flattened data:")
print(df.head(3))

# Save to Parquet
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(OUT_PARQUET, index=False, engine="pyarrow")

print(f"\n✅ lake_quality.parquet saved at: {OUT_PARQUET.resolve()}")
print(f"📦 Total records: {len(df)}")