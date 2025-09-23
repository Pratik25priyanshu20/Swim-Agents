# tests/test_harmonized_output.py

import pandas as pd
from pathlib import Path
import pprint

project_root = Path(__file__).resolve().parents[1]
harmonized_dir = project_root / "data/harmonized"
files = list(harmonized_dir.glob("*.parquet"))

summary = {}

for file in files:
    df = pd.read_parquet(file)
    source_name = file.stem
    summary[source_name] = {
        "record_count": len(df),
        "missing_geometry": df["geometry"].isna().sum() if "geometry" in df else "N/A",
        "missing_latitude": df["latitude"].isna().sum() if "latitude" in df else "N/A",
        "missing_longitude": df["longitude"].isna().sum() if "longitude" in df else "N/A",
        "missing_station_id": df["station_id"].isna().sum() if "station_id" in df else "N/A",
        "validation_flags": df["validation_flags"].value_counts().to_dict() if "validation_flags" in df else {},
        "quality_score_summary": df["quality_score"].describe().to_dict() if "quality_score" in df else "N/A",
        "columns": df.columns.tolist(),
        "date_range": (str(df["collection_date"].min()), str(df["collection_date"].max())) if "collection_date" in df else ("N/A", "N/A"),
    }

print("\n📊 Harmonized Data Summary:")
pprint.pprint(summary, sort_dicts=False)