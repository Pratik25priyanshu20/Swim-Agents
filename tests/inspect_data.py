import pandas as pd
from pathlib import Path

# Load harmonized data
samples_path = Path("data/harmonized/samples.parquet")

if not samples_path.exists():
    print("❌ samples.parquet not found.")
else:
    df = pd.read_parquet(samples_path)
    print(f"✅ Loaded {len(df)} rows.")

    # Check potential name columns
    print("\n🔍 Non-null unique values in 'station_name':")
    print(df['station_name'].dropna().unique().tolist())

    print("\n🔍 Non-null unique values in 'station_id':")
    print(df['station_id'].dropna().unique().tolist())

    print("\n🔍 Available measurement parameters:")
    print(df['measurement_parameter'].dropna().unique().tolist())