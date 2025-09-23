import ee
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --------------------
# Initialize GEE
# --------------------
ee.Initialize()

# --------------------
# Load lake sample data
# --------------------
parquet_path = Path("data/harmonized/lake_samples.parquet")
if not parquet_path.exists():
    raise FileNotFoundError("lake_samples.parquet not found.")

df = pd.read_parquet(parquet_path)

# Clean and filter required columns
df = df.dropna(subset=['lake', 'date', 'latitude', 'longitude'])
df['date'] = pd.to_datetime(df['date'])

# Sentinel-2 Bands
bands = ['B2', 'B3', 'B4', 'B5', 'B6', 'B8', 'B8A', 'B11']

# --------------------
# Function to extract reflectance values
# --------------------
def extract_sentinel2_reflectance(lat, lon, date):
    point = ee.Geometry.Point([lon, lat])
    start_date = date.strftime('%Y-%m-%d')
    end_date = (date + pd.Timedelta(days=3)).strftime('%Y-%m-%d')

    collection = ee.ImageCollection("COPERNICUS/S2_SR") \
        .filterBounds(point) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))

    image = collection.first()
    if image is None:
        return {band: None for band in bands}

    try:
        values = image.reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=point,
            scale=20,
            maxPixels=1e9
        ).getInfo()
        return {band: values.get(band, None) for band in bands}
    except Exception:
        return {band: None for band in bands}

# --------------------
# Loop through each lake sample
# --------------------
data = []

print("\n🔍 Extracting reflectance values from Sentinel-2...")
for _, row in tqdm(df.iterrows(), total=len(df)):
    result = extract_sentinel2_reflectance(row['latitude'], row['longitude'], row['date'])
    result.update({
        'lake': row['lake'],
        'date': row['date']
    })
    data.append(result)

# --------------------
# Save output
# --------------------
out_path = Path("data/eo/eo_lake_reflectance.csv")
out_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(data).to_csv(out_path, index=False)
print(f"✅ EO reflectance data saved to: {out_path}")