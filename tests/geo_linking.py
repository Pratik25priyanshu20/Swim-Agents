import pandas as pd
import requests
import time

# Load your CSV file
df = pd.read_csv("/Users/futurediary/Desktop/A.I. Agents/ERAY_HEIDELBERG/tests/Messwerte_Grundwasserguete.csv", sep=';')

# Extract unique Gemeinde
gemeinden = df['Gemeinde'].dropna().unique()

# Geocode function
def geocode(gemeinde):
    url = "https://nominatim.openstreetmap.org/search"
    params = {
        "q": f"{gemeinde}, Germany",
        "format": "json",
        "limit": 1
    }
    headers = {"User-Agent": "GeoApp/1.0"}
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200 and response.json():
        result = response.json()[0]
        return gemeinde, float(result['lat']), float(result['lon'])
    else:
        return gemeinde, None, None

# Run geocoding with 1-second pause
results = []
for g in gemeinden:
    print(f"Geocoding: {g}")
    results.append(geocode(g))
    time.sleep(1)

# Save results
geo_df = pd.DataFrame(results, columns=["Gemeinde", "Latitude", "Longitude"])
geo_df.to_csv("gemeinde_geocoded.csv", index=False)
print("✅ Geocoding complete! Saved as gemeinde_geocoded.csv")