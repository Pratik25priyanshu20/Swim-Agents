import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from swim.data_processing.etl.homogen_collector import HOMOGENDataCollector
from swim.data_processing.cleaning.harmonize_weather import harmonize_openweather_data

# Placeholder if bulk_insert doesn't exist yet
try:
    from db.ingest.bulk_insert import bulk_insert_records
except ImportError:
    def bulk_insert_records(records):
        print("bulk_insert_records called with", len(records), "records")

def main():
    # Test lake location and API key for demo purposes
    api_key = "YOUR_OPENWEATHER_API_KEY"
    location = {
        "latitude": 47.5,
        "longitude": 8.5,
        "lake_id": "test_lake_001"
    }

    collector = HOMOGENDataCollector(api_key, location)
    raw_data = collector.collect()

    harmonized_data = harmonize_openweather_data(raw_data)

    bulk_insert_records(harmonized_data)
    print("✅ HOMOGEN ingestion pipeline completed.")

if __name__ == "__main__":
    main()