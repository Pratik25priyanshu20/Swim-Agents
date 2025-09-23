import requests
from .base_extractor import BaseDataExtractor

class OpenWeatherExtractor(BaseDataExtractor):
    def __init__(self, api_key, location):
        self.api_key = api_key
        self.location = location

    def extract(self):
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "lat": self.location["latitude"],
            "lon": self.location["longitude"],
            "appid": self.api_key,
            "units": "metric"
        }

        response = requests.get(url, params=params)
        data = response.json()

        record = {
            "location_name": "Lake Constance",
            "latitude": self.location["latitude"],
            "longitude": self.location["longitude"],
            "timestamp": data.get("dt"),
            "chlorophyll_a": None,  # OpenWeather doesn't provide
            "turbidity": None,
            "dissolved_oxygen": None,
            "ph": None,
            "conductivity": None,
            "redox_potential": None,
            "source": "OpenWeatherMap",
            "confidence_score": 0.85
        }

        # Optionally map real fields like temp, humidity, etc.
        return [record]