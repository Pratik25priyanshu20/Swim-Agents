from .openweather_extractor import OpenWeatherExtractor

class HOMOGENDataCollector:
    def __init__(self, api_key, location):
        self.extractor = OpenWeatherExtractor(api_key, location)

    def collect(self):
        """
        Orchestrate data gathering from various sources.
        """
        data = self.extractor.extract()
        # Additional data collection logic can be added here
        return data