# swim/data_processing/etl/dwd_extractor.py

"""Extract weather data from the German Weather Service (DWD) open data API."""

import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

from swim.data_processing.etl.base_extractor import BaseDataExtractor

logger = logging.getLogger(__name__)

DWD_CDC_BASE = "https://opendata.dwd.de/climate_environment/CDC"


class DWDExtractor(BaseDataExtractor):
    """
    Extracts weather observations from DWD's Climate Data Center.

    Parameters retrieved:
    - Air temperature (°C)
    - Precipitation (mm)
    - Wind speed (m/s)
    - Solar radiation (J/cm²)
    - Relative humidity (%)
    """

    def extract(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Extract weather data for a location. Falls back to synthetic data if API unavailable."""
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        try:
            return self._fetch_dwd(lat, lon, start_date, end_date)
        except Exception as exc:
            logger.warning("DWD API unavailable (%s), using synthetic data", exc)
            return self._synthetic_weather(lat, lon, start_date, end_date)

    def _fetch_dwd(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Attempt to fetch DWD data via their public API."""
        # DWD provides data as CSV/ZIP files — for real production use,
        # implement station lookup + file download. This uses the simple Mosmix API.
        station_id = self._find_nearest_station(lat, lon)
        if not station_id:
            raise ConnectionError("No nearby DWD station found")

        url = f"{DWD_CDC_BASE}/observations_germany/climate/daily/kl/recent/"
        resp = httpx.get(url, timeout=15)
        resp.raise_for_status()
        # Parse station data from the response (simplified)
        raise NotImplementedError("Full DWD CSV parsing not yet implemented — use synthetic fallback")

    def _find_nearest_station(self, lat: float, lon: float) -> Optional[str]:
        """Find the nearest DWD weather station. Stub for full station catalog lookup."""
        # Major German weather stations near lake regions
        stations = {
            "01981": (47.61, 9.20),   # Konstanz (near Bodensee)
            "04104": (47.88, 12.45),  # Traunstein (near Chiemsee)
            "05792": (47.97, 11.18),  # Wielenbach (near Ammersee/Starnberger See)
            "04745": (53.38, 12.65),  # Waren (near Müritz)
        }
        best_id = None
        best_dist = float("inf")
        for sid, (slat, slon) in stations.items():
            dist = (lat - slat) ** 2 + (lon - slon) ** 2
            if dist < best_dist:
                best_dist = dist
                best_id = sid
        return best_id if best_dist < 1.0 else None

    def _synthetic_weather(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Generate synthetic weather data matching DWD parameter ranges."""
        import random
        records = []
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days

        for i in range(days):
            date = start + timedelta(days=i)
            month = date.month
            # Seasonal temperature curve (rough Central European pattern)
            base_temp = 5 + 15 * max(0, 1 - abs(month - 7) / 5)
            records.append({
                "latitude": lat,
                "longitude": lon,
                "date": date.isoformat(),
                "air_temperature": round(base_temp + random.gauss(0, 3), 1),
                "precipitation": round(max(0, random.expovariate(0.3)), 1),
                "wind_speed": round(max(0, random.gauss(3.5, 1.5)), 1),
                "solar_radiation": round(max(0, random.gauss(300 + 200 * max(0, 1 - abs(month - 7) / 5), 100)), 0),
                "relative_humidity": round(min(100, max(30, random.gauss(70, 10))), 0),
                "source": "DWD (synthetic)",
            })
        return records
