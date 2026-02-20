# swim/data_processing/etl/sentinel_extractor.py

"""Extract water quality indices from Sentinel-2 satellite imagery via Google Earth Engine."""

import logging
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from swim.data_processing.etl.base_extractor import BaseDataExtractor

logger = logging.getLogger(__name__)


class SentinelExtractor(BaseDataExtractor):
    """
    Extracts Sentinel-2 L2A surface reflectance data and derives water quality indices.

    Indices computed:
    - NDWI (Normalized Difference Water Index): (Green - NIR) / (Green + NIR)
    - NDCI (Normalized Difference Chlorophyll Index): (RE1 - Red) / (RE1 + Red)
    - Turbidity proxy: Red / Green ratio
    """

    BANDS = {
        "B2": "Blue",
        "B3": "Green",
        "B4": "Red",
        "B5": "RedEdge1",
        "B8": "NIR",
        "B11": "SWIR1",
    }

    def __init__(self):
        self._ee = None
        self._initialized = False

    def _init_ee(self):
        if self._initialized:
            return
        try:
            import ee
            if os.getenv("GOOGLE_APPLICATION_CREDENTIALS") or os.getenv("EE_PROJECT"):
                ee.Initialize(project=os.getenv("EE_PROJECT", ""))
            else:
                ee.Authenticate()
                ee.Initialize()
            self._ee = ee
            self._initialized = True
            logger.info("Earth Engine initialized successfully")
        except Exception as exc:
            logger.warning("Earth Engine not available: %s. Using synthetic fallback.", exc)
            self._ee = None

    def extract(
        self,
        lat: float,
        lon: float,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        buffer_m: int = 500,
    ) -> List[Dict[str, Any]]:
        """Extract Sentinel-2 water quality data for a point location."""
        self._init_ee()

        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

        if self._ee is None:
            return self._synthetic_data(lat, lon, start_date, end_date)

        return self._extract_ee(lat, lon, start_date, end_date, buffer_m)

    def _extract_ee(
        self, lat: float, lon: float, start_date: str, end_date: str, buffer_m: int
    ) -> List[Dict[str, Any]]:
        ee = self._ee
        point = ee.Geometry.Point([lon, lat])
        aoi = point.buffer(buffer_m)

        collection = (
            ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED")
            .filterBounds(aoi)
            .filterDate(start_date, end_date)
            .filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 30))
            .sort("system:time_start", False)
            .limit(20)
        )

        def compute_indices(image):
            green = image.select("B3").divide(10000)
            red = image.select("B4").divide(10000)
            re1 = image.select("B5").divide(10000)
            nir = image.select("B8").divide(10000)

            ndwi = green.subtract(nir).divide(green.add(nir)).rename("NDWI")
            ndci = re1.subtract(red).divide(re1.add(red)).rename("NDCI")
            turbidity = red.divide(green.add(1e-6)).rename("turbidity_ratio")

            return image.addBands([ndwi, ndci, turbidity])

        processed = collection.map(compute_indices)

        records = []
        try:
            features = processed.getRegion(point, 10).getInfo()
            if features and len(features) > 1:
                headers = features[0]
                for row in features[1:]:
                    entry = dict(zip(headers, row))
                    records.append({
                        "latitude": lat,
                        "longitude": lon,
                        "acquisition_date": datetime.utcfromtimestamp(
                            entry.get("time", 0) / 1000
                        ).isoformat(),
                        "ndwi": entry.get("NDWI"),
                        "ndci": entry.get("NDCI"),
                        "turbidity_ratio": entry.get("turbidity_ratio"),
                        "chlorophyll_index": self._ndci_to_chl(entry.get("NDCI", 0)),
                        "cloud_coverage": entry.get("CLOUDY_PIXEL_PERCENTAGE", 0),
                        "satellite_name": "Sentinel-2",
                        "sensor_type": "MSI",
                    })
        except Exception as exc:
            logger.error("Failed to extract Sentinel-2 data: %s", exc)

        return records

    @staticmethod
    def _ndci_to_chl(ndci: float) -> float:
        """Approximate chlorophyll-a (µg/L) from NDCI using empirical formula."""
        if ndci is None:
            return 0.0
        # Empirical: Chl-a ≈ 14.039 + 86.115 * NDCI + 194.325 * NDCI²
        return max(0.0, 14.039 + 86.115 * ndci + 194.325 * ndci**2)

    def _synthetic_data(
        self, lat: float, lon: float, start_date: str, end_date: str
    ) -> List[Dict[str, Any]]:
        """Generate synthetic Sentinel-2-like data when EE is unavailable."""
        import random
        records = []
        start = datetime.fromisoformat(start_date)
        end = datetime.fromisoformat(end_date)
        days = (end - start).days
        for i in range(0, days, 5):  # ~5-day revisit
            date = start + timedelta(days=i)
            ndci = random.gauss(0.1, 0.08)
            records.append({
                "latitude": lat,
                "longitude": lon,
                "acquisition_date": date.isoformat(),
                "ndwi": round(random.gauss(0.3, 0.1), 4),
                "ndci": round(ndci, 4),
                "turbidity_ratio": round(random.gauss(0.8, 0.15), 4),
                "chlorophyll_index": round(self._ndci_to_chl(ndci), 2),
                "cloud_coverage": round(random.uniform(0, 30), 1),
                "satellite_name": "Sentinel-2 (synthetic)",
                "sensor_type": "MSI",
            })
        return records
