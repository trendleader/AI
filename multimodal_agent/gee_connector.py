import json
from datetime import datetime, timedelta
from typing import Optional

import numpy as np

try:
    import ee
    GEE_AVAILABLE = True
except ImportError:
    GEE_AVAILABLE = False


class GEEConnector:
    """Google Earth Engine connector for real-time satellite vegetation data."""

    SATELLITE_CONFIGS = {
        "sentinel2": {
            "collection": "COPERNICUS/S2_SR_HARMONIZED",
            "nir": "B8",
            "red": "B4",
            "green": "B3",
            "swir": "B11",
            "cloud_filter": ("CLOUDY_PIXEL_PERCENTAGE", 20),
            "scale": 10,
        },
        "landsat": {
            "collection": "LANDSAT/LC08/C02/T1_L2",
            "nir": "SR_B5",
            "red": "SR_B4",
            "green": "SR_B3",
            "swir": "SR_B6",
            "cloud_filter": ("CLOUD_COVER", 20),
            "scale": 30,
        },
        "modis": {
            "collection": "MODIS/006/MOD13Q1",
            "nir": None,
            "red": None,
            "green": None,
            "swir": None,
            "cloud_filter": None,
            "scale": 250,
        },
    }

    def __init__(self):
        self.connected = False
        self.project: Optional[str] = None

    def connect(self, project_id: str, service_account: Optional[str] = None) -> bool:
        if not GEE_AVAILABLE:
            return False
        try:
            if service_account:
                credentials = ee.ServiceAccountCredentials(service_account, key_file=None)
                ee.Initialize(credentials, project=project_id)
            else:
                ee.Initialize(project=project_id)
            self.connected = True
            self.project = project_id
            return True
        except Exception as e:
            print(f"GEE connection error: {e}")
            return False

    def get_vegetation_indices(
        self,
        lat: float,
        lon: float,
        start_date: str,
        end_date: str,
        satellite: str = "sentinel2",
    ) -> dict:
        if not self.connected or not GEE_AVAILABLE:
            return self.get_demo_data(lat, lon)

        cfg = self.SATELLITE_CONFIGS.get(satellite, self.SATELLITE_CONFIGS["sentinel2"])

        try:
            point = ee.Geometry.Point([lon, lat])
            region = point.buffer(1000)

            collection = ee.ImageCollection(cfg["collection"]).filterBounds(region).filterDate(
                start_date, end_date
            )
            if cfg["cloud_filter"]:
                collection = collection.filter(
                    ee.Filter.lt(cfg["cloud_filter"][0], cfg["cloud_filter"][1])
                )
            collection = collection.sort("system:time_start", False)

            latest = collection.first()
            nir, red, green = cfg["nir"], cfg["red"], cfg["green"]
            scale = cfg["scale"]

            ndvi_img = latest.normalizedDifference([nir, red]).rename("NDVI")
            ndwi_img = latest.normalizedDifference([green, nir]).rename("NDWI")

            ndvi_val = ndvi_img.sample(point, scale).first().get("NDVI").getInfo()
            ndwi_val = ndwi_img.sample(point, scale).first().get("NDWI").getInfo()

            # Time series
            def add_ndvi(img):
                nd = img.normalizedDifference([nir, red])
                return nd.set("system:time_start", img.get("system:time_start"))

            ts = collection.map(add_ndvi).getRegion(point, scale).getInfo()
            dates, ndvi_ts, ndwi_ts = [], [], []
            for row in ts[1:]:
                if row[4] is not None:
                    dates.append(
                        datetime.fromtimestamp(row[3] / 1000).strftime("%Y-%m-%d")
                    )
                    ndvi_ts.append(round(float(row[4]), 4))

            anomalies = self._flag_anomalies(ndvi_val, ndwi_val)

            return {
                "ndvi": round(float(ndvi_val), 4),
                "ndwi": round(float(ndwi_val), 4),
                "time_series": {"dates": dates, "ndvi_values": ndvi_ts},
                "anomalies": anomalies,
                "source": satellite,
                "timestamp": datetime.now().isoformat(),
            }

        except Exception as e:
            print(f"GEE query error: {e}")
            return self.get_demo_data(lat, lon)

    def get_demo_data(self, lat: float, lon: float) -> dict:
        """Realistic demo data when GEE credentials are unavailable."""
        rng = np.random.default_rng(int(abs(lat * 100 + lon * 100)) % 9999)
        base_ndvi = float(np.clip(rng.normal(0.55, 0.12), 0.05, 0.95))
        base_ndwi = float(np.clip(rng.normal(0.05, 0.18), -0.8, 0.6))

        dates, ndvi_ts, ndwi_ts = [], [], []
        for weeks_ago in range(24, 0, -1):
            d = datetime.now() - timedelta(weeks=weeks_ago)
            dates.append(d.strftime("%Y-%m-%d"))
            seasonal = 0.08 * np.sin(2 * np.pi * weeks_ago / 26)
            ndvi_ts.append(round(float(np.clip(base_ndvi + seasonal + rng.normal(0, 0.02), 0, 1)), 3))
            ndwi_ts.append(round(float(np.clip(base_ndwi + rng.normal(0, 0.04), -1, 1)), 3))

        return {
            "ndvi": round(base_ndvi, 3),
            "ndwi": round(base_ndwi, 3),
            "time_series": {"dates": dates, "ndvi_values": ndvi_ts, "ndwi_values": ndwi_ts},
            "anomalies": self._flag_anomalies(base_ndvi, base_ndwi),
            "source": "demo",
            "timestamp": datetime.now().isoformat(),
        }

    def _flag_anomalies(self, ndvi: float, ndwi: float) -> list:
        anomalies = []
        if ndvi < 0.2:
            anomalies.append({"description": f"Critically low vegetation (NDVI {ndvi:.3f})", "severity": "CRITICAL"})
        elif ndvi < 0.35:
            anomalies.append({"description": f"Below-average vegetation health (NDVI {ndvi:.3f})", "severity": "HIGH"})
        elif ndvi < 0.5:
            anomalies.append({"description": f"Moderate vegetation stress (NDVI {ndvi:.3f})", "severity": "MEDIUM"})

        if ndwi < -0.3:
            anomalies.append({"description": f"Severe water stress (NDWI {ndwi:.3f})", "severity": "HIGH"})
        elif ndwi < -0.1:
            anomalies.append({"description": f"Mild water stress detected (NDWI {ndwi:.3f})", "severity": "MEDIUM"})
        return anomalies
