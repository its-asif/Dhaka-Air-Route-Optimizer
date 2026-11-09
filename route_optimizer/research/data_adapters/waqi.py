import logging
from typing import Dict, Optional
from route_optimizer.services.air_quality_service import AirQualityService

logger = logging.getLogger(__name__)

class WaqiAdapter:
    """
    WAQI REST API adapter for fetching real air quality measurements.
    Replaces OpenAQ for consistency with the main routing application.
    Schema compliance: aqi, pm25, pm10, no2, co, o3.
    """
    
    def __init__(self):
        self.service = AirQualityService()

    def fetch_latest_measurements(self, lat: float = 23.8103, lng: float = 90.4125) -> Dict:
        """
        Fetch latest air quality measurements near coordinates.
        """
        try:
            data = self.service.get_aqi_by_coordinates(lat, lng)
            if data:
                return {
                    "source": "WAQI",
                    "station_name": data.get("location", {}).get("name", "Unknown Station"),
                    "lat": lat,
                    "lng": lng,
                    "aqi": data.get("aqi", 0),
                    "pm25": data.get("pm25", 0),
                    "pm10": data.get("pm10", 0),
                    "no2": data.get("no2", 0),
                    "co": data.get("co", 0),
                    "o3": data.get("o3", 0),
                    "status": "Success"
                }
        except Exception as e:
            logger.error("Error fetching from WAQI API: %s", e)
            
        # Return structured fallback if API call fails
        return {
            "source": "WAQI (Fallback)",
            "lat": lat,
            "lng": lng,
            "aqi": 145.0,
            "pm25": 58.5,
            "pm10": 92.0,
            "no2": 24.1,
            "co": 1.2,
            "o3": 18.4,
            "status": "Fallback / Simulated"
        }
