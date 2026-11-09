import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List

class MultiDistrictDataGenerator:
    """
    Generates synthetic multi-district hourly time-series data for Federated Learning:
    Districts: Dhaka, Gazipur, Narayanganj, Munshiganj, Faridpur.
    Schema: timestamp, aqi, pm25, pm10, no2, co, o3, temperature, humidity, wind_speed, wind_direction
    """

    DISTRICTS = ["Dhaka", "Gazipur", "Narayanganj", "Munshiganj", "Faridpur"]
    
    # Base pollution coefficients representing local industrial/urban characteristics
    DISTRICT_PROFILES = {
        "Dhaka": {"aqi_base": 160, "noise": 25, "pm25_ratio": 0.60, "industrial_factor": 1.2},
        "Gazipur": {"aqi_base": 185, "noise": 30, "pm25_ratio": 0.65, "industrial_factor": 1.4},
        "Narayanganj": {"aqi_base": 175, "noise": 28, "pm25_ratio": 0.63, "industrial_factor": 1.35},
        "Munshiganj": {"aqi_base": 120, "noise": 20, "pm25_ratio": 0.55, "industrial_factor": 0.9},
        "Faridpur": {"aqi_base": 95, "noise": 15, "pm25_ratio": 0.50, "industrial_factor": 0.75},
    }

    @classmethod
    def generate_district_data(
        cls, 
        district: str, 
        days: int = 90, 
        start_date: str = "2026-01-01 00:00:00",
        seed: int = 42
    ) -> pd.DataFrame:
        if district not in cls.DISTRICTS:
            raise ValueError(f"Unknown district: {district}. Valid choices: {cls.DISTRICTS}")

        profile = cls.DISTRICT_PROFILES[district]
        district_seed = seed + cls.DISTRICTS.index(district) * 100
        np.random.seed(district_seed)

        total_hours = days * 24
        start = pd.to_datetime(start_date)
        timestamps = [start + timedelta(hours=i) for i in range(total_hours)]

        hours = np.array([ts.hour for ts in timestamps])
        day_of_week = np.array([ts.dayofweek for ts in timestamps])

        # Diurnal pattern (rush hour peaks around 8-10 AM and 6-9 PM)
        diurnal = 20 * np.sin((hours - 7) * np.pi / 12) + 15 * np.sin((hours - 18) * np.pi / 6)
        weekly = 5 * np.cos(day_of_week * 2 * np.pi / 7)
        random_noise = np.random.normal(0, profile["noise"], total_hours)

        aqi = profile["aqi_base"] + diurnal + weekly + random_noise
        aqi = np.clip(aqi, 20, 500)

        # Pollutant synthesis derived from AQI and profile ratios
        pm25 = aqi * profile["pm25_ratio"] + np.random.normal(0, 5, total_hours)
        pm10 = aqi * 0.85 + np.random.normal(0, 8, total_hours)
        no2 = aqi * 0.25 * profile["industrial_factor"] + np.random.normal(0, 3, total_hours)
        co = aqi * 0.08 * profile["industrial_factor"] + np.random.normal(0, 1, total_hours)
        o3 = np.clip(40 + 15 * np.sin((hours - 14) * np.pi / 12) + np.random.normal(0, 4, total_hours), 5, 120)

        # Weather variables
        temperature = 25 + 6 * np.sin((hours - 12) * np.pi / 12) + np.random.normal(0, 1.5, total_hours)
        humidity = 65 - 15 * np.sin((hours - 12) * np.pi / 12) + np.random.normal(0, 3, total_hours)
        wind_speed = np.clip(2.5 + np.random.exponential(1.5, total_hours), 0.5, 15.0)
        wind_direction = (np.random.normal(180, 45, total_hours)) % 360

        df = pd.DataFrame({
            "timestamp": timestamps,
            "district": district,
            "aqi": np.round(aqi, 2),
            "pm25": np.round(np.clip(pm25, 5, 400), 2),
            "pm10": np.round(np.clip(pm10, 10, 500), 2),
            "no2": np.round(np.clip(no2, 2, 200), 2),
            "co": np.round(np.clip(co, 0.1, 50), 2),
            "o3": np.round(o3, 2),
            "temperature": np.round(temperature, 2),
            "humidity": np.round(np.clip(humidity, 20, 100), 2),
            "wind_speed": np.round(wind_speed, 2),
            "wind_direction": np.round(wind_direction, 2),
            "hour": hours,
            "day_of_week": day_of_week
        })

        return df

    @classmethod
    def generate_all_districts(cls, days: int = 90) -> Dict[str, pd.DataFrame]:
        return {d: cls.generate_district_data(d, days=days) for d in cls.DISTRICTS}
