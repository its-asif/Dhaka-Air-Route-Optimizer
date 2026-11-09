import os
import pickle
import logging
from datetime import datetime
import numpy as np
from django.conf import settings

logger = logging.getLogger(__name__)

MODEL_PATH = os.path.join(settings.BASE_DIR, 'training', 'aqi_prediction_model.pkl')
SCALER_PATH = os.path.join(settings.BASE_DIR, 'training', 'feature_scaler.pkl')


class MLPredictionService:
    """
    Service to load offline trained ML model and scaler to predict AQI values
    based on pollutant concentrations and temporal/environmental features.
    """

    def __init__(self):
        self.model = None
        self.scaler = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Load trained scikit-learn model and StandardScaler pickles."""
        try:
            if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
                with open(MODEL_PATH, 'rb') as f:
                    self.model = pickle.load(f)
                with open(SCALER_PATH, 'rb') as f:
                    self.scaler = pickle.load(f)
                logger.info("Successfully loaded ML model and scaler from training directory.")
            else:
                logger.warning("ML model or scaler pickle files not found at expected paths.")
        except Exception as e:
            logger.error("Failed to load ML artifacts: %s", e)
            self.model = None
            self.scaler = None

    def predict_aqi(
        self,
        pm25: float = 0.0,
        pm10: float = 0.0,
        co: float = 0.0,
        no2: float = 0.0,
        o3: float = 0.0,
        temperature: float = 30.0,
        humidity: float = 65.0,
        wind_speed: float = 10.0,
        hour: int = None,
        day_of_week: int = None
    ) -> float:
        """
        Predict AQI score using the 10-feature trained ML model:
        [pm25, pm10, co, no2, o3, temperature, humidity, wind_speed, hour, day_of_week]
        """
        if hour is None or day_of_week is None:
            now = datetime.now()
            if hour is None:
                hour = now.hour
            if day_of_week is None:
                day_of_week = now.weekday()

        features = np.array([[
            float(pm25 or 0.0),
            float(pm10 or 0.0),
            float(co or 0.0),
            float(no2 or 0.0),
            float(o3 or 0.0),
            float(temperature),
            float(humidity),
            float(wind_speed),
            float(hour),
            float(day_of_week)
        ]])

        if self.model and self.scaler:
            try:
                scaled_features = self.scaler.transform(features)
                prediction = float(self.model.predict(scaled_features)[0])
                return round(max(0.0, prediction), 2)
            except Exception as e:
                logger.warning("ML prediction failed: %s. Using heuristic fallback.", e)

        # Fallback heuristic calculation if model unavailable or prediction fails
        weighted = (
            (pm25 or 0.0) * 0.35 +
            (pm10 or 0.0) * 0.25 +
            (no2 or 0.0) * 0.20 +
            (co or 0.0) * 0.10 +
            (o3 or 0.0) * 0.10
        )
        return round(weighted if weighted > 0 else 100.0, 2)
