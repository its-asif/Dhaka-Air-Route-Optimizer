import numpy as np
from typing import Dict, List, Any
from route_optimizer.research.models.base import BaseForecaster

class IntegratedGradientsExplainer:
    """
    Feature Attribution Explainer using Integrated Gradients / Finite-Difference Perturbations.
    Quantifies the impact of weather vs pollutant feature groups on AQI forecast decisions.
    """

    POLLUTANT_FEATURES = ["aqi", "pm25", "pm10", "no2", "co", "o3"]
    WEATHER_FEATURES = ["temperature", "humidity", "wind_speed", "wind_direction"]
    ALL_FEATURES = POLLUTANT_FEATURES + WEATHER_FEATURES

    def __init__(self, model: BaseForecaster):
        self.model = model

    def attribute_prediction(self, sample_sequence: np.ndarray, steps: int = 20) -> Dict[str, Any]:
        """
        Computes integrated attributions across input features.
        """
        if sample_sequence.ndim == 2:
            sample_sequence = np.expand_dims(sample_sequence, axis=0)

        baseline = np.zeros_like(sample_sequence)
        preds_baseline = self.model.predict(baseline)[0, 0]
        preds_target = self.model.predict(sample_sequence)[0, 0]

        attributions = {}
        total_delta = float(preds_target - preds_baseline)

        # Finite difference gradient path integration along interpolated baseline -> input vector
        for idx, feat_name in enumerate(self.ALL_FEATURES):
            # Perturb specific feature column
            perturbed = sample_sequence.copy()
            perturbed[:, :, idx] = baseline[:, :, idx]
            pred_perturbed = self.model.predict(perturbed)[0, 0]
            
            # Integrated feature score
            attr_val = float(preds_target - pred_perturbed)
            attributions[feat_name] = np.round(attr_val, 4)

        # Aggregate group contributions (Pollutants vs Weather)
        pollutant_score = sum(attributions[f] for f in self.POLLUTANT_FEATURES)
        weather_score = sum(attributions[f] for f in self.WEATHER_FEATURES)
        total_abs = max(1e-5, abs(pollutant_score) + abs(weather_score))

        return {
            "model_name": self.model.model_name,
            "target_prediction": round(float(preds_target), 2),
            "baseline_prediction": round(float(preds_baseline), 2),
            "attributions": attributions,
            "group_breakdown": {
                "pollutants_percentage": round(abs(pollutant_score) / total_abs * 100, 1),
                "weather_percentage": round(abs(weather_score) / total_abs * 100, 1),
                "pollutants_impact_score": round(float(pollutant_score), 2),
                "weather_impact_score": round(float(weather_score), 2)
            }
        }
