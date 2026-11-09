import json
import numpy as np
from django.test import TestCase, Client
from django.urls import reverse

from route_optimizer.research.data_adapters.synthetic import MultiDistrictDataGenerator
from route_optimizer.research.data_adapters.waqi import WaqiAdapter
from route_optimizer.research.data_adapters.open_meteo import OpenMeteoAdapter
from route_optimizer.research.models import MODEL_REGISTRY, get_model

from route_optimizer.research.xai.shap_explainer import AQISHAPExplainer
from route_optimizer.research.xai.feature_attribution import IntegratedGradientsExplainer
from route_optimizer.research.xai.attention_viz import AttentionVisualizer
from route_optimizer.research.routing.health_optimizer import MultiRouteHealthOptimizer
from route_optimizer.research.experiments.evaluator import ExperimentEvaluator

class DataAdapterTests(TestCase):
    def test_synthetic_multi_district_generator(self):
        districts = MultiDistrictDataGenerator.DISTRICTS
        self.assertEqual(len(districts), 5)
        df_dhaka = MultiDistrictDataGenerator.generate_district_data("Dhaka", days=7)
        self.assertEqual(len(df_dhaka), 7 * 24)
        self.assertIn("aqi", df_dhaka.columns)
        self.assertIn("pm25", df_dhaka.columns)

    def test_waqi_and_openmeteo_adapters(self):
        waqi = WaqiAdapter().fetch_latest_measurements()
        self.assertIn("aqi", waqi)
        
        meteo = OpenMeteoAdapter().fetch_weather()
        self.assertIn("temperature", meteo)

class DeepLearningModelTests(TestCase):
    def test_model_registry_and_predict(self):
        for name in MODEL_REGISTRY.keys():
            model = get_model(name)
            self.assertEqual(model.model_name, name)
            X = np.random.rand(5, 24, 10).astype(np.float32)
            y = np.random.rand(5, 1).astype(np.float32)
            res = model.fit(X, y, epochs=1)
            self.assertTrue(model.is_trained)
            preds = model.predict(X)
            self.assertEqual(preds.shape, (5, 1))



class ExplainableAITests(TestCase):
    def test_xai_explainers(self):
        model = get_model("Attention-LSTM")
        X = np.random.rand(1, 24, 10).astype(np.float32)
        
        shap_res = AQISHAPExplainer(model).explain_instance(X)
        self.assertIn("feature_contributions", shap_res)

        ig_res = IntegratedGradientsExplainer(model).attribute_prediction(X)
        self.assertIn("group_breakdown", ig_res)

        attn_res = AttentionVisualizer(model).get_temporal_attention(X)
        self.assertIn("attention_scores", attn_res)

class HealthRoutingTests(TestCase):
    def test_multi_route_health_optimizer(self):
        optimizer = MultiRouteHealthOptimizer()
        metrics = optimizer.calculate_health_metrics(distance_km=5.0, duration_mins=15.0, avg_aqi=120.0)
        self.assertIn("health_score", metrics)
        self.assertIn("inhalation_dosage", metrics)

class ResearchAPITests(TestCase):
    def setUp(self):
        self.client = Client()

    def test_dashboard_renders(self):
        response = self.client.get("/research/dashboard/")
        self.assertEqual(response.status_code, 200)

    def test_list_models_api(self):
        response = self.client.get("/research/api/models/")
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data["success"])
        self.assertGreaterEqual(len(data["models"]), 5)

    def test_predict_api(self):
        response = self.client.post(
            "/research/api/predict/",
            data=json.dumps({"model_name": "Attention-LSTM"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data["success"])
        self.assertIn("predicted_aqi", data)

    def test_explain_api(self):
        response = self.client.post(
            "/research/api/explain/",
            data=json.dumps({"model_name": "Attention-LSTM"}),
            content_type="application/json"
        )
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data["success"])
