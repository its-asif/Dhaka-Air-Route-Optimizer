from django.test import TestCase, Client
from django.urls import reverse
import json

from .models import Location, RouteHistory
from .services.dijkstra_optimizer import RouteOptimizer, DijkstraOptimizer
from .services.air_quality_service import AirQualityService
from .services.ml_prediction_service import MLPredictionService


class ModelTests(TestCase):
    """Tests for database models."""

    def test_location_creation(self):
        """Test creating a Location instance."""
        location = Location.objects.create(
            name="Park Street",
            latitude=22.5511,
            longitude=88.3527,
            overall_aqi=85.0
        )
        self.assertEqual(str(location), "Park Street")
        self.assertEqual(location.latitude, 22.5511)
        self.assertEqual(location.longitude, 88.3527)
        self.assertEqual(location.overall_aqi, 85.0)

    def test_route_history_creation(self):
        """Test creating a RouteHistory instance."""
        history = RouteHistory.objects.create(
            source_name="Park Street",
            source_lat=22.5511,
            source_lng=88.3527,
            destination_name="Victoria Memorial",
            destination_lat=22.5448,
            destination_lng=88.3426,
            priority="balanced",
            total_distance=3.5,
            estimated_time=12.0,
            average_aqi=92.0,
            route_geometry={"type": "LineString", "coordinates": []}
        )
        self.assertIn("Park Street", str(history))
        self.assertIn("Victoria Memorial", str(history))
        self.assertEqual(history.priority, "balanced")

    def test_route_history_ordering(self):
        """Test that route history is ordered by most recent first."""
        RouteHistory.objects.create(
            source_name="A", source_lat=0, source_lng=0,
            destination_name="B", destination_lat=1, destination_lng=1,
            priority="shortest", total_distance=1, estimated_time=5,
            average_aqi=50, route_geometry={}
        )
        RouteHistory.objects.create(
            source_name="C", source_lat=0, source_lng=0,
            destination_name="D", destination_lat=1, destination_lng=1,
            priority="cleanest", total_distance=2, estimated_time=10,
            average_aqi=30, route_geometry={}
        )
        histories = RouteHistory.objects.all()
        # Most recent (second created) should come first
        self.assertEqual(histories[0].source_name, "C")


class ViewTests(TestCase):
    """Tests for view endpoints."""

    def setUp(self):
        self.client = Client()

    def test_landing_page_loads(self):
        """Test that the landing home page renders successfully."""
        response = self.client.get(reverse('route_optimizer:landing'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dhaka AQI Optimizer')

    def test_dashboard_page_loads(self):
        """Test that the dashboard page renders successfully."""
        response = self.client.get(reverse('route_optimizer:dashboard'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dhaka AQI Route Optimizer')

    def test_index_page_loads(self):
        """Test that index legacy route resolves and renders successfully."""
        response = self.client.get(reverse('route_optimizer:index'))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, 'Dhaka AQI Route Optimizer')

    def test_dashboard_contains_map(self):
        """Test that dashboard page contains the map div."""
        response = self.client.get(reverse('route_optimizer:dashboard'))
        self.assertContains(response, 'id="map"')

    def test_find_route_requires_post(self):
        """Test that find_route rejects GET requests."""
        response = self.client.get(reverse('route_optimizer:find_route'))
        self.assertEqual(response.status_code, 405)

    def test_find_route_requires_valid_json(self):
        """Test that find_route rejects invalid JSON."""
        response = self.client.post(
            reverse('route_optimizer:find_route'),
            data='not-json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertFalse(data['success'])

    def test_find_route_requires_addresses(self):
        """Test that find_route validates required fields."""
        response = self.client.post(
            reverse('route_optimizer:find_route'),
            data=json.dumps({}),
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)

    def test_get_history_returns_json(self):
        """Test that history endpoint returns valid JSON."""
        response = self.client.get(reverse('route_optimizer:get_history'))
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.content)
        self.assertTrue(data['success'])
        self.assertIsInstance(data['history'], list)

    def test_get_history_limits_results(self):
        """Test that history returns at most 20 records."""
        for i in range(25):
            RouteHistory.objects.create(
                source_name=f"S{i}", source_lat=0, source_lng=0,
                destination_name=f"D{i}", destination_lat=1, destination_lng=1,
                priority="balanced", total_distance=1, estimated_time=5,
                average_aqi=50, route_geometry={}
            )
        response = self.client.get(reverse('route_optimizer:get_history'))
        data = json.loads(response.content)
        self.assertLessEqual(len(data['history']), 20)

    def test_get_aqi_requires_post(self):
        """Test that get_aqi rejects GET requests."""
        response = self.client.get(reverse('route_optimizer:get_aqi'))
        self.assertEqual(response.status_code, 405)

    def test_compare_routes_requires_post(self):
        """Test that compare_routes rejects GET requests."""
        response = self.client.get(reverse('route_optimizer:compare_routes'))
        self.assertEqual(response.status_code, 405)

    def test_compare_routes_requires_valid_json(self):
        """Test that compare_routes validates request body."""
        response = self.client.post(
            reverse('route_optimizer:compare_routes'),
            data='not-json',
            content_type='application/json'
        )
        self.assertEqual(response.status_code, 400)
        data = json.loads(response.content)
        self.assertFalse(data['success'])


class ServiceTests(TestCase):
    """Tests for service layer."""

    def test_ml_prediction_service(self):
        """Test MLPredictionService loads artifacts and computes predictions."""
        service = MLPredictionService()
        predicted = service.predict_aqi(pm25=120, pm10=140, co=5, no2=25, o3=15)
        self.assertIsInstance(predicted, float)
        self.assertGreater(predicted, 0.0)

    def test_route_optimizer_backward_compat(self):
        """Test that DijkstraOptimizer alias still works."""
        optimizer = DijkstraOptimizer()
        self.assertIsInstance(optimizer, RouteOptimizer)

    def test_aqi_service_parse_data(self):
        """Test AQI data parsing."""
        service = AirQualityService()
        raw_data = {
            'aqi': 75,
            'iaqi': {
                'pm25': {'v': 42},
                'pm10': {'v': 65},
                'no2': {'v': 18},
                'co': {'v': 3.2},
                'o3': {'v': 28}
            },
            'city': {
                'geo': [22.57, 88.36],
                'name': 'Kolkata'
            },
            'time': {'s': '2024-01-15 10:00:00'}
        }
        parsed = service._parse_aqi_data(raw_data)

        self.assertEqual(parsed['aqi'], 75)
        self.assertEqual(parsed['pm25'], 42)
        self.assertEqual(parsed['pm10'], 65)
        self.assertEqual(parsed['location']['name'], 'Kolkata')
        self.assertEqual(parsed['location']['lat'], 22.57)

    def test_weighted_aqi_calculation(self):
        """Test weighted AQI calculation with all pollutants present."""
        service = AirQualityService()
        aqi_data = {
            'aqi': 100,
            'pm25': 50,
            'pm10': 60,
            'no2': 30,
            'co': 10,
            'o3': 20
        }
        weighted = service.calculate_weighted_aqi(aqi_data)
        # All pollutants present, so total_weight = 1.0
        expected = (50 * 0.35 + 60 * 0.25 + 30 * 0.20 + 10 * 0.10 + 20 * 0.10)
        self.assertAlmostEqual(weighted, expected, places=2)

    def test_weighted_aqi_with_missing_data(self):
        """Test weighted AQI falls back to overall AQI when no pollutants."""
        service = AirQualityService()
        aqi_data = {'aqi': 120, 'pm25': 0, 'pm10': 0, 'no2': 0, 'co': 0, 'o3': 0}
        weighted = service.calculate_weighted_aqi(aqi_data)
        self.assertEqual(weighted, 120)  # Falls back to aqi field

    def test_route_optimizer_bearing_calculation(self):
        """Test bearing calculation between two points."""
        optimizer = RouteOptimizer()
        # North: lat increases, lng same
        bearing = optimizer._calculate_bearing(22.0, 88.0, 23.0, 88.0)
        self.assertAlmostEqual(bearing, 0.0, delta=1.0)

        # East: lat same, lng increases
        bearing = optimizer._calculate_bearing(22.0, 88.0, 22.0, 89.0)
        self.assertAlmostEqual(bearing, 90.0, delta=2.0)

    def test_route_optimizer_destination_point(self):
        """Test destination point calculation returns reasonable coordinates."""
        optimizer = RouteOptimizer()
        lat, lng = optimizer._calculate_destination_point(22.57, 88.36, 1.0, 0)
        # 1km north should increase lat slightly
        self.assertGreater(lat, 22.57)
        self.assertAlmostEqual(lng, 88.36, delta=0.01)


class URLTests(TestCase):
    """Tests for URL configuration."""

    def test_urls_resolve(self):
        """Test that all named URLs resolve correctly."""
        self.assertEqual(reverse('route_optimizer:landing'), '/')
        self.assertEqual(reverse('route_optimizer:dashboard'), '/dashboard/')
        self.assertEqual(reverse('route_optimizer:index'), '/index/')
        self.assertEqual(reverse('route_optimizer:find_route'), '/api/find-route/')
        self.assertEqual(reverse('route_optimizer:compare_routes'), '/api/compare-routes/')
        self.assertEqual(reverse('route_optimizer:get_history'), '/api/history/')
        self.assertEqual(reverse('route_optimizer:get_aqi'), '/api/get-aqi/')
        self.assertEqual(reverse('route_optimizer:geocode_autocomplete'), '/api/autocomplete/')

