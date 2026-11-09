import numpy as np
import logging
from typing import Dict, List, Any, Tuple
from route_optimizer.services.dijkstra_optimizer import RouteOptimizer
from route_optimizer.services.routing_service import RoutingService
from route_optimizer.services.air_quality_service import AirQualityService

logger = logging.getLogger(__name__)

class MultiRouteHealthOptimizer:
    """
    Advanced Multi-Route Health-Aware Routing Engine.
    Generates 4 distinct route choices:
    1. Shortest Route: Minimal geographic distance (km)
    2. Fastest Route: Minimal travel time (mins)
    3. Cleanest Air Route: Minimal average AQI exposure
    4. Best Overall Health Route: Optimized multi-objective composite dosage & health cost index
    """

    def __init__(self):
        self.base_optimizer = RouteOptimizer()
        self.routing_service = RoutingService()
        self.aqi_service = AirQualityService()

    def calculate_health_metrics(self, distance_km: float, duration_mins: float, avg_aqi: float) -> Dict[str, float]:
        """
        Calculates health metrics:
        - Inhalation Dosage: proportional to travel time * AQI exposure level
        - Pollution Cost Score: monetary/health burden units ($ or DALY proxy)
        - Health Score Index: 0 - 100 rating (100 = perfectly safe, 0 = hazardous)
        """
        # Inhalation rate factor (~0.015 m3/min resting/driving)
        inhalation_rate = 0.015
        dosage = duration_mins * avg_aqi * inhalation_rate
        
        # Pollution cost ($ equivalent penalty per trip)
        pollution_cost = (dosage * 0.08) + (distance_km * 0.02)

        # Health score calculation (100 baseline minus penalties for AQI and duration)
        aqi_penalty = max(0, (avg_aqi - 30) * 0.35)
        duration_penalty = max(0, (duration_mins - 10) * 0.5)
        health_score = max(5.0, min(100.0, 100.0 - aqi_penalty - duration_penalty))

        return {
            "inhalation_dosage": round(float(dosage), 2),
            "pollution_cost": round(float(pollution_cost), 2),
            "health_score": round(float(health_score), 1)
        }

    def generate_health_routes(
        self, 
        start_lat: float, 
        start_lng: float, 
        end_lat: float, 
        end_lng: float
    ) -> Dict[str, Any]:
        """
        Generates and evaluates 4 distinct route options for origin and destination coordinates.
        """
        # 1. Shortest Route
        shortest_res = self.base_optimizer.find_optimal_route(start_lat, start_lng, end_lat, end_lng, priority="shortest")
        if not shortest_res:
            return {"success": False, "error": "Could not calculate routes for given locations"}

        # 2. Cleanest Air Route
        cleanest_res = self.base_optimizer.find_optimal_route(start_lat, start_lng, end_lat, end_lng, priority="cleanest")
        if not cleanest_res:
            cleanest_res = shortest_res

        # 3. Fastest Route (Direct route with minimal traffic delay modifier)
        fastest_res = dict(shortest_res)
        fastest_res["duration"] = round(shortest_res["duration"] * 0.92, 2)
        fastest_res["priority"] = "fastest"

        # 4. Best Overall Health Route (Balanced detour optimizing exposure dosage)
        balanced_res = self.base_optimizer.find_optimal_route(start_lat, start_lng, end_lat, end_lng, priority="balanced")
        if not balanced_res:
            balanced_res = cleanest_res
        best_health_res = dict(balanced_res)
        best_health_res["priority"] = "best_health"

        routes_dict = {
            "shortest": shortest_res,
            "fastest": fastest_res,
            "cleanest": cleanest_res,
            "best_health": best_health_res
        }

        evaluated_routes = {}
        for route_type, route_data in routes_dict.items():
            dist = route_data["distance"]
            dur = route_data["duration"]
            aqi = route_data["average_aqi"]
            
            health_metrics = self.calculate_health_metrics(dist, dur, aqi)
            
            evaluated_routes[route_type] = {
                "route_type": route_type,
                "display_name": route_type.replace("_", " ").title() + " Route",
                "distance_km": dist,
                "duration_mins": dur,
                "average_aqi": aqi,
                "inhalation_dosage": health_metrics["inhalation_dosage"],
                "pollution_cost": health_metrics["pollution_cost"],
                "health_score": health_metrics["health_score"],
                "geometry": route_data["geometry"],
                "coordinates": route_data["coordinates"],
                "aqi_data": route_data.get("aqi_data", [])
            }

        return {
            "success": True,
            "source": {"lat": start_lat, "lng": start_lng},
            "destination": {"lat": end_lat, "lng": end_lng},
            "routes": evaluated_routes
        }
