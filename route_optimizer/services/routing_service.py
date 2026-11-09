import openrouteservice
from django.conf import settings
from typing import List, Dict, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


DHAKA_PRESET_LOCATIONS = [
    {"label": "Dhanmondi, Dhaka", "lat": 23.7461, "lng": 90.3742},
    {"label": "Dhanmondi 32, Dhaka", "lat": 23.7516, "lng": 90.3781},
    {"label": "Dhanmondi 27, Dhaka", "lat": 23.7547, "lng": 90.3753},
    {"label": "Dhanmondi Lake, Dhaka", "lat": 23.7470, "lng": 90.3790},
    {"label": "Dhanmondi Green Road, Dhaka", "lat": 23.7490, "lng": 90.3860},
    {"label": "Uttara, Dhaka", "lat": 23.8720, "lng": 90.3980},
    {"label": "Uttara Sector 1, Dhaka", "lat": 23.8690, "lng": 90.3980},
    {"label": "Uttara Sector 3, Dhaka", "lat": 23.8680, "lng": 90.3930},
    {"label": "Uttara Sector 7, Dhaka", "lat": 23.8740, "lng": 90.3960},
    {"label": "Uttara House Building, Dhaka", "lat": 23.8755, "lng": 90.3995},
    {"label": "Uttara Diabari, Dhaka", "lat": 23.8820, "lng": 90.3790},
    {"label": "Mirpur 1, Dhaka", "lat": 23.7950, "lng": 90.3530},
    {"label": "Mirpur 10, Dhaka", "lat": 23.8069, "lng": 90.3687},
    {"label": "Mirpur 11, Dhaka", "lat": 23.8160, "lng": 90.3660},
    {"label": "Mirpur 12, Dhaka", "lat": 23.8240, "lng": 90.3640},
    {"label": "Mirpur DOHS, Dhaka", "lat": 23.8340, "lng": 90.3690},
    {"label": "Gulshan 1, Dhaka", "lat": 23.7780, "lng": 90.4160},
    {"label": "Gulshan 2, Dhaka", "lat": 23.7930, "lng": 90.4140},
    {"label": "Banani, Dhaka", "lat": 23.7937, "lng": 90.4047},
    {"label": "Banani Road 11, Dhaka", "lat": 23.7920, "lng": 90.4070},
    {"label": "Mohakhali, Dhaka", "lat": 23.7778, "lng": 90.4055},
    {"label": "Mohakhali DOHS, Dhaka", "lat": 23.7820, "lng": 90.3970},
    {"label": "Farmgate, Dhaka", "lat": 23.7561, "lng": 90.3872},
    {"label": "Khamarbari, Farmgate, Dhaka", "lat": 23.7580, "lng": 90.3840},
    {"label": "Shahbagh, Dhaka", "lat": 23.7388, "lng": 90.3960},
    {"label": "Dhaka University, Shahbagh, Dhaka", "lat": 23.7330, "lng": 90.3930},
    {"label": "Motijheel, Dhaka", "lat": 23.7330, "lng": 90.4170},
    {"label": "Shapla Chattar, Motijheel, Dhaka", "lat": 23.7265, "lng": 90.4190},
    {"label": "Jatrabari, Dhaka", "lat": 23.7075, "lng": 90.4320},
    {"label": "Agargaon, Dhaka", "lat": 23.7770, "lng": 90.3800},
    {"label": "IDB Bhaban, Agargaon, Dhaka", "lat": 23.7760, "lng": 90.3820},
    {"label": "Badda, Dhaka", "lat": 23.7805, "lng": 90.4260},
    {"label": "Merul Badda, Dhaka", "lat": 23.7690, "lng": 90.4230},
    {"label": "Rampura, Dhaka", "lat": 23.7610, "lng": 90.4200},
    {"label": "Moghbazar, Dhaka", "lat": 23.7490, "lng": 90.4070},
    {"label": "Malibagh, Dhaka", "lat": 23.7470, "lng": 90.4150},
    {"label": "Mohammadpur, Dhaka", "lat": 23.7658, "lng": 90.3622},
    {"label": "Mohammadpur Town Hall, Dhaka", "lat": 23.7590, "lng": 90.3640},
    {"label": "Shyamoli, Dhaka", "lat": 23.7725, "lng": 90.3635},
    {"label": "Kalyanpur, Dhaka", "lat": 23.7800, "lng": 90.3600},
    {"label": "Gabtoli Bus Terminal, Dhaka", "lat": 23.7840, "lng": 90.3450},
    {"label": "Sayedabad Bus Terminal, Dhaka", "lat": 23.7120, "lng": 90.4280},
    {"label": "Sadarghat Launch Terminal, Dhaka", "lat": 23.7050, "lng": 90.4100},
    {"label": "Azimpur, Dhaka", "lat": 23.7270, "lng": 90.3860},
    {"label": "New Market, Dhaka", "lat": 23.7330, "lng": 90.3840},
    {"label": "Lalbagh Fort, Dhaka", "lat": 23.7188, "lng": 90.3881},
    {"label": "Puran Dhaka (Old Dhaka), Dhaka", "lat": 23.7100, "lng": 90.4000},
    {"label": "Chawkbazar, Old Dhaka", "lat": 23.7140, "lng": 90.3970},
    {"label": "Hazrat Shahjalal International Airport, Dhaka", "lat": 23.8399, "lng": 90.4041},
    {"label": "Bashundhara Residential Area, Dhaka", "lat": 23.8150, "lng": 90.4320},
    {"label": "Nikunja 1, Dhaka", "lat": 23.8300, "lng": 90.4170},
    {"label": "Nikunja 2, Dhaka", "lat": 23.8330, "lng": 90.4190},
    {"label": "Baridhara Diplomatic Zone, Dhaka", "lat": 23.7990, "lng": 90.4200},
    {"label": "Tejgaon Industrial Area, Dhaka", "lat": 23.7650, "lng": 90.3980},
    {"label": "Kawran Bazar, Dhaka", "lat": 23.7510, "lng": 90.3930},
    {"label": "Panthapath, Dhaka", "lat": 23.7515, "lng": 90.3875},
    {"label": "Square Hospital, Panthapath, Dhaka", "lat": 23.7525, "lng": 90.3850},
    {"label": "Bashundhara City Shopping Complex, Panthapath", "lat": 23.7505, "lng": 90.3895},
    {"label": "Elephant Road, Dhaka", "lat": 23.7380, "lng": 90.3870},
    {"label": "Bailey Road, Dhaka", "lat": 23.7420, "lng": 90.4050},
    {"label": "Kakrail, Dhaka", "lat": 23.7370, "lng": 90.4080},
    {"label": "Purana Paltan, Dhaka", "lat": 23.7310, "lng": 90.4100},
    {"label": "Khilgaon, Dhaka", "lat": 23.7520, "lng": 90.4260},
    {"label": "Keraniganj, Dhaka", "lat": 23.6800, "lng": 90.3700},
]


class RoutingService:
    """Service to handle routing using OpenRouteService"""
    
    def __init__(self):
        self.client = openrouteservice.Client(key=settings.ORS_API_KEY)
    
    def get_route(self, start_coords: Tuple[float, float], 
                  end_coords: Tuple[float, float], 
                  profile: str = 'driving-car') -> Optional[Dict]:
        """
        Get route between two coordinates.
        start_coords, end_coords: (longitude, latitude)
        """
        coords = [start_coords, end_coords]
        try:
            route = self.client.directions(
                coordinates=coords,
                profile=profile,
                format='geojson',
                instructions=True,
                elevation=False,
                radiuses=[-1, -1]  # -1 allows unlimited search radius to snap points (e.g. airports/parks) to nearest road
            )
            
            return self._parse_route_data(route)
        except Exception as e:
            logger.warning("First attempt getting route failed (%s), retrying without explicit radiuses...", e)
            try:
                route = self.client.directions(
                    coordinates=coords,
                    profile=profile,
                    format='geojson',
                    instructions=True,
                    elevation=False
                )
                return self._parse_route_data(route)
            except Exception as fallback_error:
                logger.error("Error getting route: %s", fallback_error)
                return None
    
    def get_alternative_routes(self, start_coords: Tuple[float, float],
                              end_coords: Tuple[float, float],
                              profile: str = 'driving-car') -> List[Dict]:
        """
        Get multiple alternative routes using OpenRouteService.
        Returns up to 3 different routes.
        """
        try:
            coords = [start_coords, end_coords]
            
            # Request with alternative routes parameter
            route = self.client.directions(
                coordinates=coords,
                profile=profile,
                format='geojson',
                instructions=True,
                elevation=False,
                alternative_routes={
                    'target_count': 2,  # Request 2 alternatives (total 3 routes)
                    'weight_factor': 1.4,  # How different routes should be
                    'share_factor': 0.6  # How much routes can overlap
                }
            )
            
            # OpenRouteService returns multiple routes in features array
            routes = []
            if 'features' in route:
                for feature in route['features']:
                    parsed = self._parse_single_feature(feature)
                    if parsed:
                        routes.append(parsed)
            
            return routes if routes else [self.get_route(start_coords, end_coords)]
            
        except Exception as e:
            logger.info("Alternative routes not available, using standard route: %s", e)
            # Fallback to standard route
            standard = self.get_route(start_coords, end_coords)
            return [standard] if standard else []
    
    def get_route_with_waypoints(self, coordinates: List[Tuple[float, float]], 
                                  profile: str = 'driving-car', 
                                  priority: str = 'balanced') -> Optional[Dict]:
        """
        Get route with strategic waypoints based on priority.
        """
        try:
            # For shortest, use direct route
            if priority == 'shortest':
                route = self.client.directions(
                    coordinates=[coordinates[0], coordinates[-1]],
                    profile=profile,
                    format='geojson',
                    instructions=True,
                    elevation=False,
                    radiuses=[-1, -1]
                )
                return self._parse_route_data(route)
            
            # For cleanest/pollutant routes, use waypoints with preference parameters
            elif priority in ['cleanest', 'pm25', 'pm10', 'co', 'o3', 'so2']:
                # Simplify waypoints - too many waypoints force same route
                selected_coords = self._select_strategic_waypoints(
                    coordinates, 
                    max_waypoints=4,  # Limit to 4 intermediate waypoints
                    priority=priority
                )
                
                route = self.client.directions(
                    coordinates=selected_coords,
                    profile=profile,
                    format='geojson',
                    instructions=True,
                    elevation=False,
                    radiuses=[-1] * len(selected_coords),
                    options={
                        'avoid_features': []  # Can add 'highways' to avoid major roads
                    }
                )
                return self._parse_route_data(route)
            
            # For balanced
            else:
                selected_coords = self._select_strategic_waypoints(
                    coordinates, 
                    max_waypoints=3,
                    priority=priority
                )
                
                route = self.client.directions(
                    coordinates=selected_coords,
                    profile=profile,
                    format='geojson',
                    instructions=True,
                    elevation=False,
                    radiuses=[-1] * len(selected_coords)
                )
                return self._parse_route_data(route)
                
        except Exception as e:
            logger.warning("Error with waypoints: %s", e)
            # Fallback to direct route
            try:
                route = self.client.directions(
                    coordinates=[coordinates[0], coordinates[-1]],
                    profile=profile,
                    format='geojson',
                    instructions=True,
                    elevation=False,
                    radiuses=[-1, -1]
                )
                return self._parse_route_data(route)
            except Exception as fallback_error:
                logger.error("Fallback route also failed: %s", fallback_error)
                return None
    
    def _select_strategic_waypoints(self, coordinates: List[Tuple[float, float]],
                                   max_waypoints: int = 4,
                                   priority: str = 'balanced') -> List[Tuple[float, float]]:
        """
        Select most strategic waypoints to create different routes.
        """
        if len(coordinates) <= max_waypoints + 2:
            return coordinates
        
        # Always include start and end
        selected = [coordinates[0]]
        
        # For cleanest routes, prefer waypoints furthest from direct line
        if priority in ['cleanest', 'pm25', 'pm10', 'co', 'o3', 'so2']:
            # Calculate deviation from direct line for each waypoint
            start = np.array(coordinates[0])
            end = np.array(coordinates[-1])
            
            deviations = []
            for i in range(1, len(coordinates) - 1):
                point = np.array(coordinates[i])
                # Calculate perpendicular distance from line
                deviation = self._point_line_distance(point, start, end)
                deviations.append((i, deviation))
            
            # Sort by deviation (largest first) and select top waypoints
            deviations.sort(key=lambda x: x[1], reverse=True)
            selected_indices = sorted([idx for idx, _ in deviations[:max_waypoints]])
            
            for idx in selected_indices:
                selected.append(coordinates[idx])
        
        # For balanced, evenly distribute waypoints
        else:
            step = (len(coordinates) - 2) / (max_waypoints + 1)
            for i in range(1, max_waypoints + 1):
                idx = int(i * step)
                selected.append(coordinates[idx])
        
        # Add end point
        selected.append(coordinates[-1])
        
        return selected
    
    def _point_line_distance(self, point: np.ndarray, 
                            line_start: np.ndarray, 
                            line_end: np.ndarray) -> float:
        """Calculate perpendicular distance from point to line."""
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Normalize line vector
        line_len = np.linalg.norm(line_vec)
        if line_len == 0:
            return np.linalg.norm(point_vec)
        
        line_unit = line_vec / line_len
        
        # Project point onto line
        projection_length = np.dot(point_vec, line_unit)
        projection = projection_length * line_unit
        
        # Distance is perpendicular component
        perpendicular = point_vec - projection
        return np.linalg.norm(perpendicular)
    
    def _parse_single_feature(self, feature: Dict) -> Optional[Dict]:
        """Parse a single route feature."""
        try:
            properties = feature['properties']
            geometry = feature['geometry']
            
            return {
                'distance': round(properties.get('summary', {}).get('distance', 0) / 1000, 2),
                'duration': round(properties.get('summary', {}).get('duration', 0) / 60, 2),
                'coordinates': geometry['coordinates'],
                'steps': properties.get('segments', [{}])[0].get('steps', []),
                'bbox': properties.get('bbox', []),
                'geometry': geometry
            }
        except Exception as e:
            logger.error("Error parsing feature: %s", e)
            return None
    
    def get_isochrones(self, location: Tuple[float, float], 
                       range_seconds: List[int] = [600, 1200, 1800]) -> Optional[Dict]:
        """Get isochrones (reachability areas) from a location."""
        try:
            isochrones = self.client.isochrones(
                locations=[location],
                profile='driving-car',
                range=range_seconds,
                range_type='time'
            )
            return isochrones
        except Exception as e:
            logger.error("Error getting isochrones: %s", e)
            return None



    def geocode_address(self, address: str) -> Optional[Tuple[float, float]]:
        """
        Convert address to coordinates using Pelias geocoding.
        Returns: (longitude, latitude) or None
        """
        try:
            result = self.client.pelias_search(text=address, focus_point=[90.4125, 23.8103])  # Dhaka focus
            if result and 'features' in result and len(result['features']) > 0:
                coords = result['features'][0]['geometry']['coordinates']
                return tuple(coords)  # (lng, lat)
            return None
        except Exception as e:
            logger.error("Error geocoding address: %s", e)
            return None

    def autocomplete_address(self, query: str, limit: int = 6) -> List[Dict]:
        """
        Get real-time address suggestions for typing using Dhaka preset locations
        combined with Pelias geocoding search.
        Returns list of dicts: [{'label': name, 'lat': lat, 'lng': lng}]
        """
        clean_q = query.strip().lower()
        if not clean_q:
            return []
        
        suggestions = []
        seen_labels = set()

        # 1. Search local preset locations catalog
        for loc in DHAKA_PRESET_LOCATIONS:
            if clean_q in loc['label'].lower():
                suggestions.append(loc)
                seen_labels.add(loc['label'].lower())
                if len(suggestions) >= limit:
                    return suggestions

        # 2. Query live OpenRouteService Pelias API for additional hits
        try:
            search_query = query if 'dhaka' in clean_q else f"{query}, Dhaka"
            result = self.client.pelias_search(
                text=search_query,
                focus_point=[90.4125, 23.8103],
                size=limit
            )
            if result and 'features' in result:
                for feature in result['features']:
                    props = feature.get('properties', {})
                    coords = feature.get('geometry', {}).get('coordinates', [])
                    label = props.get('label') or props.get('name')
                    if label and len(coords) == 2:
                        lbl_lower = label.lower()
                        if lbl_lower not in seen_labels:
                            suggestions.append({
                                'label': label,
                                'lng': coords[0],
                                'lat': coords[1]
                            })
                            seen_labels.add(lbl_lower)
                            if len(suggestions) >= limit:
                                break
        except Exception as e:
            logger.warning("Pelias live autocomplete fallback warning: %s", e)
        
        return suggestions
    
    def reverse_geocode(self, lng: float, lat: float) -> Optional[str]:
        """Convert coordinates to address."""
        try:
            result = self.client.pelias_reverse(point=(lng, lat))
            if result and 'features' in result and len(result['features']) > 0:
                return result['features'][0]['properties'].get('label', 'Unknown')
            return None
        except Exception as e:
            logger.error("Error reverse geocoding: %s", e)
            return None
    
    def _parse_route_data(self, route_geojson: Dict) -> Optional[Dict]:
        """Parse route GeoJSON data."""
        if not route_geojson or 'features' not in route_geojson:
            return None
        
        feature = route_geojson['features'][0]
        return self._parse_single_feature(feature)
    
    def sample_route_points(self, coordinates: List[List[float]], 
                           num_samples: int = 10) -> List[Tuple[float, float]]:
        """
        Sample points along a route for AQI checking.
        Returns: List of (lat, lng) tuples
        """
        if len(coordinates) <= num_samples:
            return [(coord[1], coord[0]) for coord in coordinates]
        
        indices = np.linspace(0, len(coordinates) - 1, num_samples, dtype=int)
        sampled = [coordinates[i] for i in indices]
        
        return [(coord[1], coord[0]) for coord in sampled]

    def get_route_via_waypoint(self, start_coords: Tuple[float, float],
                           waypoint_coords: Tuple[float, float],
                           end_coords: Tuple[float, float],
                           profile: str = 'driving-car') -> Optional[Dict]:
        """
        Get route through a single waypoint.
        All coordinates: (longitude, latitude)
        """
        try:
            coords = [start_coords, waypoint_coords, end_coords]
            
            route = self.client.directions(
                coordinates=coords,
                profile=profile,
                format='geojson',
                instructions=True,
                elevation=False,
                radiuses=[-1, -1, -1]  # -1 allows unlimited search radius to snap all points to nearest road
            )
            
            return self._parse_route_data(route)
            
        except Exception as e:
            logger.warning("Waypoint routing failed: %s", e)
            return None

