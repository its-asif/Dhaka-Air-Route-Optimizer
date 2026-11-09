# Data adapters package initialization
from .synthetic import MultiDistrictDataGenerator
from .waqi import WaqiAdapter
from .open_meteo import OpenMeteoAdapter

__all__ = ["MultiDistrictDataGenerator", "WaqiAdapter", "OpenMeteoAdapter"]
