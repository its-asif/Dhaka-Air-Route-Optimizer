from django.urls import path, include
from . import views

app_name = 'route_optimizer'

urlpatterns = [
    path('', views.landing, name='landing'),
    path('dashboard/', views.index, name='dashboard'),
    path('index/', views.index, name='index'),
    path('live-report/', views.live_report, name='live_report'),
    path('api/find-route/', views.find_route, name='find_route'),
    path('api/compare-routes/', views.compare_routes, name='compare_routes'),
    path('api/history/', views.get_history, name='get_history'),
    path('api/get-aqi/', views.get_aqi, name='get_aqi'),
    path('api/autocomplete/', views.geocode_autocomplete, name='geocode_autocomplete'),
]


