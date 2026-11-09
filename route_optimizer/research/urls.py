from django.urls import path
from . import views

app_name = "research"

urlpatterns = [
    path("dashboard/", views.dashboard_view, name="dashboard"),
    path("api/models/", views.list_models_api, name="api_models"),
    path("api/predict/", views.predict_api, name="api_predict"),
    path("api/explain/", views.explain_api, name="api_explain"),
    path("api/routes/", views.health_routes_api, name="api_health_routes"),
    path("api/experiments/leaderboard/", views.leaderboard_api, name="api_leaderboard"),
]
