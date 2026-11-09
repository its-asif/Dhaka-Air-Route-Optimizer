from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),
    path('research/', include('route_optimizer.research.urls')),
    path('', include('route_optimizer.urls')),
]

