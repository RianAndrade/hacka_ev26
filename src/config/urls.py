from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("admin/", admin.site.urls),
    path("api/inmet/", include("inmet.urls")),
    path("api/arbovirus/", include("arbovirus.urls")),
    path("api/sih/", include("sih.urls")),
]