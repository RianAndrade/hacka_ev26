from django.urls import path

from .views import (
    HospitalAdmissionCsvImportView,
    RunHospitalOccupancyForecastView,
    HospitalOccupancyForecastStatusView,
    HospitalOccupancyForecastView,
)

urlpatterns = [
    path("hospital-admissions/import-csv/", HospitalAdmissionCsvImportView.as_view()),
    path("hospital-occupancy/forecast/run/", RunHospitalOccupancyForecastView.as_view()),
    path("hospital-occupancy/forecast/status/<str:task_id>/", HospitalOccupancyForecastStatusView.as_view()),
    path("hospital-occupancy/forecast/", HospitalOccupancyForecastView.as_view()),
]