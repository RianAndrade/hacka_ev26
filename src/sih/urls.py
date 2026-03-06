from django.urls import path

from .views import (
    HospitalAdmissionCsvImportView,
    RunHospitalOccupancyForecastView,
    HospitalOccupancyForecastStatusView,
    HospitalOccupancyForecastView,
    HospitalOccupancyPredictionByHospitalView,
    HospitalOccupancyAvailableHospitalsView,
    HospitalOccupancyDashboardView,
)

urlpatterns = [
    path(
        "hospital-admissions/import-csv/",
        HospitalAdmissionCsvImportView.as_view(),
        name="hospital-admissions-import-csv",
    ),
    path(
        "hospital-occupancy/forecast/run/",
        RunHospitalOccupancyForecastView.as_view(),
        name="hospital-occupancy-forecast-run",
    ),
    path(
        "hospital-occupancy/forecast/status/<str:task_id>/",
        HospitalOccupancyForecastStatusView.as_view(),
        name="hospital-occupancy-forecast-status",
    ),
    path(
        "hospital-occupancy/forecast/",
        HospitalOccupancyForecastView.as_view(),
        name="hospital-occupancy-forecast",
    ),
    path(
        "hospital-occupancy/predictions/",
        HospitalOccupancyPredictionByHospitalView.as_view(),
        name="hospital-occupancy-predictions",
    ),
    path(
        "hospital-occupancy/hospitals/",
        HospitalOccupancyAvailableHospitalsView.as_view(),
        name="hospital-occupancy-hospitals",
    ),
    path(
        "hospital-occupancy/dashboard/",
        HospitalOccupancyDashboardView.as_view(),
        name="hospital-occupancy-dashboard",
    ),
]