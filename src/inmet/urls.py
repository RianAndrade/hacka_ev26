from django.urls import path

from inmet.api.views.inmet_upload_csv import InmetWeatherObservationCSVImportView

urlpatterns = [
    path(
        "observations/import-csv/",
        InmetWeatherObservationCSVImportView.as_view(),
        name="inmet-observations-import-csv",
    ),
]