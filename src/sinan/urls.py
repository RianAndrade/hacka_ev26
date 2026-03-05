# sinan/urls.py
from django.urls import path
from sinan.api.views import SinanNotificationCSVImportView

urlpatterns = [
    path("sinan/notifications/import-csv/", SinanNotificationCSVImportView.as_view()),
]