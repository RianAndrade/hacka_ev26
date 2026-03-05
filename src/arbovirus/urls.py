from django.urls import path
from .views import ArbovirusRiskView
from arbovirus.views import CeleryTestAPIView, TrainModelAPIView

urlpatterns = [
    path("api/v1/risk/arbovirus/", ArbovirusRiskView.as_view(), name="arbovirus-risk"),
    path("celery/test/", CeleryTestAPIView.as_view()),
    path("train-model/", TrainModelAPIView.as_view()),
]
