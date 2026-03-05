from django.urls import path
from .views import ArbovirusRiskView

urlpatterns = [
    path("api/v1/risk/arbovirus/", ArbovirusRiskView.as_view(), name="arbovirus-risk"),
]