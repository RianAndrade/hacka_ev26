from datetime import date
from typing import List, Optional, Tuple

from django.db.models import Avg, Sum
from django.utils.dateparse import parse_date

from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.exceptions import ValidationError

from .models import SinanNotification, InmetWeatherObservation


OPTIMAL_MIN = 21.0
OPTIMAL_MAX = 30.0
EIP_MIN = 25.0
EIP_MAX = 28.0


def month_start(d: date) -> date:
    return d.replace(day=1)


def next_month(d: date) -> date:
    if d.month == 12:
        return d.replace(year=d.year + 1, month=1, day=1)
    return d.replace(month=d.month + 1, day=1)


def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def risk_label(score: float) -> str:
    if score >= 0.75:
        return "high"
    if score >= 0.45:
        return "moderate"
    return "low"


def suitability_from_temp(temp_mean: Optional[float]) -> float:
    if temp_mean is None:
        return 0.0
    peak = 26.0
    if temp_mean < OPTIMAL_MIN or temp_mean > OPTIMAL_MAX:
        return 0.0
    if temp_mean <= peak:
        return (temp_mean - OPTIMAL_MIN) / (peak - OPTIMAL_MIN)
    return (OPTIMAL_MAX - temp_mean) / (OPTIMAL_MAX - peak)


def heuristic_risk(
    temp_mean: Optional[float],
    humidity_mean: Optional[float],
    rain_sum: Optional[float],
    days_temp_21_30: int,
    days_temp_25_28: int,
) -> Tuple[float, List[str]]:
    explanations: List[str] = []

    temp_score = suitability_from_temp(temp_mean)
    if temp_mean is not None and OPTIMAL_MIN <= temp_mean <= OPTIMAL_MAX:
        explanations.append("Temperatura média ficou em faixa favorável (~21–30°C).")

    eip_boost = clamp01(days_temp_25_28 / 20.0)
    if days_temp_25_28 >= 10:
        explanations.append("Muitos dias entre 25–28°C, indicando janela favorável ao ciclo de transmissão.")

    hum_score = 0.0 if humidity_mean is None else clamp01((humidity_mean - 50.0) / 30.0)
    rain_score = 0.0 if rain_sum is None else clamp01(rain_sum / 200.0)
    band_score = clamp01(days_temp_21_30 / 25.0)

    score = (
        0.50 * temp_score
        + 0.20 * band_score
        + 0.15 * eip_boost
        + 0.10 * hum_score
        + 0.05 * rain_score
    )
    return clamp01(score), explanations


class ArbovirusRiskView(APIView):
    def get(self, request):
        municipality_code = request.query_params.get("municipality_code")
        health_region_code = request.query_params.get("health_region_code")
        disease = request.query_params.get("disease", "both").lower()
        reference_month = request.query_params.get("reference_month")

        if not reference_month:
            raise ValidationError({"reference_month": "Required (YYYY-MM)."})

        ref_date = parse_date(f"{reference_month}-01")
        if not ref_date:
            raise ValidationError({"reference_month": "Invalid format. Use YYYY-MM."})

        if not (municipality_code or health_region_code):
            raise ValidationError("Provide municipality_code or health_region_code.")

        if disease not in ("dengue", "chikungunya", "both"):
            raise ValidationError({"disease": "Use dengue, chikungunya, or both."})

        ref_month_start = month_start(ref_date)
        target_month_start = next_month(ref_month_start)

        climate_qs = InmetWeatherObservation.objects.filter(
            observation_date__gte=ref_month_start,
            observation_date__lt=target_month_start,
        )

        if hasattr(InmetWeatherObservation, "municipality_code") and municipality_code:
            climate_qs = climate_qs.filter(municipality_code=municipality_code)
        if hasattr(InmetWeatherObservation, "health_region_code") and health_region_code:
            climate_qs = climate_qs.filter(health_region_code=health_region_code)

        climate_agg = climate_qs.aggregate(
            temp_mean=Avg("air_temperature_instant_celsius"),
            humidity_mean=Avg("relative_humidity_instant_percent"),
            rain_sum=Sum("precipitation_accumulated_millimeters"),
        )

        days_temp_21_30 = climate_qs.filter(
            air_temperature_instant_celsius__gte=OPTIMAL_MIN,
            air_temperature_instant_celsius__lte=OPTIMAL_MAX,
        ).values("observation_date").distinct().count()

        days_temp_25_28 = climate_qs.filter(
            air_temperature_instant_celsius__gte=EIP_MIN,
            air_temperature_instant_celsius__lte=EIP_MAX,
        ).values("observation_date").distinct().count()

        temp_mean = float(climate_agg["temp_mean"]) if climate_agg["temp_mean"] is not None else None
        humidity_mean = float(climate_agg["humidity_mean"]) if climate_agg["humidity_mean"] is not None else None
        rain_sum = float(climate_agg["rain_sum"]) if climate_agg["rain_sum"] is not None else None

        score, explanations = heuristic_risk(
            temp_mean=temp_mean,
            humidity_mean=humidity_mean,
            rain_sum=rain_sum,
            days_temp_21_30=days_temp_21_30,
            days_temp_25_28=days_temp_25_28,
        )

        sinan_qs = SinanNotification.objects.filter(
            notification_date__gte=target_month_start,
            notification_date__lt=next_month(target_month_start),
        )

        if municipality_code:
            sinan_qs = sinan_qs.filter(notification_municipality_code=municipality_code)
        if health_region_code:
            sinan_qs = sinan_qs.filter(health_region_code=health_region_code)

        def cases_by_prefix(prefix: str) -> int:
            return sinan_qs.filter(disease_code__istartswith=prefix).count()

        observed_next = {}
        if disease in ("dengue", "both"):
            observed_next["dengue_cases_next_month"] = cases_by_prefix("DENGUE")
        if disease in ("chikungunya", "both"):
            observed_next["chikungunya_cases_next_month"] = cases_by_prefix("CHIK")

        return Response(
            {
                "reference_month": ref_month_start.strftime("%Y-%m"),
                "target_month": target_month_start.strftime("%Y-%m"),
                "region": {
                    "municipality_code": municipality_code,
                    "health_region_code": health_region_code,
                },
                "risk": {
                    "score": round(score, 3),
                    "label": risk_label(score),
                },
                "climate_summary": {
                    "temp_mean": temp_mean,
                    "humidity_mean": humidity_mean,
                    "rain_sum": rain_sum,
                    "days_temp_21_30": days_temp_21_30,
                    "days_temp_25_28": days_temp_25_28,
                },
                "explanations": explanations,
                "observed_next_month_cases": observed_next,
            }
        )