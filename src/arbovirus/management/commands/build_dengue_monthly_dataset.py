from datetime import date

from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Avg, Sum, Count, Q
from django.db.models.functions import TruncMonth

from sinan.models.sinan_notification import SinanNotification
from inmet.models.inmet import InmetWeatherObservation
from arbovirus.models.dengue_monthly_dataset import DengueMonthlyDataset


class Command(BaseCommand):

    def add_arguments(self, parser):
        parser.add_argument("--municipality", required=True, type=str)
        parser.add_argument("--health-region", required=False, type=str, default="")
        parser.add_argument("--start", required=True, type=str)  # YYYY-MM-DD
        parser.add_argument("--end", required=True, type=str)    # YYYY-MM-DD

    @transaction.atomic
    def handle(self, *args, **options):
        municipality_code = options["municipality"]
        health_region_code = options["health_region"] or ""
        start = date.fromisoformat(options["start"])
        end = date.fromisoformat(options["end"])

        inmet_qs = InmetWeatherObservation.objects.filter(
            observation_date__gte=start,
            observation_date__lte=end,
        ).annotate(
            month=TruncMonth("observation_date")
        ).values("month").annotate(
            temp_mean=Avg("air_temperature_instant_celsius"),
            temp_max_mean=Avg("air_temperature_max_celsius"),
            temp_min_mean=Avg("air_temperature_min_celsius"),
            humidity_mean=Avg("relative_humidity_instant_percent"),
            humidity_max_mean=Avg("relative_humidity_max_percent"),
            humidity_min_mean=Avg("relative_humidity_min_percent"),
            rain_sum=Sum("precipitation_accumulated_millimeters"),
            pressure_mean=Avg("atmospheric_pressure_instant_hpa"),
            wind_speed_mean=Avg("wind_speed_mean_meters_per_second"),
            solar_radiation_sum=Sum("global_solar_radiation_kj_per_square_meter"),
        )

        inmet_by_month = {row["month"]: row for row in inmet_qs}

        sinan_filter = Q(notification_date__gte=start, notification_date__lte=end)
        sinan_filter &= Q(residence_municipality_code=municipality_code)

        if health_region_code:
            sinan_filter &= Q(residence_health_region_code=health_region_code)

        sinan_filter &= Q(disease_code__startswith="A9")

        sinan_qs = SinanNotification.objects.filter(
            sinan_filter
        ).annotate(
            month=TruncMonth("notification_date")
        ).values("month").annotate(
            dengue_cases=Count("id")
        )

        sinan_by_month = {row["month"]: row["dengue_cases"] for row in sinan_qs}

        all_months = sorted(set(inmet_by_month.keys()) | set(sinan_by_month.keys()))

        for m in all_months:
            w = inmet_by_month.get(m, {})
            dengue_cases = sinan_by_month.get(m, 0)

            defaults = {
                "temp_mean": float(w["temp_mean"]) if w.get("temp_mean") is not None else None,
                "temp_max_mean": float(w["temp_max_mean"]) if w.get("temp_max_mean") is not None else None,
                "temp_min_mean": float(w["temp_min_mean"]) if w.get("temp_min_mean") is not None else None,
                "humidity_mean": float(w["humidity_mean"]) if w.get("humidity_mean") is not None else None,
                "humidity_max_mean": float(w["humidity_max_mean"]) if w.get("humidity_max_mean") is not None else None,
                "humidity_min_mean": float(w["humidity_min_mean"]) if w.get("humidity_min_mean") is not None else None,
                "rain_sum": float(w["rain_sum"]) if w.get("rain_sum") is not None else None,
                "pressure_mean": float(w["pressure_mean"]) if w.get("pressure_mean") is not None else None,
                "wind_speed_mean": float(w["wind_speed_mean"]) if w.get("wind_speed_mean") is not None else None,
                "solar_radiation_sum": float(w["solar_radiation_sum"]) if w.get("solar_radiation_sum") is not None else None,
                "optimal_days": None,
                "dengue_cases": dengue_cases,
            }

            DengueMonthlyDataset.objects.update_or_create(
                month=m,
                municipality_code=municipality_code,
                health_region_code=health_region_code,
                defaults=defaults,
            )

        self._fill_next_cases(municipality_code, health_region_code)
        self.stdout.write(self.style.SUCCESS("Monthly dataset upserted successfully."))

    def _fill_next_cases(self, municipality_code: str, health_region_code: str) -> None:
        qs = DengueMonthlyDataset.objects.filter(
            municipality_code=municipality_code,
            health_region_code=health_region_code,
        ).order_by("month")

        prev = None
        for row in qs:
            if prev is not None:
                prev.dengue_cases_next = row.dengue_cases or 0
                prev.save(update_fields=["dengue_cases_next"])
            prev = row