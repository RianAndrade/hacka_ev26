from __future__ import annotations

from datetime import date, timedelta

from django.core.management.base import BaseCommand
from django.db import transaction
from django.db.models import Avg, Sum, Count, Q
from django.db.models.functions import ExtractWeek, ExtractIsoYear, TruncDate

from sinan.models.sinan_notification import SinanNotification
from inmet.models.inmet import InmetWeatherObservation
from arbovirus.models.dengue_weekly_dataset import DengueWeeklyDataset


class Command(BaseCommand):
    help = "Build/refresh dengue weekly dataset using SINAN + INMET"

    def add_arguments(self, parser):
        parser.add_argument("--municipality", required=True, type=str)
        parser.add_argument("--health-region", required=False, type=str, default="")
        parser.add_argument("--start", required=True, type=str)  # YYYY-MM-DD
        parser.add_argument("--end", required=True, type=str)    # YYYY-MM-DD
        parser.add_argument("--verbose-weeks", action="store_true", default=False)

    @transaction.atomic
    def handle(self, *args, **options):
        municipality_code: str = options["municipality"]
        health_region_code: str = (options["health_region"] or "").strip()
        start: date = date.fromisoformat(options["start"])
        end: date = date.fromisoformat(options["end"])
        verbose_weeks: bool = bool(options["verbose_weeks"])

        self.stdout.write(
            f"[build] municipality={municipality_code} health_region={health_region_code or '(none)'} "
            f"range={start}..{end}"
        )

        # --------------------------
        # INMET - daily aggregation
        # --------------------------
        inmet_qs = InmetWeatherObservation.objects.filter(
            observation_date__gte=start,
            observation_date__lte=end,
        )

        inmet_total = inmet_qs.count()
        self.stdout.write(f"[inmet] rows_in_range={inmet_total}")

        daily_weather = (
            inmet_qs
            .annotate(day=TruncDate("observation_date"))
            .values("day")
            .annotate(
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
        )

        day_map = {row["day"]: row for row in daily_weather}
        self.stdout.write(f"[inmet] unique_days={len(day_map)}")

        def week_start(d: date) -> date:
            return d - timedelta(days=d.weekday())

        first_week = week_start(start)
        last_week = week_start(end)
        self.stdout.write(f"[weeks] first_week_start={first_week} last_week_start={last_week}")

        # --------------------------
        # SINAN - weekly dengue cases
        # --------------------------
        base_sinan_filter = Q(notification_date__gte=start, notification_date__lte=end)
        base_sinan_filter &= Q(residence_municipality_code=municipality_code)
        base_sinan_filter &= Q(disease_code__startswith="A9")

        # DEBUG: counts without health-region (should match your SQL logic)
        sinan_total_no_hr = SinanNotification.objects.filter(base_sinan_filter).count()
        self.stdout.write(f"[sinan] dengue_rows(no_health_region_filter)={sinan_total_no_hr}")

        sinan_filter = base_sinan_filter
        if health_region_code:
            sinan_filter &= Q(residence_health_region_code=health_region_code)

        sinan_total = SinanNotification.objects.filter(sinan_filter).count()
        self.stdout.write(f"[sinan] dengue_rows(with_health_region_filter)={sinan_total}")

        if health_region_code and sinan_total == 0 and sinan_total_no_hr > 0:
            self.stdout.write(
                self.style.WARNING(
                    "[warn] With --health-region the SINAN filter became EMPTY. "
                    "This usually means residence_health_region_code != your value (or null). "
                    "Try running without --health-region or confirm which SINAN field is correct."
                )
            )

        sinan_weekly = (
            SinanNotification.objects.filter(sinan_filter)
            .annotate(
                iso_year=ExtractIsoYear("notification_date"),
                iso_week=ExtractWeek("notification_date"),
            )
            .values("iso_year", "iso_week")
            .annotate(dengue_cases=Count("id"))
        )

        # IMPORTANT: cast keys to int to avoid Decimal/float mismatch
        sinan_week_map = {
            (int(row["iso_year"]), int(row["iso_week"])): int(row["dengue_cases"])
            for row in sinan_weekly
            if row["iso_year"] is not None and row["iso_week"] is not None
        }
        self.stdout.write(f"[sinan] weeks_with_cases={len(sinan_week_map)}")

        # --------------------------
        # Build/Upsert DengueWeeklyDataset
        # --------------------------
        created_count = 0
        updated_count = 0
        weeks_processed = 0

        current = first_week
        while current <= last_week:
            week_days = [current + timedelta(days=i) for i in range(7)]
            rows = [day_map.get(d) for d in week_days if day_map.get(d)]

            def avg_of(key: str):
                vals = [r[key] for r in rows if r.get(key) is not None]
                return float(sum(vals) / len(vals)) if vals else None

            def sum_of(key: str):
                vals = [r[key] for r in rows if r.get(key) is not None]
                return float(sum(vals)) if vals else None

            optimal_days = 0
            for r in rows:
                t = r.get("temp_mean")
                h = r.get("humidity_mean")
                rain = r.get("rain_sum")
                if t is None or h is None:
                    continue
                if 22 <= float(t) <= 32 and float(h) >= 60 and (rain is None or float(rain) >= 0):
                    optimal_days += 1

            iso_year = int(current.isocalendar().year)
            iso_week = int(current.isocalendar().week)

            dengue_cases = sinan_week_map.get((iso_year, iso_week), 0)

            obj, created = DengueWeeklyDataset.objects.update_or_create(
                week_start=current,
                municipality_code=municipality_code,
                health_region_code=health_region_code,
                defaults={
                    "epidemiological_year": iso_year,
                    "epidemiological_week": iso_week,
                    "temp_mean": avg_of("temp_mean"),
                    "temp_max_mean": avg_of("temp_max_mean"),
                    "temp_min_mean": avg_of("temp_min_mean"),
                    "humidity_mean": avg_of("humidity_mean"),
                    "humidity_max_mean": avg_of("humidity_max_mean"),
                    "humidity_min_mean": avg_of("humidity_min_mean"),
                    "rain_sum": sum_of("rain_sum"),
                    "pressure_mean": avg_of("pressure_mean"),
                    "wind_speed_mean": avg_of("wind_speed_mean"),
                    "solar_radiation_sum": sum_of("solar_radiation_sum"),
                    "optimal_days": optimal_days,
                    "dengue_cases": dengue_cases,
                },
            )

            if created:
                created_count += 1
            else:
                updated_count += 1

            weeks_processed += 1

            if verbose_weeks and (dengue_cases > 0 or weeks_processed % 25 == 0):
                self.stdout.write(
                    f"[week] {current} iso={iso_year}-W{iso_week:02d} "
                    f"dengue_cases={dengue_cases} inmet_days={len(rows)} optimal_days={optimal_days} "
                    f"{'CREATED' if created else 'UPDATED'}"
                )

            current += timedelta(days=7)

        self.stdout.write(
            f"[done] weeks_processed={weeks_processed} created={created_count} updated={updated_count}"
        )

        self._fill_next_cases(municipality_code, health_region_code)

        # sanity: sum dengue in dataset vs sum from map
        total_dataset = (
            DengueWeeklyDataset.objects.filter(
                municipality_code=municipality_code,
                health_region_code=health_region_code,
                week_start__gte=first_week,
                week_start__lte=last_week,
            )
            .aggregate(total=Sum("dengue_cases"))
            .get("total")
            or 0
        )
        total_expected = sum(sinan_week_map.values())
        self.stdout.write(f"[check] dataset_sum_dengue_cases={int(total_dataset)} sinan_sum_from_map={int(total_expected)}")

        self.stdout.write(self.style.SUCCESS("Weekly dataset upserted successfully."))

    def _fill_next_cases(self, municipality_code: str, health_region_code: str) -> None:
        qs = list(
            DengueWeeklyDataset.objects.filter(
                municipality_code=municipality_code,
                health_region_code=health_region_code,
            ).order_by("week_start")
        )

        if len(qs) < 2:
            return

        to_update = []
        for i in range(len(qs) - 1):
            current = qs[i]
            nxt = qs[i + 1]
            current.dengue_cases_next = nxt.dengue_cases or 0
            to_update.append(current)

        DengueWeeklyDataset.objects.bulk_update(to_update, ["dengue_cases_next"])