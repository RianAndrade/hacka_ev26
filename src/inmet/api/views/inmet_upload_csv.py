import csv
from datetime import datetime
from io import TextIOWrapper
from typing import Any, Iterator, Optional

from django.db import transaction
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from inmet.models import InmetWeatherObservation
from inmet.api.serializers.inmet_upload_csv import (
    InmetWeatherObservationCSVImportSerializer,
)


def clean_str(value: Any) -> Optional[str]:
    if value is None:
        return None

    s = str(value).strip()
    if not s:
        return None

    lowered = s.lower()
    if lowered in ("nan", "none", "null"):
        return None

    return s


def normalize_inmet_number(value: Any) -> Optional[str]:
    s = clean_str(value)
    if s is None:
        return None
    return s.replace('"', "")


def to_float(value: Any) -> Optional[float]:
    s = normalize_inmet_number(value)
    if s is None:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def to_int_from_float(value: Any) -> Optional[int]:
    n = to_float(value)
    if n is None:
        return None
    try:
        return int(n)
    except (ValueError, TypeError):
        return None


def to_date_ddmmyyyy(value: Any) -> Optional[datetime.date]:
    s = clean_str(value)
    if s is None:
        return None
    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def to_time_hhmm(value: Any) -> Optional[datetime.time]:
    s = clean_str(value)
    if s is None:
        return None

    digits = "".join(ch for ch in s if ch.isdigit())
    if not digits:
        return None

    digits = digits.zfill(4)
    if len(digits) != 4:
        return None

    hh = int(digits[:2])
    mm = int(digits[2:])

    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return None

    return datetime.strptime(f"{hh:02d}:{mm:02d}", "%H:%M").time()


def clean_inmet_line(line: str) -> str:
    line = line.strip("\n\r")

    if line.startswith('"') and line.endswith('"'):
        line = line[1:-1]

    line = line.lstrip("\ufeff")
    line = line.replace('""', '"')

    return line


def iter_clean_lines(wrapper: TextIOWrapper) -> Iterator[str]:
    for line in wrapper:
        yield clean_inmet_line(line)


class InmetWeatherObservationCSVImportView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = InmetWeatherObservationCSVImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        upload = serializer.validated_data["file"]
        batch_size = serializer.validated_data["batch_size"]
        dry_run = serializer.validated_data["dry_run"]

        created = 0
        to_create = []
        errors = []

        wrapper = TextIOWrapper(upload.file, encoding="utf-8", errors="replace")
        reader = csv.DictReader(
            iter_clean_lines(wrapper),
            delimiter=";",
            quotechar='"',
        )

        required_headers = {
            "Data",
            "Hora (UTC)",
            "Temp. Ins. (C)",
            "Temp. Max. (C)",
            "Temp. Min. (C)",
            "Umi. Ins. (%)",
            "Umi. Max. (%)",
            "Umi. Min. (%)",
            "Pto Orvalho Ins. (C)",
            "Pto Orvalho Max. (C)",
            "Pto Orvalho Min. (C)",
            "Pressao Ins. (hPa)",
            "Pressao Max. (hPa)",
            "Pressao Min. (hPa)",
            "Vel. Vento (m/s)",
            "Dir. Vento (m/s)",
            "Raj. Vento (m/s)",
            "Radiacao (KJ/m²)",
            "Chuva (mm)",
        }

        missing = required_headers - set(reader.fieldnames or [])
        if missing:
            return Response(
                {
                    "detail": "CSV is missing required columns.",
                    "missing_headers": sorted(missing),
                },
                status=status.HTTP_400_BAD_REQUEST,
            )

        def flush():
            nonlocal created, to_create
            if not to_create:
                return
            if not dry_run:
                InmetWeatherObservation.objects.bulk_create(
                    to_create,
                    batch_size=batch_size,
                )
            created += len(to_create)
            to_create = []

        with transaction.atomic():
            for row_index, row in enumerate(reader, start=2):
                try:
                    obj = InmetWeatherObservation(
                        observation_date=to_date_ddmmyyyy(row.get("Data")),
                        observation_time_utc=to_time_hhmm(row.get("Hora (UTC)")),

                        air_temperature_instant_celsius=to_float(row.get("Temp. Ins. (C)")),
                        air_temperature_max_celsius=to_float(row.get("Temp. Max. (C)")),
                        air_temperature_min_celsius=to_float(row.get("Temp. Min. (C)")),

                        relative_humidity_instant_percent=to_float(row.get("Umi. Ins. (%)")),
                        relative_humidity_max_percent=to_float(row.get("Umi. Max. (%)")),
                        relative_humidity_min_percent=to_float(row.get("Umi. Min. (%)")),

                        dew_point_temperature_instant_celsius=to_float(row.get("Pto Orvalho Ins. (C)")),
                        dew_point_temperature_max_celsius=to_float(row.get("Pto Orvalho Max. (C)")),
                        dew_point_temperature_min_celsius=to_float(row.get("Pto Orvalho Min. (C)")),

                        atmospheric_pressure_instant_hpa=to_float(row.get("Pressao Ins. (hPa)")),
                        atmospheric_pressure_max_hpa=to_float(row.get("Pressao Max. (hPa)")),
                        atmospheric_pressure_min_hpa=to_float(row.get("Pressao Min. (hPa)")),

                        wind_speed_mean_meters_per_second=to_float(row.get("Vel. Vento (m/s)")),
                        wind_direction_degrees=to_int_from_float(row.get("Dir. Vento (m/s)")),
                        wind_gust_max_meters_per_second=to_float(row.get("Raj. Vento (m/s)")),

                        global_solar_radiation_kj_per_square_meter=to_float(row.get("Radiacao (KJ/m²)")),
                        precipitation_accumulated_millimeters=to_float(row.get("Chuva (mm)")),
                    )

                    to_create.append(obj)

                    if len(to_create) >= batch_size:
                        flush()

                except Exception as exc:
                    errors.append(
                        {
                            "line": row_index,
                            "error": str(exc),
                        }
                    )

            flush()

            if dry_run:
                transaction.set_rollback(True)

        return Response(
            {
                "created": created,
                "dry_run": dry_run,
                "errors_count": len(errors),
                "errors_sample": errors[:20],
            },
            status=status.HTTP_201_CREATED if not dry_run else status.HTTP_200_OK,
        )