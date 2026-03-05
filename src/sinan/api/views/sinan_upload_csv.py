# sinan/views.py
import csv
from datetime import datetime
from io import TextIOWrapper
from typing import Any, Optional

from django.db import transaction
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from sinan.models import SinanNotification
from sinan.api.serializers import SinanNotificationCSVImportSerializer


def _clean_str(value: Any) -> Optional[str]:
    """
    - None/NaN/"" -> None
    - tira espaços nas pontas
    - mantém espaços do meio
    """
    if value is None:
        return None

    s = str(value)
    s = s.strip()

    if not s:
        return None

    lowered = s.lower()
    if lowered in ("nan", "none", "null"):
        return None

    return s


def _to_int(value: Any) -> Optional[int]:
    """
    Converte "23", "23.0", 23.0 em 23.
    Vazio/NaN -> None
    """
    s = _clean_str(value)
    if s is None:
        return None

    try:
        return int(float(s))
    except (ValueError, TypeError):
        return None


def _to_date_ddmmyyyy(value: Any) -> Optional[datetime.date]:
    """
    Parse de '08/06/2019' -> date
    """
    s = _clean_str(value)
    if s is None:
        return None

    try:
        return datetime.strptime(s, "%d/%m/%Y").date()
    except ValueError:
        return None


def _extract_epi_week(value: Any) -> Optional[int]:

    s = _clean_str(value)
    if s is None:
        return None

    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 2:
        return None

    week = int(digits[-2:])
    if 1 <= week <= 53:
        return week
    return None


class SinanNotificationCSVImportView(APIView):
    parser_classes = [MultiPartParser, FormParser]

    def post(self, request, *args, **kwargs):
        serializer = SinanNotificationCSVImportSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        upload = serializer.validated_data["file"]
        batch_size = serializer.validated_data["batch_size"]
        dry_run = serializer.validated_data["dry_run"]

        created = 0
        to_create = []
        errors = []

        wrapper = TextIOWrapper(upload.file, encoding="utf-8", errors="replace")
        reader = csv.DictReader(wrapper)

        required_headers = {
            "DT_NOTIFIC",
            "SEM_NOT",
            "NU_ANO",
            "ID_AGRAVO",
            "ID_MUNICIP",
            "ID_REGIONA",
            "ID_UNIDADE",
            "CS_SEXO",
            "CS_RACA",
            "CS_ESCOL_N",
            "ID_MN_RESI",
            "ID_RG_RESI",
            "EVOLUCAO",
            "IDADE_ANOS",
        }

        missing = required_headers - set(reader.fieldnames or [])
        if missing:
            return Response(
                {"detail": "CSV com colunas faltando.", "missing_headers": sorted(missing)},
                status=status.HTTP_400_BAD_REQUEST,
            )

        def flush():
            nonlocal created, to_create
            if not to_create:
                return
            if not dry_run:
                SinanNotification.objects.bulk_create(to_create, batch_size=batch_size)
            created += len(to_create)
            to_create = []

        with transaction.atomic():
            for row_index, row in enumerate(reader, start=2):
                try:
                    obj = SinanNotification(
                        notification_date=_to_date_ddmmyyyy(row.get("DT_NOTIFIC")),
                        notification_epidemiological_week=_extract_epi_week(row.get("SEM_NOT")),
                        notification_year=_to_int(row.get("NU_ANO")),
                        disease_code=_clean_str(row.get("ID_AGRAVO")),
                        notification_municipality_code=_clean_str(row.get("ID_MUNICIP")),
                        health_region_code=_clean_str(row.get("ID_REGIONA")),
                        reporting_health_unit_code=_clean_str(row.get("ID_UNIDADE")),
                        sex=_clean_str(row.get("CS_SEXO")),
                        race_color=_clean_str(row.get("CS_RACA")),
                        education_level=_clean_str(row.get("CS_ESCOL_N")),
                        residence_municipality_code=_clean_str(row.get("ID_MN_RESI")),
                        residence_health_region_code=_clean_str(row.get("ID_RG_RESI")),
                        case_outcome=_clean_str(row.get("EVOLUCAO")),
                        age_years=_to_int(row.get("IDADE_ANOS")),
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
