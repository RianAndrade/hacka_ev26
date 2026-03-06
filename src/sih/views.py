import csv
from datetime import datetime
from decimal import Decimal

from django.db import transaction
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView
from sih.tasks import run_hospital_occupancy_forecast
from celery.result import AsyncResult
from .models import HospitalAdmission

from sih.tasks import run_hospital_occupancy_forecast


def _parse_date(value: str):
    v = (value or "").strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%Y%m%d"):
        try:
            return datetime.strptime(v, fmt).date()
        except ValueError:
            continue
    return None


def _parse_bool(value: str) -> bool:
    v = (value or "").strip().lower()
    return v in {"1", "true", "t", "yes", "y", "sim", "s"}


def _parse_int(value: str, default: int = 0) -> int:
    v = (value or "").strip()
    if not v:
        return default
    try:
        return int(float(v.replace(",", ".")))
    except ValueError:
        return default


def _parse_decimal(value: str) -> Decimal:
    v = (value or "").strip()
    if not v:
        return Decimal("0.00")
    v = v.replace(".", "").replace(",", ".") if v.count(",") == 1 else v.replace(",", ".")
    try:
        return Decimal(v)
    except Exception:
        return Decimal("0.00")


def _norm_upper(value: str) -> str:
    return (value or "").strip().upper()


class HospitalAdmissionCsvImportView(APIView):
    parser_classes = (MultiPartParser, FormParser)

    @transaction.atomic
    def post(self, request):
        uploaded = request.FILES.get("file")
        if not uploaded:
            return Response(
                {"detail": "Missing file field 'file'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        decoded = uploaded.read().decode("utf-8-sig", errors="replace").splitlines()
        reader = csv.DictReader(decoded)

        objects = []
        created = 0
        skipped = 0

        for row in reader:
            try:
                birth_date = _parse_date(row.get("NASC"))
                admission_date = _parse_date(row.get("DT_INTER"))
                if not birth_date or not admission_date:
                    skipped += 1
                    continue

                obj = HospitalAdmission(
                    record_identifier=_parse_int(row.get("ID"), default=0) or None,
                    processing_competence_year=_parse_int(row.get("ANO_CMPT")),
                    processing_competence_month=_parse_int(row.get("MES_CMPT")),
                    admission_state_code=_norm_upper(row.get("UF_ZI")),
                    bed_or_admission_specialty_code=_norm_upper(row.get("ESPEC")),
                    health_facility_registry_code=_norm_upper(row.get("CNES")),
                    patient_residence_municipality_code=_norm_upper(row.get("MUNIC_RES")),
                    admission_municipality_code=_norm_upper(row.get("MUNIC_MOV")),
                    patient_birth_date=birth_date,
                    patient_sex=_norm_upper(row.get("SEXO")),
                    admission_date=admission_date,
                    discharge_date=_parse_date(row.get("DT_SAIDA")),
                    primary_diagnosis_icd10_code=_norm_upper(row.get("DIAG_PRINC")),
                    death_during_admission=_parse_bool(row.get("MORTE")),
                    intensive_care_total_days=_parse_int(row.get("UTI_INT_TO")),
                    admission_type=_norm_upper(row.get("CAR_INT")),
                    total_amount_paid_brl=_parse_decimal(row.get("VAL_TOT")),
                )
                objects.append(obj)
            except Exception:
                skipped += 1

        if not objects:
            return Response(
                {"created": 0, "skipped": skipped, "detail": "No rows to import."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        BATCH_SIZE = 2000
        for i in range(0, len(objects), BATCH_SIZE):
            batch = objects[i : i + BATCH_SIZE]
            HospitalAdmission.objects.bulk_create(batch, ignore_conflicts=True)
            created += len(batch)

        return Response(
            {"created": created, "skipped": skipped},
            status=status.HTTP_201_CREATED,
        )


class RunHospitalOccupancyForecastView(APIView):

    def post(self, request):
        async_result = run_hospital_occupancy_forecast.delay()
        return Response(
            {"task_id": async_result.id, "status": "queued"},
            status=status.HTTP_202_ACCEPTED,
        )


class HospitalOccupancyForecastStatusView(APIView):

    def get(self, request, task_id: str):
        res = AsyncResult(task_id)
        payload = {
            "task_id": task_id,
            "state": res.state,
        }

        if res.state == "SUCCESS":
            payload["result"] = res.result
        elif res.state == "FAILURE":
            payload["error"] = str(res.result)

        return Response(payload, status=status.HTTP_200_OK)
    


class HospitalOccupancyForecastView(APIView):
    def get(self, request):
        qp = request.query_params

        payload = {
            "start_date": qp.get("start_date"),
            "horizon_days": int(qp.get("horizon_days", 30)),
            "date_init": qp.get("date_init"),
            "date_end": qp.get("date_end"),
            "date_de_teste": qp.get("date_de_teste"),
            "min_weeks": int(qp.get("min_weeks", 20)),
            "csv_out": qp.get("csv_out", "/app/models/hospital_forecast_next_30d.csv"),
            "model_out": qp.get("model_out", "/app/models/hospital_occupancy.joblib"),
        }

        sync = qp.get("sync", "0") in ("1", "true", "True")

        if sync:
            result = run_hospital_occupancy_forecast(**payload)
            return Response(result, status=status.HTTP_200_OK)

        task = run_hospital_occupancy_forecast.delay(**payload)
        return Response(
            {"task_id": task.id, "status": "queued"},
            status=status.HTTP_202_ACCEPTED,
        )