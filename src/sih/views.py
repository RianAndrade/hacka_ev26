import csv
from datetime import datetime, timedelta
from decimal import Decimal

from celery.result import AsyncResult
from django.db import transaction
from django.db.models import Avg, Case, Count, DurationField, ExpressionWrapper, F, Q, Sum, When
from django.db.models.functions import TruncMonth, TruncWeek
from django.urls import reverse
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.parsers import MultiPartParser, FormParser
from rest_framework.response import Response
from rest_framework.views import APIView

from sih.tasks import run_hospital_occupancy_forecast
from .models import HospitalAdmission, HospitalOccupancyPrediction

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

        batch_size = 2000
        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
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


class HospitalOccupancyPredictionByHospitalView(APIView):

    def get(self, request):
        hospital = (request.query_params.get("hospital") or "").strip()
        horizon_days = request.query_params.get("horizon_days", "30").strip()
        start_date_raw = (request.query_params.get("start_date") or "").strip()

        if not hospital:
            return Response(
                {"detail": "Query param 'hospital' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if horizon_days not in {"30", "60", "90"}:
            return Response(
                {"detail": "Query param 'horizon_days' must be one of: 30, 60, 90."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if start_date_raw:
            start_date = _parse_date(start_date_raw)
            if not start_date:
                return Response(
                    {"detail": "Invalid 'start_date'. Use YYYY-MM-DD, DD/MM/YYYY or YYYYMMDD."},
                    status=status.HTTP_400_BAD_REQUEST,
                )
        else:
            start_date = datetime.today().date()

        end_date = start_date + timedelta(days=int(horizon_days))

        queryset = (
            HospitalOccupancyPrediction.objects
            .filter(
                hospital__iexact=hospital,
                week_start__gte=start_date,
                week_start__lte=end_date,
            )
            .order_by("week_start")
        )

        predictions = {}
        for item in queryset:
            predictions[item.week_start.isoformat()] = {
                "estimated_total": item.estimated_total,
                "estimated_avg_length_of_stay": item.estimated_avg_length_of_stay,
            }

        return Response(
            {
                "hospital": hospital,
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "horizon_days": int(horizon_days),
                "weeks": predictions,
            },
            status=status.HTTP_200_OK,
        )

class HospitalOccupancyAvailableHospitalsView(APIView):

    def get(self, request):
        hospitals = (
            HospitalOccupancyPrediction.objects
            .order_by("hospital")
            .values_list("hospital", flat=True)
            .distinct()
        )

        return Response(
            {
                "count": len(hospitals),
                "results": list(hospitals),
            },
            status=status.HTTP_200_OK,
        )
    
class HospitalOccupancyDashboardView(TemplateView):
    template_name = "sih/hospital_occupancy_dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["available_hospitals_url"] = reverse("hospital-occupancy-hospitals")
        context["predictions_url"] = reverse("hospital-occupancy-predictions")
        context["forecast_dashboard_url"] = reverse("hospital-occupancy-dashboard")
        context["historical_dashboard_url"] = reverse("hospital-admission-historical-dashboard")
        return context

def _distinct_non_empty_values(queryset, field_name: str):
    return list(
        queryset.exclude(**{f"{field_name}__isnull": True})
        .exclude(**{f"{field_name}__exact": ""})
        .values_list(field_name, flat=True)
        .distinct()
        .order_by(field_name)
    )


def _parse_optional_int(value):
    v = (value or "").strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        return None


class HospitalAdmissionHistoricalFilterOptionsView(APIView):

    def get(self, request):
        base_queryset = HospitalAdmission.objects.all()

        years = list(
            base_queryset.values_list("admission_date__year", flat=True)
            .distinct()
            .order_by("admission_date__year")
        )

        sexes = _distinct_non_empty_values(base_queryset, "patient_sex")
        diagnosis_codes = _distinct_non_empty_values(
            base_queryset,
            "primary_diagnosis_icd10_code",
        )
        admission_types = _distinct_non_empty_values(base_queryset, "admission_type")
        specialty_codes = _distinct_non_empty_values(
            base_queryset,
            "bed_or_admission_specialty_code",
        )
        facility_codes = _distinct_non_empty_values(
            base_queryset,
            "health_facility_registry_code",
        )

        return Response(
            {
                "years": [year for year in years if year is not None],
                "sexes": sexes,
                "diagnosis_codes": diagnosis_codes,
                "admission_types": admission_types,
                "specialty_codes": specialty_codes,
                "facility_codes": facility_codes,
                "period_options": [
                    {"value": "week", "label": "Semanal"},
                    {"value": "month", "label": "Mensal"},
                ],
                "death_options": [
                    {"value": "true", "label": "Sim"},
                    {"value": "false", "label": "Não"},
                ],
            },
            status=status.HTTP_200_OK,
        )


class HospitalAdmissionHistoricalSummaryView(APIView):

    def get(self, request):
        year = _parse_optional_int(request.query_params.get("year"))
        period = (request.query_params.get("period") or "week").strip().lower()

        patient_sex = (request.query_params.get("patient_sex") or "").strip()
        diagnosis_code = (request.query_params.get("diagnosis_code") or "").strip()
        admission_type = (request.query_params.get("admission_type") or "").strip()
        specialty_code = (request.query_params.get("specialty_code") or "").strip()
        facility_code = (request.query_params.get("facility_code") or "").strip()
        death_during_admission = (
            request.query_params.get("death_during_admission") or ""
        ).strip().lower()

        min_stay_days = _parse_optional_int(request.query_params.get("min_stay_days"))
        max_stay_days = _parse_optional_int(request.query_params.get("max_stay_days"))
        min_icu_days = _parse_optional_int(request.query_params.get("min_icu_days"))
        max_icu_days = _parse_optional_int(request.query_params.get("max_icu_days"))

        weeks_count = _parse_optional_int(request.query_params.get("weeks_count"))
        start_month = _parse_optional_int(request.query_params.get("start_month"))

        if year is None:
            return Response(
                {"detail": "Query param 'year' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if not facility_code:
            return Response(
                {"detail": "Query param 'facility_code' is required."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if period not in {"week", "month"}:
            return Response(
                {"detail": "Query param 'period' must be 'week' or 'month'."},
                status=status.HTTP_400_BAD_REQUEST,
            )

        if period == "week":
            if weeks_count is None or weeks_count <= 0:
                return Response(
                    {
                        "detail": (
                            "Query param 'weeks_count' is required and must be "
                            "greater than 0 when period='week'."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

            if start_month is None or start_month < 1 or start_month > 12:
                return Response(
                    {
                        "detail": (
                            "Query param 'start_month' is required and must be "
                            "between 1 and 12 when period='week'."
                        )
                    },
                    status=status.HTTP_400_BAD_REQUEST,
                )

        queryset = HospitalAdmission.objects.filter(
            admission_date__year=year,
            health_facility_registry_code=facility_code,
        )

        if period == "week" and start_month is not None:
            queryset = queryset.filter(admission_date__month__gte=start_month)

        if patient_sex:
            queryset = queryset.filter(patient_sex=patient_sex)

        if diagnosis_code:
            queryset = queryset.filter(primary_diagnosis_icd10_code=diagnosis_code)

        if admission_type:
            queryset = queryset.filter(admission_type=admission_type)

        if specialty_code:
            queryset = queryset.filter(bed_or_admission_specialty_code=specialty_code)

        if death_during_admission == "true":
            queryset = queryset.filter(death_during_admission=True)
        elif death_during_admission == "false":
            queryset = queryset.filter(death_during_admission=False)

        queryset = queryset.annotate(
            stay_duration=Case(
                When(
                    discharge_date__isnull=False,
                    then=ExpressionWrapper(
                        F("discharge_date") - F("admission_date"),
                        output_field=DurationField(),
                    ),
                ),
                default=None,
                output_field=DurationField(),
            )
        )

        if min_stay_days is not None:
            queryset = queryset.filter(stay_duration__gte=timedelta(days=min_stay_days))

        if max_stay_days is not None:
            queryset = queryset.filter(stay_duration__lte=timedelta(days=max_stay_days))

        if min_icu_days is not None:
            queryset = queryset.filter(intensive_care_total_days__gte=min_icu_days)

        if max_icu_days is not None:
            queryset = queryset.filter(intensive_care_total_days__lte=max_icu_days)

        period_trunc = (
            TruncWeek("admission_date")
            if period == "week"
            else TruncMonth("admission_date")
        )

        grouped_queryset = (
            queryset.annotate(period_start=period_trunc)
            .values("period_start")
            .annotate(
                occurrences=Count("record_identifier"),
                avg_stay_duration=Avg("stay_duration"),
                deaths=Count(
                    "record_identifier",
                    filter=Q(death_during_admission=True),
                ),
                total_amount_paid=Sum("total_amount_paid_brl"),
            )
            .order_by("period_start")
        )

        grouped_list = list(grouped_queryset)

        if period == "week" and weeks_count is not None:
            grouped_list = grouped_list[:weeks_count]

        periods = {}
        selected_period_keys = []

        for item in grouped_list:
            avg_stay_duration = item["avg_stay_duration"]
            avg_stay_days = None

            if avg_stay_duration is not None:
                avg_stay_days = round(avg_stay_duration.total_seconds() / 86400, 2)

            total_amount_paid = item["total_amount_paid"]
            if total_amount_paid is not None:
                total_amount_paid = float(total_amount_paid)

            period_start = item["period_start"]
            period_key = (
                period_start.isoformat()
                if hasattr(period_start, "isoformat")
                else str(period_start)
            )

            selected_period_keys.append(period_start)

            periods[period_key] = {
                "occurrences": item["occurrences"],
                "avg_length_of_stay": avg_stay_days,
                "deaths": item["deaths"],
                "total_amount_paid": round(total_amount_paid or 0, 2),
            }

        summary_queryset = queryset

        if selected_period_keys:
            summary_queryset = queryset.annotate(period_start=period_trunc).filter(
                period_start__in=selected_period_keys
            )
        elif period == "week":
            summary_queryset = queryset.none()

        overall = summary_queryset.aggregate(
            total_occurrences=Count("record_identifier"),
            avg_stay_duration=Avg("stay_duration"),
            total_deaths=Count(
                "record_identifier",
                filter=Q(death_during_admission=True),
            ),
            total_amount_paid=Sum("total_amount_paid_brl"),
        )

        overall_avg_stay = overall["avg_stay_duration"]
        overall_avg_stay_days = None

        if overall_avg_stay is not None:
            overall_avg_stay_days = round(overall_avg_stay.total_seconds() / 86400, 2)

        return Response(
            {
                "year": year,
                "period": period,
                "filters": {
                    "patient_sex": patient_sex or None,
                    "diagnosis_code": diagnosis_code or None,
                    "admission_type": admission_type or None,
                    "specialty_code": specialty_code or None,
                    "facility_code": facility_code,
                    "death_during_admission": death_during_admission or None,
                    "min_stay_days": min_stay_days,
                    "max_stay_days": max_stay_days,
                    "min_icu_days": min_icu_days,
                    "max_icu_days": max_icu_days,
                    "weeks_count": weeks_count,
                    "start_month": start_month,
                },
                "summary": {
                    "total_occurrences": overall["total_occurrences"] or 0,
                    "avg_length_of_stay": overall_avg_stay_days,
                    "total_deaths": overall["total_deaths"] or 0,
                    "total_amount_paid": round(float(overall["total_amount_paid"] or 0), 2),
                },
                "periods": periods,
            },
            status=status.HTTP_200_OK,
        )

class HospitalAdmissionHistoricalDashboardView(TemplateView):
    template_name = "sih/hospital_occupancy_historical_dashboard.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["filter_options_url"] = reverse("hospital-admission-historical-filter-options")
        context["historical_summary_url"] = reverse("hospital-admission-historical-summary")
        context["forecast_dashboard_url"] = reverse("hospital-occupancy-dashboard")
        context["historical_dashboard_url"] = reverse("hospital-admission-historical-dashboard")
        return context