import csv
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from django.core.management.base import BaseCommand
from django.db import transaction

from sih.models import HospitalAdmission, HospitalOccupancyPrediction
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


class Command(BaseCommand):
    help = "Bootstrap hospital admissions and occupancy predictions if tables are empty."

    def add_arguments(self, parser):
        parser.add_argument(
            "--csv-path",
            type=str,
            default="/app/seed_data/hospital_admissions.csv",
        )
        parser.add_argument(
            "--start-date",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--horizon-days",
            type=int,
            default=30,
        )
        parser.add_argument(
            "--date-init",
            type=str,
            default="2021-01-01",
        )
        parser.add_argument(
            "--date-end",
            type=str,
            default="2024-12-31",
        )
        parser.add_argument(
            "--date-de-teste",
            type=str,
            default=None,
        )
        parser.add_argument(
            "--min-weeks",
            type=int,
            default=20,
        )
        parser.add_argument(
            "--csv-out",
            type=str,
            default="/app/models/hospital_forecast_next_30d.csv",
        )
        parser.add_argument(
            "--model-out",
            type=str,
            default="/app/models/hospital_occupancy.joblib",
        )
        parser.add_argument(
            "--force-import",
            action="store_true",
        )
        parser.add_argument(
            "--force-forecast",
            action="store_true",
        )

    def handle(self, *args, **options):
        csv_path = options["csv_path"]
        force_import = options["force_import"]
        force_forecast = options["force_forecast"]

        admissions_exists = HospitalAdmission.objects.exists()
        predictions_exists = HospitalOccupancyPrediction.objects.exists()

        self.stdout.write(self.style.NOTICE("Checking bootstrap status..."))
        self.stdout.write(f"HospitalAdmission has data: {admissions_exists}")
        self.stdout.write(f"HospitalOccupancyPrediction has data: {predictions_exists}")

        should_import = force_import or not admissions_exists
        should_forecast = force_forecast or not predictions_exists

        if should_import:
            self._import_csv(csv_path)
        else:
            self.stdout.write(self.style.SUCCESS("Skipping admission import: data already exists."))

        if should_forecast:
            self._run_forecast(options)
        else:
            self.stdout.write(self.style.SUCCESS("Skipping forecast generation: predictions already exist."))

        self.stdout.write(self.style.SUCCESS("Bootstrap finished."))

    @transaction.atomic
    def _import_csv(self, csv_path: str):
        path = Path(csv_path)

        if not path.exists():
            self.stdout.write(
                self.style.WARNING(
                    f"Admission CSV not found at {csv_path}. Skipping import."
                )
            )
            return

        self.stdout.write(self.style.NOTICE(f"Importing admissions from: {csv_path}"))

        with path.open("r", encoding="utf-8-sig", errors="replace") as file_obj:
            reader = csv.DictReader(file_obj)

            objects = []
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
            self.stdout.write(self.style.WARNING("No valid rows found to import."))
            return

        batch_size = 2000
        created = 0

        for i in range(0, len(objects), batch_size):
            batch = objects[i:i + batch_size]
            HospitalAdmission.objects.bulk_create(batch, ignore_conflicts=True)
            created += len(batch)

        self.stdout.write(
            self.style.SUCCESS(
                f"Admission import finished. Created: {created} | Skipped: {skipped}"
            )
        )

    def _run_forecast(self, options):
        if not HospitalAdmission.objects.exists():
            self.stdout.write(
                self.style.WARNING(
                    "Skipping forecast: HospitalAdmission has no data."
                )
            )
            return

        self.stdout.write(self.style.NOTICE("Generating hospital occupancy forecast..."))

        result = run_hospital_occupancy_forecast(
            start_date=options["start_date"],
            horizon_days=options["horizon_days"],
            date_init=options["date_init"],
            date_end=options["date_end"],
            date_de_teste=options["date_de_teste"],
            min_weeks=options["min_weeks"],
            csv_out=options["csv_out"],
            model_out=options["model_out"],
        )

        if result.get("ok"):
            self.stdout.write(self.style.SUCCESS(f"Forecast finished: {result}"))
        else:
            self.stdout.write(self.style.WARNING(f"Forecast not generated: {result}"))