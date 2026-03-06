from django.db import models


class HospitalAdmission(models.Model):
    record_identifier = models.BigAutoField(primary_key=True)

    processing_competence_year = models.PositiveSmallIntegerField()
    processing_competence_month = models.PositiveSmallIntegerField()

    admission_state_code = models.CharField(max_length=10)
    bed_or_admission_specialty_code = models.CharField(max_length=80)
    health_facility_registry_code = models.CharField(max_length=255)

    patient_residence_municipality_code = models.CharField(max_length=10)
    admission_municipality_code = models.CharField(max_length=10)

    patient_birth_date = models.DateField()
    patient_sex = models.CharField(max_length=10)

    admission_date = models.DateField()
    discharge_date = models.DateField(null=True, blank=True)

    primary_diagnosis_icd10_code = models.CharField(max_length=10)

    death_during_admission = models.BooleanField(default=False)
    intensive_care_total_days = models.PositiveSmallIntegerField(default=0)

    admission_type = models.CharField(max_length=30)

    total_amount_paid_brl = models.DecimalField(max_digits=12, decimal_places=2)


class HospitalOccupancyPrediction(models.Model):
    hospital = models.CharField(max_length=255, db_index=True)
    week_start = models.DateField(db_index=True)

    real_total = models.IntegerField(null=True, blank=True)
    estimated_total = models.FloatField()

    updated_at = models.DateTimeField(auto_now=True, db_index=True)

    class Meta:
        db_table = "hospital_occupancy_predictions"
        constraints = [
            models.UniqueConstraint(
                fields=["hospital", "week_start"],
                name="uq_hospital_week_start_current",
            )
        ]
        indexes = [
            models.Index(fields=["week_start", "hospital"], name="idx_current_week_hospital"),
        ]


class HospitalOccupancyPredictionRun(models.Model):
    """
    One row per execution (e.g., every Sunday).
    Groups all audit predictions produced in this run.
    """
    run_at = models.DateTimeField(auto_now_add=True, db_index=True)

    train_from = models.IntegerField(null=True, blank=True)
    train_to = models.IntegerField(null=True, blank=True)
    test_year = models.IntegerField(null=True, blank=True)

    model_path = models.CharField(max_length=512, blank=True, default="")
    csv_path = models.CharField(max_length=512, blank=True, default="")

    mae = models.FloatField(null=True, blank=True)
    rmse = models.FloatField(null=True, blank=True)
    r2 = models.FloatField(null=True, blank=True)

    rows_count = models.IntegerField(null=True, blank=True)

    class Meta:
        db_table = "hospital_occupancy_prediction_runs"
        indexes = [
            models.Index(fields=["run_at"], name="idx_pred_run_at"),
        ]


class HospitalOccupancyPredictionAudit(models.Model):
    """
    Historical snapshot of predictions for auditing.
    """
    run = models.ForeignKey(
        HospitalOccupancyPredictionRun,
        on_delete=models.CASCADE,
        related_name="predictions",
        db_index=True,
    )

    hospital = models.CharField(max_length=255, db_index=True)
    week_start = models.DateField(db_index=True)

    real_total = models.IntegerField(null=True, blank=True)
    estimated_total = models.FloatField()

    class Meta:
        db_table = "hospital_occupancy_predictions_audit"
        indexes = [
            models.Index(fields=["run", "hospital", "week_start"], name="idx_audit_run_hosp_week"),
            models.Index(fields=["hospital", "week_start"], name="idx_audit_hosp_week"),
        ]