from django.db import models


class DengueMonthlyDataset(models.Model):

    month = models.DateField(
        help_text="First day of the month (YYYY-MM-01)."
    )

    municipality_code = models.CharField(
        max_length=20,
        null=True
    )

    health_region_code = models.CharField(
        max_length=20,
        null=True
    )

    temp_mean = models.FloatField(
        null=True
    )

    temp_max_mean = models.FloatField(
        null=True
    )

    temp_min_mean = models.FloatField(
        null=True
    )

    humidity_mean = models.FloatField(
        null=True
    )

    humidity_max_mean = models.FloatField(
        null=True
    )

    humidity_min_mean = models.FloatField(
        null=True
    )

    rain_sum = models.FloatField(
        null=True
    )

    pressure_mean = models.FloatField(
        null=True
    )

    wind_speed_mean = models.FloatField(
        null=True
    )

    solar_radiation_sum = models.FloatField(
        null=True
    )

    optimal_days = models.PositiveSmallIntegerField(
        null=True
    )

    dengue_cases = models.PositiveIntegerField(
        null=True
    )

    dengue_cases_next = models.PositiveIntegerField(
        null=True
    )

    class Meta:
        db_table = "dengue_monthly_dataset"
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "month",
                    "municipality_code",
                    "health_region_code",
                ],
                name="uniq_dengue_monthly_dataset_month_city_region",
            ),
        ]
        indexes = [
            models.Index(
                fields=[
                    "month",
                ],
                name="idx_dengue_monthly_month",
            ),
            models.Index(
                fields=[
                    "municipality_code",
                    "month",
                ],
                name="idx_dengue_monthly_city_month",
            ),
        ]