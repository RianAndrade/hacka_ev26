from django.db import models


class DengueWeeklyDataset(models.Model):

    week_start = models.DateField()

    epidemiological_year = models.PositiveSmallIntegerField(
        null=True
    )

    epidemiological_week = models.PositiveSmallIntegerField(
        null=True
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
        db_table = "dengue_weekly_dataset"
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "week_start",
                    "municipality_code",
                    "health_region_code",
                ],
                name="uq_dw_wk_city_reg",
            ),
        ]
        indexes = [
            models.Index(
                fields=["week_start"],
                name="ix_dw_week",
            ),
            models.Index(
                fields=["epidemiological_year", "epidemiological_week"],
                name="ix_dw_epi",
            ),
            models.Index(
                fields=["municipality_code", "week_start"],
                name="ix_dw_city_week",
            ),
        ]
