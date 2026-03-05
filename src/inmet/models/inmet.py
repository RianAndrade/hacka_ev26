from django.db import models


class InmetWeatherObservation(models.Model):
    """
    Represents an INMET weather observation.

    Fields meaning:
    - observation_date: Date of the meteorological observation (day/month/year in source).
    - observation_time_utc: Time of the measurement in Coordinated Universal Time (UTC).
    - air_temperature_instant_celsius: Instantaneous air temperature at reading time (°C).
    - air_temperature_max_celsius: Maximum air temperature in the interval (°C).
    - air_temperature_min_celsius: Minimum air temperature in the interval (°C).
    - relative_humidity_instant_percent: Instantaneous relative humidity at observation time (%).
    - relative_humidity_max_percent: Maximum relative humidity in the interval (%).
    - relative_humidity_min_percent: Minimum relative humidity in the interval (%).
    - dew_point_temperature_instant_celsius: Instantaneous dew point temperature at reading time (°C).
    - dew_point_temperature_max_celsius: Maximum dew point temperature in the interval (°C).
    - dew_point_temperature_min_celsius: Minimum dew point temperature in the interval (°C).
    - atmospheric_pressure_instant_hpa: Instantaneous atmospheric pressure at station level (hPa).
    - atmospheric_pressure_max_hpa: Maximum atmospheric pressure in the interval (hPa).
    - atmospheric_pressure_min_hpa: Minimum atmospheric pressure in the interval (hPa).
    - wind_speed_mean_meters_per_second: Mean wind speed in the interval (m/s).
    - wind_direction_degrees: Predominant wind direction (azimuth degrees).
    - wind_gust_max_meters_per_second: Maximum instantaneous wind speed (gust) in the interval (m/s).
    - global_solar_radiation_kj_per_square_meter: Accumulated global solar radiation in the interval (kJ/m²).
    - precipitation_accumulated_millimeters: Accumulated precipitation in the interval (mm).
    """
    observation_date = models.DateField(
        null=True
    )

    observation_time_utc = models.TimeField(
        null=True
    )

    air_temperature_instant_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    air_temperature_max_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    air_temperature_min_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    relative_humidity_instant_percent = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    relative_humidity_max_percent = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    relative_humidity_min_percent = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    dew_point_temperature_instant_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    dew_point_temperature_max_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    dew_point_temperature_min_celsius = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    atmospheric_pressure_instant_hpa = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True
    )

    atmospheric_pressure_max_hpa = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True
    )

    atmospheric_pressure_min_hpa = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True
    )

    wind_speed_mean_meters_per_second = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    wind_direction_degrees = models.PositiveSmallIntegerField(
        null=True
    )

    wind_gust_max_meters_per_second = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True
    )

    global_solar_radiation_kj_per_square_meter = models.DecimalField(
        max_digits=9,
        decimal_places=2,
        null=True
    )

    precipitation_accumulated_millimeters = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True
    )

    class Meta:
        db_table = "inmet_weather_observation"