import os

from celery import Celery
from celery.schedules import crontab

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

app = Celery("config")
app.config_from_object("django.conf:settings", namespace="CELERY")
app.autodiscover_tasks()

app.conf.beat_schedule = {
    "run-hospital-occupancy-forecast-every-sunday": {
        "task": "sih.tasks.run_hospital_occupancy_forecast",
        "schedule": crontab(minute=0, hour=3, day_of_week=0),
        "kwargs": {
            "horizon_days": 90,
        },
    }
}