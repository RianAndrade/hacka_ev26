from celery import shared_task
from arbovirus.ml.train_dengue_model import train_and_evaluate


@shared_task
def train_dengue_model_task(municipality_code: str):
    result = train_and_evaluate(municipality_code)
    return result