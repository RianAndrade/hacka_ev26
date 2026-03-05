# arbovirus/task.py

import logging
from celery import shared_task

LOGGER = logging.getLogger(__name__)


@shared_task
def celery_test_task() -> str:
    LOGGER.info("Celery test task executed")
    return "celery is working"