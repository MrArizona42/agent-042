from __future__ import annotations

import os

os.environ.setdefault("CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")

from gateway.services.celery_client import CeleryClient
from worker.config import WorkerSettings


def test_celery_client_emits_sent_events_for_flower() -> None:
    client = CeleryClient("amqp://guest:guest@localhost:5672//")

    app = client._get_app()

    assert app.conf.task_send_sent_event is True

    client.close()


def test_worker_settings_default_to_observable_prefork_runtime() -> None:
    settings = WorkerSettings(CELERY_BROKER_URL="amqp://guest:guest@localhost:5672//")

    assert settings.worker_pool == "prefork"
    assert settings.worker_concurrency == 2
    assert settings.worker_send_task_events is True
    assert settings.worker_cancel_long_running_tasks_on_connection_loss is True
