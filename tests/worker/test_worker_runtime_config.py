from __future__ import annotations

import os

os.environ.setdefault("PLATFORM__CELERY_BROKER_URL", "amqp://guest:guest@localhost:5672//")

from gateway.services.celery_client import CeleryClient
from shared.config import load_settings


def test_celery_client_emits_sent_events_for_flower() -> None:
    client = CeleryClient("amqp://guest:guest@localhost:5672//")

    app = client._get_app()

    assert app.conf.task_send_sent_event is True

    client.close()


def test_worker_settings_default_to_observable_prefork_runtime() -> None:
    settings = load_settings(
        overrides={"platform": {"celery_broker_url": "amqp://guest:guest@localhost:5672//"}}
    )
    worker = settings.worker

    assert worker.pool == "prefork"
    assert worker.concurrency == 2
    assert worker.send_task_events is True
    assert worker.cancel_long_running_tasks_on_connection_loss is True
    assert settings.platform.celery_broker_url == "amqp://guest:guest@localhost:5672//"


def test_worker_settings_ignore_runtime_nested_env_names(monkeypatch) -> None:
    monkeypatch.setenv("WORKER__CONCURRENCY", "4")

    settings = load_settings(
        overrides={"platform": {"celery_broker_url": "amqp://guest:guest@localhost:5672//"}}
    )

    assert settings.worker.concurrency == 2
