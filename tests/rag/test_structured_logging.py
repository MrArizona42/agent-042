from __future__ import annotations

import json
import logging

from clients.observability.logging import (
    bind_log_context,
    clear_log_context,
    configure_logging,
    reset_log_context,
)


def test_configure_logging_emits_json_with_service_and_context(capsys) -> None:
    configure_logging(service="gateway", level="INFO")
    token = bind_log_context(
        request_id="req-123",
        user_id="user-raw-uuid",
        chat_session_id="session-123",
    )
    try:
        logging.getLogger("tests.logging").info(
            "request prepared",
            extra={"event": "chat.request.prepared"},
        )
    finally:
        reset_log_context(token)

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["service"] == "gateway"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "tests.logging"
    assert payload["message"] == "request prepared"
    assert payload["event"] == "chat.request.prepared"
    assert payload["request_id"] == "req-123"
    assert payload["user_id"] == "user-raw-uuid"
    assert payload["chat_session_id"] == "session-123"
    assert "environment" not in payload


def test_clear_log_context_removes_bound_fields(capsys) -> None:
    configure_logging(service="worker", level="INFO")
    bind_log_context(request_id="req-456")
    clear_log_context()

    logging.getLogger("tests.logging").info("done")

    captured = capsys.readouterr()
    payload = json.loads(captured.out)

    assert payload["service"] == "worker"
    assert "request_id" not in payload
