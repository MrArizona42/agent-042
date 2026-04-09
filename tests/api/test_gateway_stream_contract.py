from __future__ import annotations

from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_test_app():
    from gateway.api.v1 import openai_compat

    app = FastAPI()
    app.include_router(openai_compat.router, prefix="/v1")
    return app


async def _done_only_stream():
    yield b"data: [DONE]\n\n"


def test_stream_false_is_rejected_before_process_chat_stream() -> None:
    from gateway.api.v1 import openai_compat

    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(openai_compat.process_chat, "stream_chat", new=AsyncMock()) as stream_mock:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": False,
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Successful chat generation requires stream=true."
    stream_mock.assert_not_called()


def test_streaming_response_sets_request_id_header() -> None:
    from gateway.api.v1 import openai_compat

    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        openai_compat.process_chat,
        "stream_chat",
        new=AsyncMock(return_value=_done_only_stream()),
    ) as stream_mock:
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    request_id = response.headers.get("X-Request-Id")
    assert request_id
    assert response.headers.get("Cache-Control") == "no-cache"
    assert stream_mock.await_args.kwargs["request_id"] == request_id
    assert stream_mock.await_args.kwargs["rich_stream"] is False


def test_rich_stream_header_enables_opt_in_stream_mode() -> None:
    from gateway.api.v1 import openai_compat

    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        openai_compat.process_chat,
        "stream_chat",
        new=AsyncMock(return_value=_done_only_stream()),
    ) as stream_mock:
        response = client.post(
            "/v1/chat/completions",
            headers={"X-UI-Rich-Stream": "1"},
            json={
                "messages": [{"role": "user", "content": "hello"}],
                "stream": True,
            },
        )

    assert response.status_code == 200
    assert stream_mock.await_args.kwargs["rich_stream"] is True


def test_prompt_preview_route_returns_preview_payload() -> None:
    from gateway.api.v1 import openai_compat

    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        openai_compat.process_chat,
        "get_prompt_preview",
        new=AsyncMock(
            return_value={
                "request_id": "req-123",
                "model": "test-model",
                "prompt_messages": [{"role": "system", "content": "hi"}],
                "rag_context": [],
            }
        ),
    ) as preview_mock:
        response = client.get("/v1/chat/prompt-preview/req-123")

    assert response.status_code == 200
    assert response.json()["request_id"] == "req-123"
    assert response.json()["prompt_messages"][0]["role"] == "system"
    preview_mock.assert_awaited_once_with("req-123", requester_user_id=None)


def test_prompt_preview_route_returns_404_when_missing() -> None:
    from gateway.api.v1 import openai_compat

    app = _make_test_app()
    client = TestClient(app, raise_server_exceptions=False)

    with patch.object(
        openai_compat.process_chat,
        "get_prompt_preview",
        new=AsyncMock(return_value=None),
    ):
        response = client.get("/v1/chat/prompt-preview/missing")

    assert response.status_code == 404
    assert response.json()["detail"] == "Prompt preview not found"
