from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from ui.client import GatewayClient


class _FakeResponse:
    def __init__(self, *, lines=None, headers=None, json_data=None):
        self._lines = list(lines or [])
        self.headers = headers or {}
        self._json_data = json_data or {}
        self.closed = False

    def raise_for_status(self) -> None:
        return None

    def iter_lines(self, decode_unicode=True):
        del decode_unicode
        yield from self._lines

    def json(self):
        return self._json_data

    def close(self) -> None:
        self.closed = True


class _FakeSession:
    def __init__(self, *, post_response: _FakeResponse, get_response: _FakeResponse | None = None):
        self.post_response = post_response
        self.get_response = get_response or _FakeResponse(json_data={})
        self.post_calls = []
        self.get_calls = []
        self.headers = {}

    def post(self, url, **kwargs):
        self.post_calls.append((url, kwargs))
        return self.post_response

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self.get_response


def _make_client(fake_session: _FakeSession) -> GatewayClient:
    with patch("ui.client.requests.Session", return_value=fake_session):
        with patch(
            "ui.client.get_ui_settings",
            return_value=SimpleNamespace(chat_timeout=30.0, health_timeout=5.0, models_timeout=5.0),
        ):
            return GatewayClient("http://gateway.test")


def test_chat_stream_parses_rich_named_events_and_sets_header() -> None:
    fake_response = _FakeResponse(
        headers={"X-Request-Id": "req-123"},
        lines=[
            "event: thinking_token",
            'data: {"request_id": "req-123", "content": "plan"}',
            "",
            "event: answer_token",
            'data: {"request_id": "req-123", "content": "hello"}',
            "",
            "event: done",
            'data: {"request_id": "req-123", "content": "<think>plan</think>\\n\\nhello"}',
            "",
        ],
    )
    fake_session = _FakeSession(post_response=fake_response)
    client = _make_client(fake_session)

    stream = client.chat_stream({"messages": [{"role": "user", "content": "hi"}]}, rich_stream=True)
    events = list(stream.events)

    assert stream.request_id == "req-123"
    assert [(event.event, event.data) for event in events] == [
        ("thinking_token", {"request_id": "req-123", "content": "plan"}),
        ("answer_token", {"request_id": "req-123", "content": "hello"}),
        ("done", {"request_id": "req-123", "content": "<think>plan</think>\n\nhello"}),
    ]
    assert fake_session.post_calls[0][1]["headers"] == {"X-UI-Rich-Stream": "1"}
    assert fake_response.closed is True


def test_chat_stream_parses_standard_sse_and_done_marker() -> None:
    fake_response = _FakeResponse(
        headers={"X-Request-Id": "req-456"},
        lines=[
            'data: {"choices": [{"delta": {"content": "hello"}, "finish_reason": null}]}',
            "",
            "data: [DONE]",
            "",
        ],
    )
    fake_session = _FakeSession(post_response=fake_response)
    client = _make_client(fake_session)

    stream = client.chat_stream({"messages": [{"role": "user", "content": "hi"}]})
    events = list(stream.events)

    assert [(event.event, event.data) for event in events] == [
        ("message", {"choices": [{"delta": {"content": "hello"}, "finish_reason": None}]}),
        ("done_marker", "[DONE]"),
    ]
    assert fake_session.post_calls[0][1]["headers"] is None


def test_get_prompt_preview_calls_route_and_returns_json() -> None:
    fake_session = _FakeSession(
        post_response=_FakeResponse(),
        get_response=_FakeResponse(
            json_data={
                "request_id": "req-123",
                "prompt_messages": [{"role": "system", "content": "hi"}],
            }
        ),
    )
    client = _make_client(fake_session)

    preview = client.get_prompt_preview("req-123")

    assert preview == {
        "request_id": "req-123",
        "prompt_messages": [{"role": "system", "content": "hi"}],
    }
    assert fake_session.get_calls[0][0].endswith("/v1/chat/prompt-preview/req-123")
