from __future__ import annotations

import asyncio
import json
import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

os.environ.setdefault("CELERY_BROKER_URL", "amqp://guest:guest@localhost//")

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import PreparedChatRequest, _ProcessChat
from shared.vllm_payloads import canonicalize_assistant_content
from worker.tasks import (
    EVENT_ANSWER_TOKEN,
    EVENT_THINKING_TOKEN,
    _build_done_event,
    _merge_usage,
    _ThinkTagStreamParser,
)


class _DummyCeleryClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None
        self.revoke_calls: list[dict[str, object]] = []

    def enqueue_generate_response(self, **kwargs):
        self.last_kwargs = kwargs
        return "task-1"

    def revoke_task(self, task_id: str, **kwargs) -> None:
        self.revoke_calls.append({"task_id": task_id, **kwargs})


class _DummyRedisStream:
    def __init__(self, celery_client: _DummyCeleryClient) -> None:
        self._celery_client = celery_client

    async def store_prompt_preview(
        self, request_id: str, preview: dict, *, ttl_seconds: int
    ) -> None:
        del request_id, preview, ttl_seconds

    async def subscribe(self, conversation_id: str, timeout: float):
        del conversation_id, timeout
        request_id = str(self._celery_client.last_kwargs["request_id"])
        yield {"type": EVENT_THINKING_TOKEN, "content": "plan"}
        yield {"type": EVENT_ANSWER_TOKEN, "content": "answer"}
        yield {
            "type": "done",
            "request_id": request_id,
            "thinking_content": "plan",
            "answer_content": "answer",
            "content": canonicalize_assistant_content("plan", "answer"),
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }


class _TimeoutRedisStream:
    async def store_prompt_preview(
        self, request_id: str, preview: dict, *, ttl_seconds: int
    ) -> None:
        del request_id, preview, ttl_seconds

    async def subscribe(self, conversation_id: str, timeout: float):
        del conversation_id, timeout
        yield {
            "type": "error",
            "error": "Timeout waiting for response",
            "error_type": "timeout",
        }


class _OneTokenRedisStream:
    async def store_prompt_preview(
        self, request_id: str, preview: dict, *, ttl_seconds: int
    ) -> None:
        del request_id, preview, ttl_seconds

    async def subscribe(self, conversation_id: str, timeout: float):
        del conversation_id, timeout
        yield {"type": EVENT_ANSWER_TOKEN, "content": "partial"}
        await asyncio.sleep(3600)


def _decode_sse_payload(chunk: bytes) -> object:
    text = chunk.decode()
    assert text.startswith("data: ")
    payload = text[6:].strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


def test_merge_usage_preserves_non_null_values() -> None:
    merged = _merge_usage(
        {"prompt_tokens": 12, "completion_tokens": None, "total_tokens": None},
        {"prompt_tokens": None, "completion_tokens": 5, "total_tokens": None},
    )

    assert merged == {"prompt_tokens": 12, "completion_tokens": 5, "total_tokens": 17}


def test_think_tag_parser_splits_thinking_and_answer_stream() -> None:
    parser = _ThinkTagStreamParser()
    events = []
    for chunk in ["<thi", "nk>pla", "n</th", "ink>ans", "wer"]:
        events.extend(parser.feed(chunk))
    events.extend(parser.flush())

    thinking = "".join(
        content for event_type, content in events if event_type == EVENT_THINKING_TOKEN
    )
    answer = "".join(content for event_type, content in events if event_type == EVENT_ANSWER_TOKEN)

    assert thinking == "plan"
    assert answer == "answer"


def test_done_event_includes_request_id_and_canonical_content() -> None:
    event = _build_done_event(
        request_id="req-123",
        thinking_content="plan",
        answer_content="answer",
        finish_reason=None,
        task_id="task-1",
        usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": None},
    )

    assert event["request_id"] == "req-123"
    assert event["content"] == "<think>plan</think>\n\nanswer"
    assert event["usage"] == {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18}


def test_async_chat_threads_request_id_and_new_event_types() -> None:
    async def _run() -> tuple[list[bytes], _DummyCeleryClient]:
        process = _ProcessChat()
        celery_client = _DummyCeleryClient()
        redis_stream = _DummyRedisStream(celery_client)
        process.init_services(celery_client=celery_client, redis_stream=redis_stream)

        prepared = PreparedChatRequest(
            generation_payload={"model": "test-model", "messages": []},
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[],
            prompt_messages=[],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}])

        with patch(
            "gateway.services.processing.get_settings",
            return_value=SimpleNamespace(streaming_timeout=1.0),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                stream = await process.stream_chat(req, request_id="req-from-route")
                result = [chunk async for chunk in stream]

        return result, celery_client

    chunks, celery_client = asyncio.run(_run())

    assert celery_client.last_kwargs is not None
    assert celery_client.last_kwargs["request_id"] == "req-from-route"
    assert chunks[-1] == b"data: [DONE]\n\n"
    assert celery_client.revoke_calls == []


def test_async_stream_timeout_revokes_stalled_task() -> None:
    async def _run():
        process = _ProcessChat()
        celery_client = _DummyCeleryClient()
        process.init_services(
            celery_client=celery_client,
            redis_stream=_TimeoutRedisStream(),
        )
        process._persist_exchange = AsyncMock()

        prepared = PreparedChatRequest(
            generation_payload={"model": "test-model", "messages": []},
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[],
            prompt_messages=[],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)

        with patch(
            "gateway.services.processing.get_settings",
            return_value=SimpleNamespace(streaming_timeout=1.0),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(req, request_id="req-timeout")
                chunks = [chunk async for chunk in generator]

        return celery_client, chunks, process

    celery_client, chunks, process = asyncio.run(_run())

    assert [_decode_sse_payload(chunk) for chunk in chunks] == [
        {"error": {"message": "Timeout waiting for response", "type": "timeout"}}
    ]
    assert celery_client.revoke_calls == [
        {"task_id": "task-1", "terminate": True, "signal": "SIGTERM"}
    ]
    process._persist_exchange.assert_not_awaited()


def test_async_stream_close_revokes_inflight_task() -> None:
    async def _run():
        process = _ProcessChat()
        celery_client = _DummyCeleryClient()
        process.init_services(
            celery_client=celery_client,
            redis_stream=_OneTokenRedisStream(),
        )

        prepared = PreparedChatRequest(
            generation_payload={"model": "test-model", "messages": []},
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[],
            prompt_messages=[],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)

        with patch(
            "gateway.services.processing.get_settings",
            return_value=SimpleNamespace(streaming_timeout=1.0),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(req, request_id="req-close")
                first_chunk = await anext(generator)
                await generator.aclose()

        return celery_client, first_chunk

    celery_client, first_chunk = asyncio.run(_run())

    assert _decode_sse_payload(first_chunk) == {
        "id": "chatcmpl-req-close",
        "object": "chat.completion.chunk",
        "choices": [{"index": 0, "delta": {"content": "partial"}, "finish_reason": None}],
    }
    assert celery_client.revoke_calls == [
        {"task_id": "task-1", "terminate": True, "signal": "SIGTERM"}
    ]


def test_redis_stream_timeout_tracks_idle_time_not_total_runtime() -> None:
    from gateway.services.redis_stream import RedisStreamService

    class _FakeLoop:
        def __init__(self) -> None:
            self.now = 0.0

        def time(self) -> float:
            return self.now

    class _FakePubSub:
        def __init__(self, loop: _FakeLoop) -> None:
            self._loop = loop
            self._steps = [
                (0.9, {"type": "message", "data": json.dumps({"type": "thinking_token"})}),
                (0.2, None),
                (0.7, {"type": "message", "data": json.dumps({"type": "done"})}),
            ]

        async def subscribe(self, channel_name: str) -> None:
            del channel_name

        async def get_message(self, ignore_subscribe_messages: bool, timeout: float):
            del ignore_subscribe_messages, timeout
            delta, message = self._steps.pop(0)
            self._loop.now += delta
            return message

        async def unsubscribe(self, channel_name: str) -> None:
            del channel_name

        async def close(self) -> None:
            return None

    class _FakeRedis:
        def __init__(self, pubsub: _FakePubSub) -> None:
            self._pubsub = pubsub

        def pubsub(self) -> _FakePubSub:
            return self._pubsub

    async def _no_sleep(_: float) -> None:
        return None

    async def _run() -> list[dict[str, object]]:
        loop = _FakeLoop()
        pubsub = _FakePubSub(loop)
        service = RedisStreamService("redis://test")

        with (
            patch(
                "gateway.services.redis_stream.aioredis.from_url",
                return_value=_FakeRedis(pubsub),
            ),
            patch("gateway.services.redis_stream._monotonic_time", side_effect=loop.time),
            patch("gateway.services.redis_stream.asyncio.sleep", new=_no_sleep),
        ):
            return [event async for event in service.subscribe("conv-1", timeout=1.0)]

    events = asyncio.run(_run())

    assert [event["type"] for event in events] == ["thinking_token", "done"]
