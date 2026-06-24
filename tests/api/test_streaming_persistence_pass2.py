from __future__ import annotations

import asyncio
import json
import uuid
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.domain.processing import PreparedChatRequest, _ProcessChat
from gateway.schemas.openai_chat import ChatCompletionRequest


class _DummyCeleryClient:
    def __init__(self) -> None:
        self.last_kwargs: dict[str, object] | None = None

    def enqueue_generate_response(self, **kwargs):
        self.last_kwargs = kwargs
        return "task-2"


class _DoneRedisStream:
    async def subscribe(self, conversation_id: str, timeout: float):
        del conversation_id, timeout
        yield {"type": "thinking_token", "content": "plan"}
        yield {"type": "answer_token", "content": "hello"}
        yield {"type": "answer_token", "content": " world"}
        yield {
            "type": "done",
            "request_id": "req-123",
            "thinking_content": "plan",
            "answer_content": "hello world",
            "content": "<think>plan</think>\n\nhello world",
            "finish_reason": "stop",
            "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
        }


class _ErrorRedisStream:
    async def subscribe(self, conversation_id: str, timeout: float):
        del conversation_id, timeout
        yield {"type": "error", "error": "boom", "error_type": "server_error"}


def _decode_sse_payload(chunk: bytes):
    text = chunk.decode()
    assert text.startswith("data: ")
    payload = text[6:].strip()
    if payload == "[DONE]":
        return payload
    return json.loads(payload)


def _decode_named_sse_event(chunk: bytes):
    text = chunk.decode()
    lines = [line for line in text.strip().splitlines() if line]
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    return lines[0][7:], json.loads(lines[1][6:])


def test_async_streaming_emits_answer_only_chunks_and_persists_on_done() -> None:
    async def _run():
        process = _ProcessChat()
        process.init_services(
            celery_client=_DummyCeleryClient(),
            redis_stream=_DoneRedisStream(),
        )
        process._persist_exchange = AsyncMock()

        prepared = PreparedChatRequest(
            generation_payload={"model": "test-model", "messages": []},
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[],
            prompt_messages=[],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
        chat_session_id = str(uuid.uuid4())

        with patch(
            "gateway.domain.processing.get_settings",
            return_value=SimpleNamespace(gateway=SimpleNamespace(streaming_timeout=1.0)),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(
                    req,
                    user_id="user-1",
                    chat_session_id=chat_session_id,
                    request_id="req-123",
                )
                chunks = [chunk async for chunk in generator]

        return process, chunks, chat_session_id

    process, chunks, chat_session_id = asyncio.run(_run())

    assert [_decode_sse_payload(chunk) for chunk in chunks[:2]] == [
        {
            "id": "chatcmpl-req-123",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": "hello"}, "finish_reason": None}],
        },
        {
            "id": "chatcmpl-req-123",
            "object": "chat.completion.chunk",
            "choices": [{"index": 0, "delta": {"content": " world"}, "finish_reason": None}],
        },
    ]

    finish_payload = _decode_sse_payload(chunks[2])
    usage_payload = _decode_sse_payload(chunks[3])
    done_payload = _decode_sse_payload(chunks[4])

    assert finish_payload["choices"][0]["finish_reason"] == "stop"
    assert usage_payload["choices"] == []
    assert usage_payload["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }
    assert done_payload == "[DONE]"

    process._persist_exchange.assert_awaited_once()
    persist_args = process._persist_exchange.await_args.args
    assert persist_args[0].messages[0].content == "hello"
    assert (
        persist_args[1]["choices"][0]["message"]["content"] == "<think>plan</think>\n\nhello world"
    )
    assert persist_args[1]["usage"] == {
        "prompt_tokens": 11,
        "completion_tokens": 5,
        "total_tokens": 16,
    }
    assert persist_args[2] == chat_session_id


def test_async_streaming_error_emits_error_chunk_and_skips_persistence() -> None:
    async def _run():
        process = _ProcessChat()
        process.init_services(
            celery_client=_DummyCeleryClient(),
            redis_stream=_ErrorRedisStream(),
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
            "gateway.domain.processing.get_settings",
            return_value=SimpleNamespace(gateway=SimpleNamespace(streaming_timeout=1.0)),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(
                    req,
                    user_id="user-1",
                    chat_session_id=str(uuid.uuid4()),
                    request_id="req-123",
                )
                chunks = [chunk async for chunk in generator]

        return process, chunks

    process, chunks = asyncio.run(_run())

    payload = _decode_sse_payload(chunks[0])
    assert payload == {"error": {"message": "boom", "type": "server_error"}}
    process._persist_exchange.assert_not_awaited()


def test_async_rich_stream_emits_named_events_and_persists_on_done() -> None:
    async def _run():
        process = _ProcessChat()
        process.init_services(
            celery_client=_DummyCeleryClient(),
            redis_stream=_DoneRedisStream(),
        )
        process._persist_exchange = AsyncMock()

        prepared = PreparedChatRequest(
            generation_payload={"model": "test-model", "messages": []},
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[],
            prompt_messages=[],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)
        chat_session_id = str(uuid.uuid4())

        with patch(
            "gateway.domain.processing.get_settings",
            return_value=SimpleNamespace(gateway=SimpleNamespace(streaming_timeout=1.0)),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(
                    req,
                    user_id="user-1",
                    chat_session_id=chat_session_id,
                    request_id="req-123",
                    rich_stream=True,
                )
                chunks = [chunk async for chunk in generator]

        return process, chunks, chat_session_id

    process, chunks, chat_session_id = asyncio.run(_run())

    assert [_decode_named_sse_event(chunk) for chunk in chunks[:4]] == [
        ("thinking_token", {"request_id": "req-123", "content": "plan"}),
        ("answer_token", {"request_id": "req-123", "content": "hello"}),
        ("answer_token", {"request_id": "req-123", "content": " world"}),
        (
            "usage",
            {
                "request_id": "req-123",
                "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16},
            },
        ),
    ]

    done_event, done_payload = _decode_named_sse_event(chunks[4])
    assert done_event == "done"
    assert done_payload == {
        "request_id": "req-123",
        "thinking_content": "plan",
        "answer_content": "hello world",
        "content": "<think>plan</think>\n\nhello world",
        "finish_reason": "stop",
    }
    process._persist_exchange.assert_awaited_once()
    assert process._persist_exchange.await_args.args[2] == chat_session_id


def test_async_rich_stream_error_emits_named_error_event() -> None:
    async def _run():
        process = _ProcessChat()
        process.init_services(
            celery_client=_DummyCeleryClient(),
            redis_stream=_ErrorRedisStream(),
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
            "gateway.domain.processing.get_settings",
            return_value=SimpleNamespace(gateway=SimpleNamespace(streaming_timeout=1.0)),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                generator = await process.stream_chat(
                    req,
                    request_id="req-123",
                    rich_stream=True,
                )
                chunks = [chunk async for chunk in generator]

        return process, chunks

    process, chunks = asyncio.run(_run())

    event_name, payload = _decode_named_sse_event(chunks[0])
    assert event_name == "error"
    assert payload == {
        "request_id": "req-123",
        "error": "boom",
        "error_type": "server_error",
    }
    process._persist_exchange.assert_not_awaited()
