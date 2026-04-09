from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import (
    PROMPT_PREVIEW_TTL_SECONDS,
    SERVICE_USER_ID,
    PreparedChatRequest,
    _ProcessChat,
)


class _PreviewRedisStream:
    def __init__(self) -> None:
        self.stored: tuple[str, dict, int] | None = None
        self.preview: dict | None = None

    async def store_prompt_preview(
        self, request_id: str, preview: dict, *, ttl_seconds: int
    ) -> None:
        self.stored = (request_id, preview, ttl_seconds)

    async def get_prompt_preview(self, request_id: str) -> dict | None:
        del request_id
        return self.preview


async def _empty_stream():
    if False:
        yield b""


def test_stream_chat_stores_prompt_preview_before_stream_start() -> None:
    async def _run():
        process = _ProcessChat()
        redis_stream = _PreviewRedisStream()
        process.init_services(redis_stream=redis_stream, celery_client=object())

        prepared = PreparedChatRequest(
            generation_payload={
                "model": "test-model",
                "messages": [{"role": "system", "content": "hi"}],
            },
            budget_meta={"model_max_tokens": 64, "budget_guard": 8, "min_response_budget": 4},
            rag_context_chunks=[{"knowledge_base": "arxiv", "content": "doc"}],
            prompt_messages=[{"role": "system", "content": "hi"}],
        )
        req = ChatCompletionRequest(messages=[{"role": "user", "content": "hello"}], stream=True)

        with patch(
            "gateway.services.processing.get_settings",
            return_value=SimpleNamespace(),
        ):
            with patch.object(process, "_prepare_request", return_value=prepared):
                with patch.object(
                    process, "_stream_chat_async", new=AsyncMock(return_value=_empty_stream())
                ):
                    await process.stream_chat(
                        req,
                        user_id="user-1",
                        chat_session_id="session-1",
                        request_id="req-123",
                    )

        return redis_stream

    redis_stream = asyncio.run(_run())

    assert redis_stream.stored is not None
    request_id, preview, ttl_seconds = redis_stream.stored
    assert request_id == "req-123"
    assert ttl_seconds == PROMPT_PREVIEW_TTL_SECONDS
    assert preview["owner_user_id"] == "user-1"
    assert preview["chat_session_id"] == "session-1"
    assert preview["model"] == "test-model"
    assert preview["prompt_messages"] == [{"role": "system", "content": "hi"}]
    assert preview["rag_context"] == [{"knowledge_base": "arxiv", "content": "doc"}]


def test_get_prompt_preview_returns_sanitized_preview_for_owner() -> None:
    async def _run():
        process = _ProcessChat()
        redis_stream = _PreviewRedisStream()
        redis_stream.preview = {
            "request_id": "req-123",
            "owner_user_id": "user-1",
            "chat_session_id": "session-1",
            "model": "test-model",
            "prompt_messages": [{"role": "system", "content": "hi"}],
            "rag_context": [],
        }
        process.init_services(redis_stream=redis_stream, celery_client=object())
        return await process.get_prompt_preview("req-123", requester_user_id="user-1")

    preview = asyncio.run(_run())

    assert preview == {
        "request_id": "req-123",
        "chat_session_id": "session-1",
        "model": "test-model",
        "prompt_messages": [{"role": "system", "content": "hi"}],
        "rag_context": [],
    }


def test_get_prompt_preview_allows_service_user_and_blocks_other_users() -> None:
    async def _run(requester_user_id: str):
        process = _ProcessChat()
        redis_stream = _PreviewRedisStream()
        redis_stream.preview = {
            "request_id": "req-123",
            "owner_user_id": "user-1",
            "chat_session_id": "session-1",
            "model": "test-model",
            "prompt_messages": [],
            "rag_context": [],
        }
        process.init_services(redis_stream=redis_stream, celery_client=object())
        return await process.get_prompt_preview("req-123", requester_user_id=requester_user_id)

    assert asyncio.run(_run(SERVICE_USER_ID)) is not None
    assert asyncio.run(_run("user-2")) is None
