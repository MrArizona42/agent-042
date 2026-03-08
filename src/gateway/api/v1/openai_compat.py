from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import process_chat

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models() -> Any:
    # proxy to vLLM /v1/models
    return await process_chat.list_models()


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
    # Pass user_id and chat_session_id from the header / request state
    user_id: str | None = getattr(request.state, "user_id", None)
    chat_session_id = (
        payload.chat_session_id
        or request.headers.get("x-chat-session-id")
    )

    if payload.stream:
        generator = process_chat.stream_chat(
            payload, user_id=user_id, chat_session_id=chat_session_id
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    return await process_chat.chat(
        payload, user_id=user_id, chat_session_id=chat_session_id
    )

