from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import process_chat

router = APIRouter()


@router.get("/models")
async def list_models() -> Any:
    # proxy to vLLM /v1/models
    return await process_chat.list_models()


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest) -> Any:
    if payload.stream:
        generator = process_chat.stream_chat(payload)
        return StreamingResponse(generator, media_type="text/event-stream")

    return await process_chat.chat(payload)

