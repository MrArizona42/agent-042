from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import process_chat
from shared.config import get_kb_config

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models() -> Any:
    # proxy to vLLM /v1/models
    return await process_chat.list_models()


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
    # Validate rag_sources before processing
    if payload.rag_sources:
        for src in payload.rag_sources:
            kb_cfg = get_kb_config(src.knowledge_base)
            if kb_cfg is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge base '{src.knowledge_base}' unavailable",
                )
            effective_alias = src.alias or kb_cfg.default_alias
            if effective_alias not in kb_cfg.aliases:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Alias '{effective_alias}' is not valid for "
                        f"knowledge base '{src.knowledge_base}'"
                    ),
                )

    # Pass user_id and chat_session_id from the header / request state
    user_id: str | None = getattr(request.state, "user_id", None)
    chat_session_id = payload.chat_session_id or request.headers.get("x-chat-session-id")

    if payload.stream:
        generator = process_chat.stream_chat(
            payload, user_id=user_id, chat_session_id=chat_session_id
        )
        return StreamingResponse(generator, media_type="text/event-stream")

    return await process_chat.chat(payload, user_id=user_id, chat_session_id=chat_session_id)
