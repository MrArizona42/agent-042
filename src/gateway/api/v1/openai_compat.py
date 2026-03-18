from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.processing import process_chat
from shared.config import get_knowledge_bases

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
        kb_registry = get_knowledge_bases()
        for src in payload.rag_sources:
            if src.knowledge_base not in kb_registry:
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge base '{src.knowledge_base}' unavailable",
                )
            kb_cfg = kb_registry[src.knowledge_base]
            if src.alias not in kb_cfg.aliases:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Alias '{src.alias}' is not valid for "
                        f"knowledge base '{src.knowledge_base}'"
                    ),
                )

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

