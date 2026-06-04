from __future__ import annotations

import logging
import uuid as _uuid
from typing import Any

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from gateway.schemas.openai_chat import ChatCompletionRequest
from gateway.services.budget import BudgetValidationError
from gateway.services.processing import process_chat
from shared.catalog import get_kb_config
from shared.vllm_payloads import ResponseBudgetExceededError

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
    rich_stream = request.headers.get("x-ui-rich-stream") == "1"

    try:
        if not payload.stream:
            raise HTTPException(
                status_code=400,
                detail="Successful chat generation requires stream=true.",
            )

        request_id = str(_uuid.uuid4())
        generator = await process_chat.stream_chat(
            payload,
            user_id=user_id,
            chat_session_id=chat_session_id,
            request_id=request_id,
            rich_stream=rich_stream,
        )
        return StreamingResponse(
            generator,
            media_type="text/event-stream",
            headers={
                "X-Request-Id": request_id,
                "Cache-Control": "no-cache",
            },
        )
    except (BudgetValidationError, ResponseBudgetExceededError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/chat/prompt-preview/{request_id}")
async def get_prompt_preview(request_id: str, request: Request) -> Any:
    user_id: str | None = getattr(request.state, "user_id", None)
    preview = await process_chat.get_prompt_preview(
        request_id,
        requester_user_id=user_id,
    )
    if preview is None:
        raise HTTPException(status_code=404, detail="Prompt preview not found")
    return preview
