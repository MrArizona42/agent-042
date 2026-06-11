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
from shared.logging import bind_log_context, reset_log_context
from shared.vllm_payloads import ResponseBudgetExceededError

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/models")
async def list_models() -> Any:
    # proxy to vLLM /v1/models
    return await process_chat.list_models()


@router.post("/chat/completions")
async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Any:
    request_id = str(_uuid.uuid4())
    user_id: str | None = getattr(request.state, "user_id", None)
    chat_session_id = payload.chat_session_id or request.headers.get("x-chat-session-id")
    rich_stream = request.headers.get("x-ui-rich-stream") == "1"
    # Validate rag_sources before processing
    if payload.rag_sources:
        for src in payload.rag_sources:
            kb_cfg = get_kb_config(src.knowledge_base)
            if kb_cfg is None:
                process_chat.publish_inference_event(
                    "chat.request.rejected",
                    request_id=request_id,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    payload={"reason": "unknown_knowledge_base"},
                )
                raise HTTPException(
                    status_code=404,
                    detail=f"Knowledge base '{src.knowledge_base}' unavailable",
                )
            effective_alias = src.alias or kb_cfg.default_alias
            if effective_alias not in kb_cfg.aliases:
                process_chat.publish_inference_event(
                    "chat.request.rejected",
                    request_id=request_id,
                    user_id=user_id,
                    chat_session_id=chat_session_id,
                    payload={"reason": "unknown_knowledge_base_alias"},
                )
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Alias '{effective_alias}' is not valid for "
                        f"knowledge base '{src.knowledge_base}'"
                    ),
                )

    log_token = bind_log_context(
        request_id=request_id,
        user_id=user_id,
        chat_session_id=chat_session_id,
        route="/v1/chat/completions",
    )

    try:
        logger.info(
            "Chat completion request received",
            extra={
                "event": "chat.request.received",
                "stream": payload.stream,
                "message_count": len(payload.messages),
                "rag_sources_count": len(payload.rag_sources or ()),
            },
        )
        if not payload.stream:
            process_chat.publish_inference_event(
                "chat.request.rejected",
                request_id=request_id,
                user_id=user_id,
                chat_session_id=chat_session_id,
                payload={"reason": "stream_required"},
            )
            logger.info(
                "Chat completion request rejected because stream=false",
                extra={"event": "chat.request.rejected", "reason": "stream_required"},
            )
            raise HTTPException(
                status_code=400,
                detail="Successful chat generation requires stream=true.",
            )

        process_chat.publish_inference_event(
            "chat.request.accepted",
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
            model=payload.model,
            payload={
                "message_count": len(payload.messages),
                "rag_sources_count": len(payload.rag_sources or ()),
                "rich_stream": rich_stream,
            },
        )
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
        process_chat.publish_inference_event(
            "chat.request.rejected",
            request_id=request_id,
            user_id=user_id,
            chat_session_id=chat_session_id,
            model=payload.model,
            payload={"reason": type(exc).__name__},
        )
        logger.info(
            "Chat completion request rejected by budget validation",
            extra={"event": "chat.request.rejected", "reason": type(exc).__name__},
        )
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        reset_log_context(log_token)


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
