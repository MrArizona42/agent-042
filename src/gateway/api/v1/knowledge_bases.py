"""Admin endpoints for knowledge-base discovery and config reload."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from shared.config import get_settings
from gateway.services.processing import process_chat
from gateway.services.rag_service import RAGService
from shared.config import clear_knowledge_base_caches

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/knowledge-bases")
async def list_knowledge_bases() -> list[dict[str, Any]]:
    """List available knowledge bases grouped by task.

    Returns a JSON array with one entry per task and nested KB metadata.
    """
    return RAGService.available_knowledge_bases_by_task()


@router.post("/admin/reload-config")
async def reload_config(request: Request) -> dict[str, str]:
    """Hot-reload knowledge-base config from disk.

    Clears the KB registry / index caches and invalidates any cached
    retrievers and build configs so the next request re-reads everything.

    Requires an authenticated user session.  When auth is disabled the
    endpoint returns 503 to avoid accidental public access.
    """
    if getattr(request.app.state, "session_manager", None) is None:
        raise HTTPException(
            status_code=503,
            detail="Config reload is unavailable when auth is disabled",
        )

    if getattr(request.state, "session_id", None) is None:
        raise HTTPException(
            status_code=403,
            detail="Config reload requires an authenticated user session",
        )

    # Auth middleware guarantees user_id on request.state for session-authenticated routes
    _ = request.state.user_id

    clear_knowledge_base_caches()
    process_chat.reload_config_caches(settings=get_settings())

    logger.info("Knowledge-base config reloaded by user")
    return {"status": "reloaded"}
