"""Admin endpoints for knowledge-base discovery and config reload."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Request

from gateway.services.processing import process_chat
from gateway.services.rag_service import RAGService
from shared.config import clear_knowledge_base_caches

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/knowledge-bases")
async def list_knowledge_bases() -> list[dict[str, Any]]:
    """List available knowledge bases, their aliases, and update strategies.

    Returns a JSON array with one entry per knowledge base.
    """
    result: list[dict[str, Any]] = []
    for kb_name, info in RAGService.available_knowledge_bases().items():
        result.append(
            {
                "knowledge_base": kb_name,
                "label": info.get("label", ""),
                "description": info.get("description", ""),
                "aliases": info.get("aliases", {}),
                "default_alias": info.get("default_alias", ""),
                "update_strategy": info.get("update_strategy", "replace"),
            }
        )
    return result


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

    # Auth middleware guarantees user_id on request.state for authed routes
    _ = request.state.user_id

    clear_knowledge_base_caches()

    # Also invalidate the RAGService instance caches held by the singleton
    rag_service = getattr(process_chat, "_rag_service", None)
    if rag_service is not None:
        rag_service.invalidate_caches()

    logger.info("Knowledge-base config reloaded by user")
    return {"status": "reloaded"}
