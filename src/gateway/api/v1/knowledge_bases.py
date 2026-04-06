"""Admin endpoints for knowledge-base discovery."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from gateway.services.rag_service import RAGService

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
