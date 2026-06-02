from __future__ import annotations

from fastapi import APIRouter

from shared.config import get_settings

router = APIRouter()


@router.get("/health")
def health() -> dict:
    return {"status": "ok"}


@router.get("/config")
def config() -> dict:
    settings = get_settings()
    return {
        "service": settings.gateway.service_name,
        "vllm_base_url": settings.platform.vllm_base_url,
        "tasks": ["chat", "summarize", "code"],
        "rag_modes": ["off", "stub"],
    }
