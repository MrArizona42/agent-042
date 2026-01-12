from __future__ import annotations

from fastapi import APIRouter

from gateway.api.v1 import discovery, openai_compat

router = APIRouter()

router.include_router(discovery.router, tags=["discovery"])
router.include_router(openai_compat.router, prefix="/v1", tags=["openai-compat"])

