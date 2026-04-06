from __future__ import annotations

from fastapi import APIRouter

from gateway.api.v1 import chat_sessions, discovery, knowledge_bases, openai_compat

router = APIRouter()

router.include_router(discovery.router, tags=["discovery"])
router.include_router(openai_compat.router, prefix="/v1", tags=["openai-compat"])
router.include_router(chat_sessions.router, prefix="/v1", tags=["chat-sessions"])
router.include_router(knowledge_bases.router, prefix="/v1", tags=["knowledge-bases"])
