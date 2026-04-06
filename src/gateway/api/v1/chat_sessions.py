"""Chat session API — create, list, get messages, delete."""

from __future__ import annotations

import uuid

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from sqlalchemy import select

from shared.db.engine import get_session_factory
from shared.db.models import ChatMessage, ChatSession

router = APIRouter()


@router.post("/chat/sessions")
async def create_session(request: Request, title: str | None = None) -> JSONResponse:
    """Create a new chat session for the authenticated user."""
    user_id = uuid.UUID(request.state.user_id)

    async with get_session_factory()() as db:
        session = ChatSession(user_id=user_id, title=title)
        db.add(session)
        await db.commit()
        return JSONResponse(
            {
                "id": str(session.id),
                "title": session.title,
                "created_at": session.created_at.isoformat(),
            },
            status_code=201,
        )


@router.get("/chat/sessions")
async def list_sessions(request: Request) -> JSONResponse:
    """List the authenticated user's chat sessions (newest first)."""
    user_id = uuid.UUID(request.state.user_id)

    async with get_session_factory()() as db:
        result = await db.execute(
            select(ChatSession)
            .where(ChatSession.user_id == user_id)
            .order_by(ChatSession.updated_at.desc())
        )
        sessions = result.scalars().all()
        return JSONResponse(
            [
                {
                    "id": str(s.id),
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "updated_at": s.updated_at.isoformat(),
                }
                for s in sessions
            ]
        )


@router.get("/chat/sessions/{session_id}/messages")
async def get_session_messages(request: Request, session_id: uuid.UUID) -> JSONResponse:
    """Return all messages for a chat session (ownership enforced)."""
    user_id = uuid.UUID(request.state.user_id)

    async with get_session_factory()() as db:
        # Verify ownership
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        msg_result = await db.execute(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at)
        )
        messages = msg_result.scalars().all()
        return JSONResponse(
            [
                {
                    "id": str(m.id),
                    "role": m.role,
                    "content": m.content,
                    "created_at": m.created_at.isoformat(),
                }
                for m in messages
            ]
        )


@router.delete("/chat/sessions/{session_id}")
async def delete_session(request: Request, session_id: uuid.UUID) -> JSONResponse:
    """Delete a chat session (cascade deletes messages)."""
    user_id = uuid.UUID(request.state.user_id)

    async with get_session_factory()() as db:
        result = await db.execute(
            select(ChatSession).where(ChatSession.id == session_id, ChatSession.user_id == user_id)
        )
        session = result.scalar_one_or_none()
        if session is None:
            return JSONResponse({"detail": "Session not found"}, status_code=404)

        await db.delete(session)
        await db.commit()
        return JSONResponse({"detail": "Session deleted"})
