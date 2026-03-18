"""Redis-backed session management for OAuth2 sessions."""

from __future__ import annotations

import json
import logging
import uuid

import redis.asyncio as aioredis

from shared.config import get_settings

logger = logging.getLogger(__name__)

_SESSION_PREFIX = "session:"
_OAUTH_STATE_PREFIX = "oauth_state:"
_OAUTH_STATE_TTL = 600  # 10 minutes


class SessionManager:
    """Manages auth sessions and OAuth state in Redis."""

    def __init__(self, redis: aioredis.Redis) -> None:
        self._redis = redis
        self._ttl = get_settings().session_ttl_seconds

    # ------------------------------------------------------------------
    # Auth sessions
    # ------------------------------------------------------------------

    async def create_session(
        self,
        user_id: str,
        access_token: str,
        refresh_token: str,
        expires_at: int,
    ) -> str:
        """Create a new session and return its UUID string."""
        session_id = str(uuid.uuid4())
        data = json.dumps(
            {
                "user_id": user_id,
                "access_token": access_token,
                "refresh_token": refresh_token,
                "expires_at": expires_at,
            }
        )
        await self._redis.set(f"{_SESSION_PREFIX}{session_id}", data, ex=self._ttl)
        logger.info(f"Session {session_id} created (ttl={self._ttl}s)")
        return session_id

    async def get_session(self, session_id: str) -> dict | None:
        """Retrieve session data or ``None`` if missing / expired."""
        raw = await self._redis.get(f"{_SESSION_PREFIX}{session_id}")
        if raw is None:
            return None
        return json.loads(raw)

    async def update_session(self, session_id: str, data: dict) -> None:
        """Overwrite session data, preserving existing TTL."""
        ttl = await self._redis.ttl(f"{_SESSION_PREFIX}{session_id}")
        if ttl < 0:
            ttl = self._ttl
        await self._redis.set(
            f"{_SESSION_PREFIX}{session_id}", json.dumps(data), ex=ttl
        )

    async def delete_session(self, session_id: str) -> None:
        """Delete a session from Redis."""
        await self._redis.delete(f"{_SESSION_PREFIX}{session_id}")
        logger.info(f"Session {session_id} deleted")

    # ------------------------------------------------------------------
    # OAuth state (CSRF protection + PKCE code_verifier)
    # ------------------------------------------------------------------

    async def store_oauth_state(self, state: str, code_verifier: str) -> None:
        """Store *code_verifier* keyed by *state* with a 10-minute TTL."""
        await self._redis.set(
            f"{_OAUTH_STATE_PREFIX}{state}",
            code_verifier,
            ex=_OAUTH_STATE_TTL,
        )

    async def consume_oauth_state(self, state: str) -> str | None:
        """Atomically get and delete the *code_verifier* for *state*.

        Returns:
            code_verifier string or ``None`` if state is invalid / expired.
        """
        key = f"{_OAUTH_STATE_PREFIX}{state}"
        pipe = self._redis.pipeline()
        pipe.get(key)
        pipe.delete(key)
        results = await pipe.execute()
        code_verifier = results[0]
        if code_verifier is None:
            return None
        # redis may return bytes or str depending on decode_responses
        if isinstance(code_verifier, bytes):
            code_verifier = code_verifier.decode("utf-8")
        return code_verifier
