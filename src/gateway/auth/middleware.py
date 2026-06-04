"""Auth middleware — validates session cookie or internal API key on every protected request."""

from __future__ import annotations

import hmac
import logging
import time

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import JSONResponse

from shared.config import get_settings, secret_value

logger = logging.getLogger(__name__)

# Paths that do NOT require authentication.
_PUBLIC_PREFIXES = (
    "/health",
    "/auth/",
    "/docs",
    "/openapi.json",
    "/redoc",
)

_SESSION_COOKIE = "session_id"

# Refresh the access token if it expires within this many seconds.
_REFRESH_WINDOW_SECONDS = 120

# Internal service identity used when authenticating via API key.
_SERVICE_USER_ID = "__service__"


class AuthMiddleware(BaseHTTPMiddleware):
    """Reject unauthenticated requests to protected routes.

    On every non-public request the middleware tries, in order:
    1. ``X-API-Key`` header — compared against ``GATEWAY_INTERNAL_API_KEY``
       for service-to-service calls (e.g. Airflow eval runner).
    2. ``session_id`` cookie (or ``Authorization: Bearer`` header) — looked
       up in Redis for user sessions via OAuth2/OIDC.
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        path = request.url.path

        # Allow public routes through without auth
        if any(path.startswith(prefix) for prefix in _PUBLIC_PREFIXES):
            return await call_next(request)

        # --- Internal API key authentication ---
        api_key = request.headers.get("x-api-key")
        if api_key:
            internal_key = secret_value(get_settings().auth.internal_api_key)
            if internal_key and hmac.compare_digest(api_key, internal_key):
                request.state.user_id = _SERVICE_USER_ID
                request.state.session_id = None
                return await call_next(request)
            return JSONResponse({"detail": "Invalid API key"}, status_code=401)

        # --- Extract session id ---
        session_id: str | None = request.cookies.get(_SESSION_COOKIE)
        if session_id is None:
            auth_header = request.headers.get("authorization", "")
            if auth_header.lower().startswith("bearer "):
                session_id = auth_header[7:].strip()

        if not session_id:
            return JSONResponse({"detail": "Not authenticated"}, status_code=401)

        # --- Look up session in Redis ---
        session_mgr = request.app.state.session_manager
        session_data = await session_mgr.get_session(session_id)
        if session_data is None:
            return JSONResponse({"detail": "Session expired or invalid"}, status_code=401)

        # --- Refresh access token if near expiry ---
        expires_at = session_data.get("expires_at", 0)
        if expires_at - time.time() < _REFRESH_WINDOW_SECONDS:
            try:
                oidc_client = request.app.state.oidc_client
                refreshed = await oidc_client.refresh_access_token(session_data["refresh_token"])
                session_data["access_token"] = refreshed["access_token"]
                session_data["refresh_token"] = refreshed.get(
                    "refresh_token", session_data["refresh_token"]
                )
                session_data["expires_at"] = oidc_client.compute_expiry(
                    refreshed.get("expires_in", 3600)
                )
                await session_mgr.update_session(session_id, session_data)
                logger.info(f"Access token refreshed for session {session_id}")
            except Exception:
                logger.warning(
                    f"Failed to refresh access token for session {session_id}",
                    exc_info=True,
                )
                # Don't block the request — the current token may still be valid.

        # --- Inject into request state ---
        request.state.user_id = session_data["user_id"]
        request.state.session_id = session_id

        return await call_next(request)
