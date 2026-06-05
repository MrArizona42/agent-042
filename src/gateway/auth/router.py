"""Auth router — login, callback, logout, me."""

from __future__ import annotations

import logging
import uuid
from urllib.parse import urlparse

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, RedirectResponse
from sqlalchemy import select

from shared.config import get_settings
from shared.db.engine import get_session_factory
from shared.db.models import User

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])


def _session_cookie_secure(request: Request) -> bool:
    """Return whether the session cookie should require HTTPS."""
    redirect_uri = get_settings().auth.google_redirect_uri
    if redirect_uri:
        scheme = urlparse(redirect_uri).scheme.lower()
        if scheme in {"http", "https"}:
            return scheme == "https"

    forwarded_proto = request.headers.get("x-forwarded-proto")
    if forwarded_proto:
        return forwarded_proto.split(",", 1)[0].strip().lower() == "https"
    return request.url.scheme == "https"


@router.get("/login")
async def login(request: Request) -> RedirectResponse:
    """Start the OAuth2 / PKCE flow: redirect user to Google."""
    oidc_client = request.app.state.oidc_client
    session_mgr = request.app.state.session_manager

    state = oidc_client.generate_state()
    code_verifier = oidc_client.generate_code_verifier()
    code_challenge = oidc_client.generate_code_challenge(code_verifier)

    await session_mgr.store_oauth_state(state, code_verifier)

    url = oidc_client.build_authorization_url(state, code_challenge)
    return RedirectResponse(url, status_code=302)


@router.get("/callback")
async def callback(request: Request, code: str, state: str) -> RedirectResponse:
    """Handle the Google OAuth2 callback."""
    oidc_client = request.app.state.oidc_client
    session_mgr = request.app.state.session_manager

    # 1. Validate state → get code_verifier
    code_verifier = await session_mgr.consume_oauth_state(state)
    if code_verifier is None:
        return JSONResponse({"detail": "Invalid or expired OAuth state"}, status_code=400)

    # 2. Exchange authorization code for tokens
    try:
        tokens = await oidc_client.exchange_code(code, code_verifier)
    except Exception:
        logger.exception("Token exchange failed")
        return JSONResponse({"detail": "Token exchange failed"}, status_code=502)

    # 3. Validate ID token
    try:
        claims = oidc_client.validate_id_token(tokens["id_token"])
    except Exception:
        logger.exception("ID token validation failed")
        return JSONResponse({"detail": "ID token validation failed"}, status_code=401)

    # 4. Upsert user in PostgreSQL
    sub = claims["sub"]
    email = claims.get("email")
    name = claims.get("name")
    picture = claims.get("picture")

    async with get_session_factory()() as db:
        result = await db.execute(select(User).where(User.provider == "google", User.sub == sub))
        user = result.scalar_one_or_none()
        if user is None:
            user = User(
                id=uuid.uuid4(),
                provider="google",
                sub=sub,
                email=email,
                name=name,
                picture=picture,
            )
            db.add(user)
        else:
            user.email = email
            user.name = name
            user.picture = picture
        await db.commit()
        user_id = str(user.id)

    # 5. Create session in Redis
    expires_at = oidc_client.compute_expiry(tokens.get("expires_in", 3600))
    session_id = await session_mgr.create_session(
        user_id=user_id,
        access_token=tokens["access_token"],
        refresh_token=tokens.get("refresh_token", ""),
        expires_at=expires_at,
    )

    # 6. Redirect to root with session cookie
    response = RedirectResponse("/", status_code=302)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=_session_cookie_secure(request),
        samesite="lax",
        max_age=request.app.state.session_manager._ttl,
        path="/",
    )
    return response


@router.get("/logout")
async def logout(request: Request) -> RedirectResponse:
    """Delete the session from Redis, clear the cookie, redirect to root."""
    session_id = request.cookies.get("session_id")
    if session_id:
        session_mgr = request.app.state.session_manager
        await session_mgr.delete_session(session_id)

    response = RedirectResponse("/", status_code=302)
    response.delete_cookie("session_id", path="/")
    return response


@router.get("/me")
async def me(request: Request) -> JSONResponse:
    """Return the currently authenticated user's profile."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        auth_header = request.headers.get("authorization", "")
        if auth_header.lower().startswith("bearer "):
            session_id = auth_header[7:].strip()
    if not session_id:
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)

    session_mgr = request.app.state.session_manager
    session_data = await session_mgr.get_session(session_id)
    if session_data is None:
        return JSONResponse({"detail": "Session expired or invalid"}, status_code=401)

    user_id = session_data["user_id"]

    async with get_session_factory()() as db:
        result = await db.execute(select(User).where(User.id == uuid.UUID(user_id)))
        user = result.scalar_one_or_none()
        if user is None:
            return JSONResponse({"detail": "User not found"}, status_code=404)

    return JSONResponse(
        {
            "user_id": str(user.id),
            "email": user.email,
            "name": user.name,
            "picture": user.picture,
        }
    )
