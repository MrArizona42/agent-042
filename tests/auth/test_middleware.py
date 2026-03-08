"""Tests for AuthMiddleware — session validation and route protection."""

from __future__ import annotations

from unittest.mock import AsyncMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from gateway.auth.middleware import AuthMiddleware


def _make_app(session_data=None):
    """Create a minimal FastAPI app with AuthMiddleware for testing."""
    app = FastAPI()

    # Set app.state BEFORE adding middleware
    mock_session_mgr = AsyncMock()
    mock_session_mgr.get_session = AsyncMock(return_value=session_data)
    mock_session_mgr.update_session = AsyncMock()

    mock_oidc = AsyncMock()

    app.state.session_manager = mock_session_mgr
    app.state.oidc_client = mock_oidc

    app.add_middleware(AuthMiddleware)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/auth/login")
    def login():
        return {"url": "https://accounts.google.com"}

    @app.get("/v1/models")
    def models(request: Request):
        return {"user_id": request.state.user_id}

    @app.post("/v1/chat/completions")
    def chat(request: Request):
        return {"user_id": request.state.user_id}

    return app


class TestPublicRoutes:
    """Public routes should pass through without session validation."""

    def test_health_is_public(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "ok"}

    def test_auth_login_is_public(self):
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/auth/login")
        assert resp.status_code == 200


class TestProtectedRoutes:
    """Protected routes should return 401 without a valid session."""

    def test_v1_models_requires_auth(self):
        app = _make_app(session_data=None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/v1/models")
        assert resp.status_code == 401

    def test_v1_chat_requires_auth(self):
        app = _make_app(session_data=None)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.post("/v1/chat/completions")
        assert resp.status_code == 401

    def test_valid_session_passes(self):
        session_data = {
            "user_id": "test-user-id",
            "access_token": "at",
            "refresh_token": "rt",
            "expires_at": 9999999999,
        }
        app = _make_app(session_data=session_data)
        client = TestClient(app, raise_server_exceptions=False)
        client.cookies.set("session_id", "valid-session")
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "test-user-id"

    def test_bearer_token_works(self):
        session_data = {
            "user_id": "bearer-user",
            "access_token": "at",
            "refresh_token": "rt",
            "expires_at": 9999999999,
        }
        app = _make_app(session_data=session_data)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get(
            "/v1/models",
            headers={"Authorization": "Bearer valid-session"},
        )
        assert resp.status_code == 200
        assert resp.json()["user_id"] == "bearer-user"

    def test_expired_session_returns_401(self):
        app = _make_app(session_data=None)  # Redis miss
        client = TestClient(app, raise_server_exceptions=False)
        client.cookies.set("session_id", "expired-session")
        resp = client.get("/v1/models")
        assert resp.status_code == 401
