"""Tests for CSRF protection — state parameter validation in the callback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app_config.runtime import get_settings
from gateway.auth.router import _session_cookie_secure
from gateway.auth.router import router as auth_router


def _make_app(session_mgr=None, oidc_client=None):
    """Create a minimal app with the auth router."""
    app = FastAPI()
    app.include_router(auth_router)

    mock_session_mgr = session_mgr or AsyncMock()
    mock_oidc = oidc_client or MagicMock()

    app.state.session_manager = mock_session_mgr
    app.state.oidc_client = mock_oidc

    return app


class TestCallbackCSRF:
    """Callback should reject invalid or missing state parameters."""

    def test_missing_state_param_returns_422(self):
        """FastAPI returns 422 for missing required query params."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        # No state parameter at all
        resp = client.get("/auth/callback?code=some-code")
        assert resp.status_code == 422

    def test_invalid_state_returns_400(self):
        """Unrecognized state should be rejected."""
        mock_session_mgr = AsyncMock()
        mock_session_mgr.consume_oauth_state = AsyncMock(return_value=None)

        app = _make_app(session_mgr=mock_session_mgr)
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/auth/callback?code=some-code&state=invalid-state")
        assert resp.status_code == 400
        assert "Invalid or expired" in resp.json()["detail"]

    def test_missing_code_param_returns_422(self):
        """FastAPI returns 422 for missing required query params."""
        app = _make_app()
        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/auth/callback?state=some-state")
        assert resp.status_code == 422


class TestSessionCookiePolicy:
    """Session cookie security should match the configured callback scheme."""

    def test_http_redirect_uri_allows_local_cookie(self, monkeypatch):
        monkeypatch.setenv(
            "AUTH__GOOGLE_REDIRECT_URI",
            "http://localhost:9000/auth/callback",
        )
        get_settings.cache_clear()

        app = FastAPI()

        @app.get("/cookie-secure")
        def cookie_secure(request: Request):
            return {"secure": _session_cookie_secure(request)}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/cookie-secure")

        assert resp.status_code == 200
        assert resp.json() == {"secure": False}

        get_settings.cache_clear()

    def test_https_redirect_uri_sets_secure_cookie(self, monkeypatch):
        monkeypatch.setenv(
            "AUTH__GOOGLE_REDIRECT_URI",
            "https://agent.example/auth/callback",
        )
        get_settings.cache_clear()

        app = FastAPI()

        @app.get("/cookie-secure")
        def cookie_secure(request: Request):
            return {"secure": _session_cookie_secure(request)}

        client = TestClient(app, raise_server_exceptions=False)
        resp = client.get("/cookie-secure")

        assert resp.status_code == 200
        assert resp.json() == {"secure": True}

        get_settings.cache_clear()
