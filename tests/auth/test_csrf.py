"""Tests for CSRF protection — state parameter validation in the callback."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

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
