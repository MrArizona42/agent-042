"""Google OIDC client — authorization URL, token exchange, ID token validation."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time
from typing import Any

import httpx
from authlib.integrations.httpx_client import AsyncOAuth2Client
from authlib.jose import JsonWebKey, JsonWebToken

from shared.config import Settings, secret_value

logger = logging.getLogger(__name__)

# Google OIDC endpoints (fallback values; discovery_doc overrides when available).
_GOOGLE_AUTH_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
_GOOGLE_JWKS_URI = "https://www.googleapis.com/oauth2/v3/certs"


class OIDCClient:
    """Thin wrapper around Google's OIDC endpoints."""

    def __init__(self, settings: Settings) -> None:
        self.client_id = settings.auth.google_client_id
        self.client_secret = secret_value(settings.auth.google_client_secret) or ""
        self.redirect_uri = settings.auth.google_redirect_uri
        self.discovery_url = settings.auth.google_discovery_url

        self._jwt = JsonWebToken(algorithms=["RS256"])
        self._jwks_key_set: Any | None = None

    def _oauth_client(self) -> AsyncOAuth2Client:
        return AsyncOAuth2Client(
            client_id=self.client_id,
            client_secret=self.client_secret,
            redirect_uri=self.redirect_uri,
            scope="openid email profile",
            token_endpoint_auth_method="client_secret_post",
        )

    # ------------------------------------------------------------------
    # PKCE helpers
    # ------------------------------------------------------------------

    @staticmethod
    def generate_code_verifier() -> str:
        """Generate a random code_verifier (43–128 chars, URL-safe)."""
        return base64.urlsafe_b64encode(os.urandom(32)).rstrip(b"=").decode("ascii")

    @staticmethod
    def generate_code_challenge(code_verifier: str) -> str:
        """SHA-256 code_challenge derived from *code_verifier*."""
        digest = hashlib.sha256(code_verifier.encode("ascii")).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")

    @staticmethod
    def generate_state() -> str:
        """Random opaque state string for CSRF protection."""
        return base64.urlsafe_b64encode(os.urandom(24)).rstrip(b"=").decode("ascii")

    # ------------------------------------------------------------------
    # Authorization URL
    # ------------------------------------------------------------------

    def build_authorization_url(self, state: str, code_challenge: str) -> str:
        """Return the full Google authorization URL."""
        client = self._oauth_client()
        uri, _ = client.create_authorization_url(
            _GOOGLE_AUTH_ENDPOINT,
            state=state,
            code_challenge=code_challenge,
            code_challenge_method="S256",
            access_type="offline",
            prompt="consent",
        )
        return uri

    # ------------------------------------------------------------------
    # Token exchange
    # ------------------------------------------------------------------

    async def exchange_code(self, code: str, code_verifier: str) -> dict:
        """Exchange an authorization code for tokens.

        Returns:
            dict with keys ``access_token``, ``id_token``, ``refresh_token``, ``expires_in``.
        """
        async with self._oauth_client() as client:
            return await client.fetch_token(
                _GOOGLE_TOKEN_ENDPOINT,
                grant_type="authorization_code",
                code=code,
                code_verifier=code_verifier,
                redirect_uri=self.redirect_uri,
            )

    # ------------------------------------------------------------------
    # ID token validation
    # ------------------------------------------------------------------

    def _get_jwks_key_set(self) -> Any:
        if self._jwks_key_set is None:
            with httpx.Client(timeout=15) as client:
                response = client.get(_GOOGLE_JWKS_URI)
                response.raise_for_status()
                self._jwks_key_set = JsonWebKey.import_key_set(response.json())
        return self._jwks_key_set

    def validate_id_token(self, id_token: str) -> dict:
        """Validate and decode a Google ID token.

        Verifies: RS256 signature via JWKS, ``iss``, ``aud``, ``exp``.

        Returns:
            Claims dict (sub, email, name, picture, …).

        Raises:
            authlib.jose.errors.JoseError on any validation failure.
        """
        claims = self._jwt.decode(
            id_token,
            self._get_jwks_key_set(),
            claims_options={
                "iss": {
                    "essential": True,
                    "values": ["https://accounts.google.com", "accounts.google.com"],
                },
                "aud": {
                    "essential": True,
                    "values": [self.client_id],
                },
                "exp": {"essential": True},
                "sub": {"essential": True},
            },
        )
        claims.validate()
        return dict(claims)

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    async def refresh_access_token(self, refresh_token: str) -> dict:
        """Use *refresh_token* to obtain a fresh access token (+ optionally a new id_token).

        Returns:
            dict with ``access_token``, ``expires_in``, and optionally ``id_token``.
        """
        async with self._oauth_client() as client:
            data = await client.refresh_token(
                _GOOGLE_TOKEN_ENDPOINT,
                refresh_token=refresh_token,
            )
        # Google may not return a new refresh_token; keep the original.
        if "refresh_token" not in data:
            data["refresh_token"] = refresh_token
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def compute_expiry(self, expires_in: int) -> int:
        """Return an absolute UTC timestamp *expires_in* seconds from now."""
        return int(time.time()) + expires_in
