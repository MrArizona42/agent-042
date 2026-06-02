"""Google OIDC client — authorization URL, token exchange, ID token validation."""

from __future__ import annotations

import base64
import hashlib
import logging
import os
import time

import httpx
import jwt

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

        # Cache for JWKS public keys
        self._jwks_client: jwt.PyJWKClient | None = None

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
        params = {
            "client_id": self.client_id,
            "redirect_uri": self.redirect_uri,
            "response_type": "code",
            "scope": "openid email profile",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "access_type": "offline",
            "prompt": "consent",
        }
        # Use httpx URL builder
        url = httpx.URL(_GOOGLE_AUTH_ENDPOINT, params=params)
        return str(url)

    # ------------------------------------------------------------------
    # Token exchange
    # ------------------------------------------------------------------

    async def exchange_code(self, code: str, code_verifier: str) -> dict:
        """Exchange an authorization code for tokens.

        Returns:
            dict with keys ``access_token``, ``id_token``, ``refresh_token``, ``expires_in``.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                _GOOGLE_TOKEN_ENDPOINT,
                data={
                    "code": code,
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "redirect_uri": self.redirect_uri,
                    "grant_type": "authorization_code",
                    "code_verifier": code_verifier,
                },
            )
            resp.raise_for_status()
            return resp.json()

    # ------------------------------------------------------------------
    # ID token validation
    # ------------------------------------------------------------------

    def _get_jwks_client(self) -> jwt.PyJWKClient:
        if self._jwks_client is None:
            self._jwks_client = jwt.PyJWKClient(_GOOGLE_JWKS_URI)
        return self._jwks_client

    def validate_id_token(self, id_token: str) -> dict:
        """Validate and decode a Google ID token.

        Verifies: RS256 signature via JWKS, ``iss``, ``aud``, ``exp``.

        Returns:
            Claims dict (sub, email, name, picture, …).

        Raises:
            jwt.InvalidTokenError on any validation failure.
        """
        jwks_client = self._get_jwks_client()
        signing_key = jwks_client.get_signing_key_from_jwt(id_token)

        # Decode with signature and exp/aud verification.
        # Issuer is checked manually because Google may use either
        # "https://accounts.google.com" or "accounts.google.com",
        # and older PyJWT versions don't accept a list for `issuer`.
        claims = jwt.decode(
            id_token,
            signing_key.key,
            algorithms=["RS256"],
            audience=self.client_id,
            options={
                "verify_exp": True,
                "verify_iss": False,  # Manual check below
                "verify_aud": True,
            },
        )

        # Manual issuer validation
        allowed_issuers = {"https://accounts.google.com", "accounts.google.com"}
        if claims.get("iss") not in allowed_issuers:
            raise jwt.InvalidIssuerError("Invalid issuer")

        return claims

    # ------------------------------------------------------------------
    # Token refresh
    # ------------------------------------------------------------------

    async def refresh_access_token(self, refresh_token: str) -> dict:
        """Use *refresh_token* to obtain a fresh access token (+ optionally a new id_token).

        Returns:
            dict with ``access_token``, ``expires_in``, and optionally ``id_token``.
        """
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                _GOOGLE_TOKEN_ENDPOINT,
                data={
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "refresh_token": refresh_token,
                    "grant_type": "refresh_token",
                },
            )
            resp.raise_for_status()
            data = resp.json()
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
