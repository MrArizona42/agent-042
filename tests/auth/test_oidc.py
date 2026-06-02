"""Tests for OIDC client — PKCE generation, authorization URL building, ID token validation."""

from __future__ import annotations

import hashlib
import time
from base64 import urlsafe_b64encode
from unittest.mock import MagicMock

import jwt
import pytest
from cryptography.hazmat.primitives.asymmetric import rsa

from gateway.auth.oidc import OIDCClient
from shared.config import Settings

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_settings(**overrides) -> Settings:
    """Return gateway settings with the auth section configured for OIDCClient."""
    auth_values = {
        "google_client_id": "test-client-id.apps.googleusercontent.com",
        "google_client_secret": "test-client-secret",
        "google_redirect_uri": "https://example.com/auth/callback",
        "google_discovery_url": ("https://accounts.google.com/.well-known/openid-configuration"),
    }
    auth_values.update(overrides.pop("auth", {}))
    return Settings(auth=auth_values, **overrides)


def _generate_rsa_keypair():
    """Generate an RSA key pair for testing JWT signing/verification."""
    private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return private_key


# ---------------------------------------------------------------------------
# PKCE Tests
# ---------------------------------------------------------------------------


class TestPKCE:
    def test_code_verifier_length(self):
        verifier = OIDCClient.generate_code_verifier()
        assert 43 <= len(verifier) <= 128

    def test_code_verifier_url_safe(self):
        verifier = OIDCClient.generate_code_verifier()
        # URL-safe base64 chars only
        allowed = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_"
        assert all(c in allowed for c in verifier)

    def test_code_challenge_is_sha256(self):
        verifier = OIDCClient.generate_code_verifier()
        challenge = OIDCClient.generate_code_challenge(verifier)

        # Re-derive manually
        digest = hashlib.sha256(verifier.encode("ascii")).digest()
        expected = urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
        assert challenge == expected

    def test_state_is_random(self):
        s1 = OIDCClient.generate_state()
        s2 = OIDCClient.generate_state()
        assert s1 != s2
        assert len(s1) > 10


# ---------------------------------------------------------------------------
# Authorization URL Tests
# ---------------------------------------------------------------------------


class TestAuthorizationURL:
    def test_url_contains_required_params(self):
        client = OIDCClient(_make_settings())
        state = "test-state"
        challenge = "test-challenge"
        url = client.build_authorization_url(state, challenge)

        assert "client_id=test-client-id" in url
        assert "redirect_uri=" in url
        assert "state=test-state" in url
        assert "code_challenge=test-challenge" in url
        assert "code_challenge_method=S256" in url
        assert "scope=openid" in url
        assert "access_type=offline" in url
        assert "prompt=consent" in url


# ---------------------------------------------------------------------------
# ID Token Validation Tests
# ---------------------------------------------------------------------------


class TestIDTokenValidation:
    def test_valid_token_is_accepted(self):
        private_key = _generate_rsa_keypair()

        claims = {
            "iss": "https://accounts.google.com",
            "aud": "test-client-id.apps.googleusercontent.com",
            "sub": "123456789",
            "email": "test@example.com",
            "name": "Test User",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256", headers={"kid": "test-kid"})

        client = OIDCClient(_make_settings())

        # Mock the JWKS client to return our test key
        mock_jwk = MagicMock()
        mock_jwk.key = private_key.public_key()
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_jwk
        client._jwks_client = mock_jwks_client

        result = client.validate_id_token(token)
        assert result["sub"] == "123456789"
        assert result["email"] == "test@example.com"

    def test_expired_token_is_rejected(self):
        private_key = _generate_rsa_keypair()

        claims = {
            "iss": "https://accounts.google.com",
            "aud": "test-client-id.apps.googleusercontent.com",
            "sub": "123456789",
            "exp": int(time.time()) - 3600,  # Expired!
            "iat": int(time.time()) - 7200,
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        client = OIDCClient(_make_settings())
        mock_jwk = MagicMock()
        mock_jwk.key = private_key.public_key()
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_jwk
        client._jwks_client = mock_jwks_client

        with pytest.raises(jwt.ExpiredSignatureError):
            client.validate_id_token(token)

    def test_wrong_audience_is_rejected(self):
        private_key = _generate_rsa_keypair()

        claims = {
            "iss": "https://accounts.google.com",
            "aud": "wrong-client-id",  # Wrong!
            "sub": "123456789",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        client = OIDCClient(_make_settings())
        mock_jwk = MagicMock()
        mock_jwk.key = private_key.public_key()
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_jwk
        client._jwks_client = mock_jwks_client

        with pytest.raises(jwt.InvalidAudienceError):
            client.validate_id_token(token)

    def test_wrong_issuer_is_rejected(self):
        private_key = _generate_rsa_keypair()

        claims = {
            "iss": "https://evil.example.com",  # Wrong!
            "aud": "test-client-id.apps.googleusercontent.com",
            "sub": "123456789",
            "exp": int(time.time()) + 3600,
            "iat": int(time.time()),
        }

        token = jwt.encode(claims, private_key, algorithm="RS256")

        client = OIDCClient(_make_settings())
        mock_jwk = MagicMock()
        mock_jwk.key = private_key.public_key()
        mock_jwks_client = MagicMock()
        mock_jwks_client.get_signing_key_from_jwt.return_value = mock_jwk
        client._jwks_client = mock_jwks_client

        with pytest.raises(jwt.InvalidIssuerError):
            client.validate_id_token(token)


# ---------------------------------------------------------------------------
# Expiry computation
# ---------------------------------------------------------------------------


class TestComputeExpiry:
    def test_compute_expiry(self):
        client = OIDCClient(_make_settings())
        now = int(time.time())
        result = client.compute_expiry(3600)
        assert result >= now + 3599  # Allow 1s tolerance
        assert result <= now + 3601
