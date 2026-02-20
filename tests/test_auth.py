# tests/test_auth.py

"""Unit tests for the authentication module."""

import os
import time

import pytest

fastapi = pytest.importorskip("fastapi", reason="fastapi not installed")


class TestJWT:
    """Test JWT creation and verification."""

    def test_create_and_verify(self):
        from swim.shared.auth import create_jwt, verify_jwt
        secret = "test-secret-key-12345"
        token = create_jwt({"sub": "test-user"}, secret, ttl_seconds=60)
        payload = verify_jwt(token, secret)
        assert payload is not None
        assert payload["sub"] == "test-user"

    def test_wrong_secret_rejected(self):
        from swim.shared.auth import create_jwt, verify_jwt
        token = create_jwt({"sub": "user"}, "correct-secret")
        payload = verify_jwt(token, "wrong-secret")
        assert payload is None

    def test_expired_token_rejected(self):
        from swim.shared.auth import create_jwt, verify_jwt
        secret = "test-secret"
        token = create_jwt({"sub": "user"}, secret, ttl_seconds=-1)  # already expired
        payload = verify_jwt(token, secret)
        assert payload is None

    def test_malformed_token_rejected(self):
        from swim.shared.auth import verify_jwt
        assert verify_jwt("not-a-jwt", "secret") is None
        assert verify_jwt("a.b", "secret") is None
        assert verify_jwt("", "secret") is None

    def test_token_has_iat_and_exp(self):
        from swim.shared.auth import create_jwt, verify_jwt
        secret = "test"
        token = create_jwt({"sub": "u"}, secret, ttl_seconds=3600)
        payload = verify_jwt(token, secret)
        assert "iat" in payload
        assert "exp" in payload
        assert payload["exp"] > payload["iat"]


class TestAuthDependency:
    """Test the require_auth dependency logic."""

    def test_dev_mode_allows_all(self, monkeypatch):
        """When no credentials configured, auth is bypassed."""
        monkeypatch.delenv("SWIM_API_KEYS", raising=False)
        monkeypatch.delenv("SWIM_JWT_SECRET", raising=False)
        from swim.shared.auth import _auth_enabled
        assert _auth_enabled() is False

    def test_api_keys_enable_auth(self, monkeypatch):
        monkeypatch.setenv("SWIM_API_KEYS", "key1,key2")
        # Need to reimport to pick up env change
        from swim.shared.auth import _auth_enabled, _get_api_keys
        assert _auth_enabled() is True
        assert "key1" in _get_api_keys()
        assert "key2" in _get_api_keys()

    def test_jwt_secret_enables_auth(self, monkeypatch):
        monkeypatch.delenv("SWIM_API_KEYS", raising=False)
        monkeypatch.setenv("SWIM_JWT_SECRET", "my-secret")
        from swim.shared.auth import _auth_enabled
        assert _auth_enabled() is True


class TestGenerateToken:
    def test_generate_requires_secret(self, monkeypatch):
        monkeypatch.delenv("SWIM_JWT_SECRET", raising=False)
        from swim.shared.auth import generate_token
        with pytest.raises(ValueError, match="not configured"):
            generate_token()

    def test_generate_returns_valid_jwt(self, monkeypatch):
        monkeypatch.setenv("SWIM_JWT_SECRET", "test-secret")
        from swim.shared.auth import generate_token, verify_jwt
        token = generate_token(subject="client-1")
        payload = verify_jwt(token, "test-secret")
        assert payload is not None
        assert payload["sub"] == "client-1"
