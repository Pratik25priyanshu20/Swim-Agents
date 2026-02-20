# swim/shared/auth.py

"""API authentication: API-key and JWT bearer token validation."""

import hashlib
import hmac
import json
import logging
import os
import time
from typing import Optional

from fastapi import Depends, HTTPException, Security
from fastapi.security import APIKeyHeader, HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_api_keys() -> list[str]:
    """Return list of valid API keys from env (comma-separated)."""
    raw = os.getenv("SWIM_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def _get_jwt_secret() -> str:
    return os.getenv("SWIM_JWT_SECRET", "")


def _auth_enabled() -> bool:
    """Auth is enabled when at least one credential source is configured."""
    return bool(_get_api_keys() or _get_jwt_secret())


# ---------------------------------------------------------------------------
# Minimal JWT helpers (HS256, no external library)
# ---------------------------------------------------------------------------

def _b64url_encode(data: bytes) -> str:
    import base64
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_decode(s: str) -> bytes:
    import base64
    s += "=" * (4 - len(s) % 4)
    return base64.urlsafe_b64decode(s)


def create_jwt(payload: dict, secret: str, ttl_seconds: int = 3600) -> str:
    """Create a simple HS256 JWT."""
    header = {"alg": "HS256", "typ": "JWT"}
    payload = {**payload, "iat": int(time.time()), "exp": int(time.time()) + ttl_seconds}
    segments = [
        _b64url_encode(json.dumps(header).encode()),
        _b64url_encode(json.dumps(payload).encode()),
    ]
    signing_input = ".".join(segments).encode()
    signature = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
    segments.append(_b64url_encode(signature))
    return ".".join(segments)


def verify_jwt(token: str, secret: str) -> Optional[dict]:
    """Verify an HS256 JWT. Returns payload dict or None."""
    try:
        parts = token.split(".")
        if len(parts) != 3:
            return None
        signing_input = f"{parts[0]}.{parts[1]}".encode()
        expected_sig = hmac.new(secret.encode(), signing_input, hashlib.sha256).digest()
        actual_sig = _b64url_decode(parts[2])
        if not hmac.compare_digest(expected_sig, actual_sig):
            return None
        payload = json.loads(_b64url_decode(parts[1]))
        if payload.get("exp", 0) < time.time():
            return None
        return payload
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FastAPI dependency
# ---------------------------------------------------------------------------

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
_bearer_scheme = HTTPBearer(auto_error=False)


async def require_auth(
    api_key: Optional[str] = Security(_api_key_header),
    bearer: Optional[HTTPAuthorizationCredentials] = Security(_bearer_scheme),
) -> dict:
    """FastAPI dependency — validates API key or JWT.

    Returns a dict with at least {"authenticated": True, "method": "..."}.
    When auth is not configured (dev mode), returns {"authenticated": True, "method": "none"}.
    """
    if not _auth_enabled():
        # Dev mode — no credentials configured, allow all
        return {"authenticated": True, "method": "none"}

    # Try API key first
    valid_keys = _get_api_keys()
    if api_key and valid_keys:
        if api_key in valid_keys:
            return {"authenticated": True, "method": "api_key"}

    # Try JWT
    jwt_secret = _get_jwt_secret()
    if bearer and jwt_secret:
        payload = verify_jwt(bearer.credentials, jwt_secret)
        if payload:
            return {"authenticated": True, "method": "jwt", "claims": payload}

    logger.warning("Authentication failed: invalid credentials")
    raise HTTPException(status_code=401, detail="Invalid or missing credentials")


# ---------------------------------------------------------------------------
# Token generation endpoint helper
# ---------------------------------------------------------------------------

def generate_token(subject: str = "swim-client", ttl: int = 3600) -> str:
    """Generate a JWT for the given subject. Requires SWIM_JWT_SECRET."""
    secret = _get_jwt_secret()
    if not secret:
        raise ValueError("SWIM_JWT_SECRET not configured")
    return create_jwt({"sub": subject}, secret, ttl)
