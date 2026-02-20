# swim/shared/sanitize.py

"""Input sanitization to prevent prompt injection, XSS, and other attacks."""

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

# Max lengths for different input types
MAX_QUERY_LENGTH = 2000
MAX_LAKE_NAME_LENGTH = 100
MAX_IMAGE_NAME_LENGTH = 255
MAX_WEBHOOK_URL_LENGTH = 500

# Characters allowed in lake names (alphanumeric, German umlauts, spaces, hyphens)
LAKE_NAME_PATTERN = re.compile(r"^[\w\sÄäÖöÜüß\-\.]+$", re.UNICODE)

# Known prompt injection patterns
PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?prior", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*/?script", re.IGNORECASE),
    re.compile(r"javascript\s*:", re.IGNORECASE),
    re.compile(r"on(error|load|click)\s*=", re.IGNORECASE),
]

# SQL injection patterns
SQL_INJECTION_PATTERNS = [
    re.compile(r";\s*(DROP|DELETE|UPDATE|INSERT|ALTER)\s", re.IGNORECASE),
    re.compile(r"'\s*(OR|AND)\s+\d+\s*=\s*\d+", re.IGNORECASE),
    re.compile(r"UNION\s+(ALL\s+)?SELECT", re.IGNORECASE),
    re.compile(r"--\s*$", re.MULTILINE),
]


def _check_injection(text: str) -> Optional[str]:
    """Check for prompt injection / SQL injection. Returns pattern name or None."""
    for pattern in PROMPT_INJECTION_PATTERNS:
        if pattern.search(text):
            return "prompt_injection"
    for pattern in SQL_INJECTION_PATTERNS:
        if pattern.search(text):
            return "sql_injection"
    return None


def sanitize_query(text: str) -> str:
    """Sanitize a user query string.

    Raises ValueError if injection is detected.
    """
    if not text:
        return ""
    text = text.strip()
    if len(text) > MAX_QUERY_LENGTH:
        text = text[:MAX_QUERY_LENGTH]
        logger.warning("Query truncated to %d chars", MAX_QUERY_LENGTH)

    threat = _check_injection(text)
    if threat:
        logger.warning("Blocked %s attempt in query", threat)
        raise ValueError(f"Input rejected: suspicious pattern detected ({threat})")

    # Strip HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    return text


def sanitize_lake_name(name: str) -> str:
    """Validate and sanitize a lake name."""
    if not name:
        raise ValueError("Lake name cannot be empty")
    name = name.strip()
    if len(name) > MAX_LAKE_NAME_LENGTH:
        raise ValueError(f"Lake name too long (max {MAX_LAKE_NAME_LENGTH})")
    if not LAKE_NAME_PATTERN.match(name):
        raise ValueError("Lake name contains invalid characters")
    return name


def sanitize_image_name(name: str) -> str:
    """Validate an image filename — prevent path traversal."""
    if not name:
        raise ValueError("Image name cannot be empty")
    name = name.strip()
    if len(name) > MAX_IMAGE_NAME_LENGTH:
        raise ValueError(f"Image name too long (max {MAX_IMAGE_NAME_LENGTH})")
    # Block path traversal
    if ".." in name or "/" in name or "\\" in name:
        raise ValueError("Invalid image name: path traversal not allowed")
    # Only allow common image extensions
    if not re.match(r"^[\w\-. ]+\.(jpg|jpeg|png|bmp|tiff?)$", name, re.IGNORECASE):
        raise ValueError("Invalid image file extension")
    return name


def sanitize_webhook_url(url: str) -> str:
    """Validate a webhook URL."""
    if not url:
        return ""
    url = url.strip()
    if len(url) > MAX_WEBHOOK_URL_LENGTH:
        raise ValueError(f"Webhook URL too long (max {MAX_WEBHOOK_URL_LENGTH})")
    if not url.startswith(("http://", "https://")):
        raise ValueError("Webhook URL must start with http:// or https://")
    # Block internal/private IPs
    if re.search(r"https?://(localhost|127\.0\.0\.\d+|10\.\d+|172\.(1[6-9]|2\d|3[01])\.|192\.168\.)", url):
        raise ValueError("Webhook URL cannot target internal addresses")
    return url
