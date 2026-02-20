# swim/shared/rate_limit.py

"""Per-IP sliding-window rate limiter as FastAPI middleware."""

import logging
import time
from collections import defaultdict
from typing import Dict, List, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

# Default: 60 requests per minute
DEFAULT_RATE = 60
DEFAULT_WINDOW = 60  # seconds


class RateLimiter(BaseHTTPMiddleware):
    """Sliding-window rate limiter per client IP.

    Exempt paths (health, root, metrics) are not rate-limited.
    """

    EXEMPT_PATHS = {"/", "/health", "/metrics", "/.well-known/agent.json"}

    def __init__(self, app, max_requests: int = DEFAULT_RATE, window_seconds: int = DEFAULT_WINDOW):
        super().__init__(app)
        self.max_requests = max_requests
        self.window = window_seconds
        # ip -> list of request timestamps
        self._buckets: Dict[str, List[float]] = defaultdict(list)
        self._last_cleanup = time.monotonic()

    def _cleanup(self):
        """Evict expired entries every 5 minutes to prevent unbounded growth."""
        now = time.monotonic()
        if now - self._last_cleanup < 300:
            return
        cutoff = now - self.window
        stale_ips = []
        for ip, timestamps in self._buckets.items():
            self._buckets[ip] = [t for t in timestamps if t > cutoff]
            if not self._buckets[ip]:
                stale_ips.append(ip)
        for ip in stale_ips:
            del self._buckets[ip]
        self._last_cleanup = now

    def _is_limited(self, ip: str) -> Tuple[bool, int]:
        """Check if IP is rate-limited. Returns (limited, remaining)."""
        now = time.monotonic()
        cutoff = now - self.window
        # Prune old entries for this IP
        self._buckets[ip] = [t for t in self._buckets[ip] if t > cutoff]
        count = len(self._buckets[ip])

        if count >= self.max_requests:
            return True, 0

        self._buckets[ip].append(now)
        return False, self.max_requests - count - 1

    async def dispatch(self, request: Request, call_next):
        # Skip rate limiting for exempt paths
        if request.url.path in self.EXEMPT_PATHS:
            return await call_next(request)

        ip = request.client.host if request.client else "unknown"
        limited, remaining = self._is_limited(ip)

        if limited:
            logger.warning("Rate limit exceeded for %s on %s", ip, request.url.path)
            return JSONResponse(
                {"detail": "Rate limit exceeded. Try again later."},
                status_code=429,
                headers={
                    "X-RateLimit-Limit": str(self.max_requests),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(self.window),
                },
            )

        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self.max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        self._cleanup()
        return response
