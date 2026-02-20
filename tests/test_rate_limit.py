# tests/test_rate_limit.py

"""Unit tests for the rate limiter."""

import pytest

starlette = pytest.importorskip("starlette", reason="starlette not installed")

from swim.shared.rate_limit import RateLimiter


class TestRateLimiterLogic:
    """Test the rate limiting logic directly (without HTTP)."""

    def test_within_limit_not_blocked(self):
        limiter = RateLimiter(app=None, max_requests=5, window_seconds=60)
        for _ in range(5):
            limited, remaining = limiter._is_limited("1.2.3.4")
            assert limited is False

    def test_exceeds_limit_blocked(self):
        limiter = RateLimiter(app=None, max_requests=3, window_seconds=60)
        for _ in range(3):
            limiter._is_limited("1.2.3.4")
        limited, remaining = limiter._is_limited("1.2.3.4")
        assert limited is True
        assert remaining == 0

    def test_different_ips_independent(self):
        limiter = RateLimiter(app=None, max_requests=2, window_seconds=60)
        limiter._is_limited("1.1.1.1")
        limiter._is_limited("1.1.1.1")
        # IP 1 is at limit
        limited_1, _ = limiter._is_limited("1.1.1.1")
        assert limited_1 is True
        # IP 2 is fresh
        limited_2, _ = limiter._is_limited("2.2.2.2")
        assert limited_2 is False

    def test_remaining_count_decreases(self):
        limiter = RateLimiter(app=None, max_requests=5, window_seconds=60)
        _, rem1 = limiter._is_limited("1.2.3.4")
        assert rem1 == 4
        _, rem2 = limiter._is_limited("1.2.3.4")
        assert rem2 == 3

    def test_cleanup_removes_stale_entries(self):
        limiter = RateLimiter(app=None, max_requests=10, window_seconds=1)
        limiter._is_limited("old-ip")
        assert "old-ip" in limiter._buckets
        # Force cleanup by setting last_cleanup far in the past
        import time
        limiter._last_cleanup = time.monotonic() - 400
        limiter._buckets["old-ip"] = [time.monotonic() - 100]  # expired entry
        limiter._cleanup()
        assert "old-ip" not in limiter._buckets

    def test_exempt_paths(self):
        """Verify exempt paths are defined correctly."""
        assert "/" in RateLimiter.EXEMPT_PATHS
        assert "/health" in RateLimiter.EXEMPT_PATHS
        assert "/metrics" in RateLimiter.EXEMPT_PATHS
