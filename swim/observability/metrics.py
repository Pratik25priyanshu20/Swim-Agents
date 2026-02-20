# swim/observability/metrics.py

"""Prometheus metrics for the SWIM platform.

Provides counters, histograms, and gauges for pipeline and agent monitoring.
Uses prometheus_client if available; falls back to no-op stubs otherwise.
"""

import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        generate_latest,
        CONTENT_TYPE_LATEST,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed — metrics disabled")

# ---------------------------------------------------------------------------
# Registry & metrics (created only once)
# ---------------------------------------------------------------------------

if PROMETHEUS_AVAILABLE:
    REGISTRY = CollectorRegistry()

    # Counters
    REQUEST_COUNT = Counter(
        "swim_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
        registry=REGISTRY,
    )
    PIPELINE_RUNS = Counter(
        "swim_pipeline_runs_total",
        "Total pipeline executions",
        ["status"],
        registry=REGISTRY,
    )
    AGENT_CALLS = Counter(
        "swim_agent_calls_total",
        "Total calls to individual agents",
        ["agent", "status"],
        registry=REGISTRY,
    )
    AUTH_FAILURES = Counter(
        "swim_auth_failures_total",
        "Authentication failures",
        registry=REGISTRY,
    )
    RATE_LIMIT_HITS = Counter(
        "swim_rate_limit_hits_total",
        "Rate limit rejections",
        registry=REGISTRY,
    )

    # Histograms
    REQUEST_LATENCY = Histogram(
        "swim_request_duration_seconds",
        "Request latency in seconds",
        ["method", "endpoint"],
        buckets=(0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120),
        registry=REGISTRY,
    )
    PIPELINE_LATENCY = Histogram(
        "swim_pipeline_duration_seconds",
        "Pipeline execution latency",
        buckets=(5, 10, 30, 60, 120, 300),
        registry=REGISTRY,
    )
    AGENT_LATENCY = Histogram(
        "swim_agent_duration_seconds",
        "Per-agent call latency",
        ["agent"],
        buckets=(1, 2, 5, 10, 30, 60),
        registry=REGISTRY,
    )

    # Gauges
    ACTIVE_PIPELINES = Gauge(
        "swim_active_pipelines",
        "Currently running pipelines",
        registry=REGISTRY,
    )
    LATEST_RISK_SCORE = Gauge(
        "swim_latest_risk_score",
        "Most recent risk score",
        ["location"],
        registry=REGISTRY,
    )
    RISK_LEVEL_GAUGE = Gauge(
        "swim_risk_level",
        "Current risk level as numeric (0=low, 1=moderate, 2=high, 3=critical)",
        ["location"],
        registry=REGISTRY,
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_metrics_text() -> str:
    """Return Prometheus text exposition format."""
    if not PROMETHEUS_AVAILABLE:
        return "# prometheus_client not installed\n"
    return generate_latest(REGISTRY).decode("utf-8")


def get_content_type() -> str:
    if not PROMETHEUS_AVAILABLE:
        return "text/plain"
    return CONTENT_TYPE_LATEST


def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record an API request."""
    if not PROMETHEUS_AVAILABLE:
        return
    REQUEST_COUNT.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, endpoint=endpoint).observe(duration)


def record_pipeline(status: str, duration: float, location: str = "", risk_score: float = 0.0, risk_level: str = ""):
    """Record a pipeline run."""
    if not PROMETHEUS_AVAILABLE:
        return
    PIPELINE_RUNS.labels(status=status).inc()
    PIPELINE_LATENCY.observe(duration)
    if location:
        LATEST_RISK_SCORE.labels(location=location).set(risk_score)
        level_num = {"low": 0, "moderate": 1, "high": 2, "critical": 3}.get(risk_level, -1)
        RISK_LEVEL_GAUGE.labels(location=location).set(level_num)


def record_agent_call(agent: str, status: str, duration: float):
    """Record an individual agent call."""
    if not PROMETHEUS_AVAILABLE:
        return
    AGENT_CALLS.labels(agent=agent, status=status).inc()
    AGENT_LATENCY.labels(agent=agent).observe(duration)


def record_auth_failure():
    if PROMETHEUS_AVAILABLE:
        AUTH_FAILURES.inc()


def record_rate_limit():
    if PROMETHEUS_AVAILABLE:
        RATE_LIMIT_HITS.inc()


def pipeline_in_progress():
    """Context manager for tracking active pipelines."""
    if not PROMETHEUS_AVAILABLE:
        class _Noop:
            def __enter__(self): return self
            def __exit__(self, *a): pass
        return _Noop()
    return _ActivePipeline()


class _ActivePipeline:
    def __enter__(self):
        ACTIVE_PIPELINES.inc()
        return self

    def __exit__(self, *args):
        ACTIVE_PIPELINES.dec()
