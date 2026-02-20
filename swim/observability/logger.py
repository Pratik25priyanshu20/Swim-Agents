# swim/observability/logger.py

"""Structured logging with OpenTelemetry-compatible trace propagation."""

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from functools import wraps
from typing import Any, Dict, Optional

# Trace context propagation
_trace_id: ContextVar[str] = ContextVar("trace_id", default="")
_span_id: ContextVar[str] = ContextVar("span_id", default="")
_agent_name: ContextVar[str] = ContextVar("agent_name", default="")


def new_trace_id() -> str:
    return uuid.uuid4().hex[:16]


def new_span_id() -> str:
    return uuid.uuid4().hex[:8]


def set_trace_context(trace_id: str, agent: str = "") -> None:
    _trace_id.set(trace_id)
    _agent_name.set(agent)


def get_trace_id() -> str:
    return _trace_id.get()


class StructuredFormatter(logging.Formatter):
    """JSON structured log formatter with trace context."""

    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "trace_id": _trace_id.get(""),
            "span_id": _span_id.get(""),
            "agent": _agent_name.get(""),
        }
        if record.exc_info and record.exc_info[0]:
            log_entry["exception"] = self.formatException(record.exc_info)
        # Merge extra fields
        for key in ("duration_ms", "status", "agent_name", "task_type", "error_type"):
            if hasattr(record, key):
                log_entry[key] = getattr(record, key)
        return json.dumps(log_entry)


def setup_logging(level: int = logging.INFO, structured: bool = True) -> None:
    """Configure root logger with structured or human-readable output."""
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers.clear()

    handler = logging.StreamHandler(sys.stdout)
    if structured:
        handler.setFormatter(StructuredFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s | %(message)s")
        )
    root.addHandler(handler)


def traced(agent_name: str):
    """Decorator that adds trace context and timing to a function."""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace = _trace_id.get() or new_trace_id()
            span = new_span_id()
            _trace_id.set(trace)
            _span_id.set(span)
            _agent_name.set(agent_name)
            logger = logging.getLogger(agent_name)
            start = time.monotonic()
            logger.info("Starting %s", func.__name__, extra={"status": "started"})
            try:
                result = await func(*args, **kwargs)
                duration = (time.monotonic() - start) * 1000
                logger.info(
                    "Completed %s in %.0fms",
                    func.__name__,
                    duration,
                    extra={"status": "completed", "duration_ms": round(duration)},
                )
                return result
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                logger.error(
                    "Failed %s after %.0fms: %s",
                    func.__name__,
                    duration,
                    exc,
                    extra={
                        "status": "failed",
                        "duration_ms": round(duration),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            trace = _trace_id.get() or new_trace_id()
            span = new_span_id()
            _trace_id.set(trace)
            _span_id.set(span)
            _agent_name.set(agent_name)
            logger = logging.getLogger(agent_name)
            start = time.monotonic()
            logger.info("Starting %s", func.__name__, extra={"status": "started"})
            try:
                result = func(*args, **kwargs)
                duration = (time.monotonic() - start) * 1000
                logger.info(
                    "Completed %s in %.0fms",
                    func.__name__,
                    duration,
                    extra={"status": "completed", "duration_ms": round(duration)},
                )
                return result
            except Exception as exc:
                duration = (time.monotonic() - start) * 1000
                logger.error(
                    "Failed %s after %.0fms: %s",
                    func.__name__,
                    duration,
                    exc,
                    extra={
                        "status": "failed",
                        "duration_ms": round(duration),
                        "error_type": type(exc).__name__,
                    },
                    exc_info=True,
                )
                raise
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    return decorator


class SpanContext:
    """Context manager for creating trace spans."""

    def __init__(self, name: str, agent: str = ""):
        self.name = name
        self.agent = agent
        self.start_time = 0.0
        self.logger = logging.getLogger(agent or "swim")

    def __enter__(self):
        self.start_time = time.monotonic()
        parent_span = _span_id.get("")
        _span_id.set(new_span_id())
        if not _trace_id.get():
            _trace_id.set(new_trace_id())
        if self.agent:
            _agent_name.set(self.agent)
        self.logger.info("Span started: %s", self.name, extra={"status": "started"})
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (time.monotonic() - self.start_time) * 1000
        if exc_type:
            self.logger.error(
                "Span failed: %s (%.0fms)",
                self.name,
                duration,
                extra={"status": "failed", "duration_ms": round(duration)},
            )
        else:
            self.logger.info(
                "Span completed: %s (%.0fms)",
                self.name,
                duration,
                extra={"status": "completed", "duration_ms": round(duration)},
            )
        return False
