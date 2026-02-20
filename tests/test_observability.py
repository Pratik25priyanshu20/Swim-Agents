# tests/test_observability.py

"""Tests for the observability module: structured logging, error tracking, DAG visualizer."""

import json
import logging

import pytest

from swim.observability.logger import (
    StructuredFormatter,
    new_trace_id,
    new_span_id,
    set_trace_context,
    SpanContext,
)
from swim.observability.error_tracking import ErrorTracker
from swim.observability.dag_visualizer import (
    get_pipeline_dag,
    format_execution_trace,
    build_mermaid_diagram,
)


class TestStructuredLogger:
    def test_trace_id_generation(self):
        tid = new_trace_id()
        assert len(tid) == 16
        assert tid.isalnum()

    def test_span_id_generation(self):
        sid = new_span_id()
        assert len(sid) == 8

    def test_formatter_produces_json(self):
        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="hello", args=(), exc_info=None,
        )
        output = formatter.format(record)
        parsed = json.loads(output)
        assert parsed["message"] == "hello"
        assert parsed["level"] == "INFO"

    def test_span_context_sets_trace(self):
        set_trace_context("test123", "HOMOGEN")
        with SpanContext("test_span", agent="HOMOGEN"):
            pass  # Just verify no exception


class TestErrorTracker:
    def test_record_and_summary(self):
        tracker = ErrorTracker()
        tracker.record("PREDIKT", ValueError("bad input"))
        summary = tracker.get_summary()
        assert summary["total_errors"] == 1
        assert "PREDIKT" in summary["by_agent"]

    def test_agent_health_no_errors(self):
        tracker = ErrorTracker()
        health = tracker.get_agent_health("CALIBRO")
        assert health["healthy"] is True
        assert health["total_errors"] == 0

    def test_max_history_trimming(self):
        tracker = ErrorTracker(max_history=5)
        for i in range(10):
            tracker.record("AGENT", RuntimeError(f"err {i}"))
        assert len(tracker._errors) == 5

    def test_clear(self):
        tracker = ErrorTracker()
        tracker.record("X", ValueError("test"))
        tracker.clear()
        assert tracker.get_summary()["total_errors"] == 0


class TestDAGVisualizer:
    def test_pipeline_dag_structure(self):
        dag = get_pipeline_dag()
        assert "nodes" in dag
        assert "edges" in dag
        node_ids = {n["id"] for n in dag["nodes"]}
        assert {"homogen", "calibro", "visios", "predikt", "risk"} == node_ids

    def test_format_execution_trace(self):
        result = {
            "agent_results": {
                "homogen": {"status": "success"},
                "calibro": {"status": "success"},
                "visios": {"status": "skipped"},
                "predikt": {"status": "error"},
            },
            "execution_summary": {
                "per_agent": {"homogen": 1.5, "calibro": 2.0, "visios": 0.0, "predikt": 3.0},
                "time": 6.5,
            },
            "risk_assessment": {"level": "moderate", "score": 0.45},
        }
        trace = format_execution_trace(result)
        assert "HOMOGEN" in trace
        assert "MODERATE" in trace

    def test_mermaid_diagram_basic(self):
        diagram = build_mermaid_diagram()
        assert diagram.startswith("graph TD")
        assert "homogen" in diagram
        assert "risk" in diagram
