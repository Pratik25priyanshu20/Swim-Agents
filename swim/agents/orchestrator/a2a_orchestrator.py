# swim/agents/orchestrator/a2a_orchestrator.py

"""A2A client orchestrator with parallel execution, retry, circuit breaker, and tracing."""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

import httpx

from a2a.client import A2AClient
from a2a.types import MessageSendParams, SendMessageRequest

from swim.shared.a2a_config import get_agent_url, AGENT_PORTS
from swim.agents.orchestrator.risk_fusion import compute_calibrated_risk_fusion
from swim.observability.logger import new_trace_id, set_trace_context, SpanContext
from swim.observability.error_tracking import error_tracker
from swim.observability.metrics import pipeline_in_progress, record_agent_call, record_pipeline
from swim.shared.alerting import alert_manager
from swim.shared.database.connection import db

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

class CircuitBreaker:
    """Simple circuit breaker: opens after `threshold` consecutive failures."""

    def __init__(self, threshold: int = 3, reset_timeout: float = 60.0):
        self.threshold = threshold
        self.reset_timeout = reset_timeout
        self._failures: Dict[str, int] = {}
        self._opened_at: Dict[str, float] = {}

    def is_open(self, agent: str) -> bool:
        if agent not in self._opened_at:
            return False
        if time.monotonic() - self._opened_at[agent] > self.reset_timeout:
            # Half-open: allow one attempt
            del self._opened_at[agent]
            self._failures[agent] = 0
            return False
        return True

    def record_success(self, agent: str):
        self._failures[agent] = 0
        self._opened_at.pop(agent, None)

    def record_failure(self, agent: str):
        self._failures[agent] = self._failures.get(agent, 0) + 1
        if self._failures[agent] >= self.threshold:
            self._opened_at[agent] = time.monotonic()
            logger.warning("Circuit OPEN for %s after %d failures", agent, self._failures[agent])


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

class SWIMOrchestrator:
    """Orchestrates SWIM agents via A2A with parallel execution and resilience."""

    def __init__(self):
        self._clients: Dict[str, A2AClient] = {}
        self._http = httpx.AsyncClient(timeout=120)
        self._circuit = CircuitBreaker(threshold=3, reset_timeout=60)
        self._webhooks: List[str] = []

    def register_webhook(self, url: str):
        """Register a URL for push notification on pipeline completion."""
        self._webhooks.append(url)

    async def _get_client(self, agent_name: str) -> A2AClient:
        if agent_name not in self._clients:
            url = get_agent_url(agent_name)
            self._clients[agent_name] = await A2AClient.get_client_from_agent_card_url(
                self._http, url
            )
        return self._clients[agent_name]

    async def _send_to_agent(self, agent_name: str, query: str, retries: int = 3) -> str:
        """Send a query with retry + circuit breaker."""
        if self._circuit.is_open(agent_name):
            raise ConnectionError(f"Circuit open for {agent_name} — skipping")

        last_exc: Optional[Exception] = None
        for attempt in range(1, retries + 1):
            try:
                client = await self._get_client(agent_name)
                payload = {
                    "message": {
                        "role": "user",
                        "parts": [{"type": "text", "text": query}],
                        "messageId": uuid4().hex,
                    },
                }
                request = SendMessageRequest(params=MessageSendParams(**payload))
                response = await client.send_message(request)
                text = self._extract_text(response)
                self._circuit.record_success(agent_name)
                return text
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "%s attempt %d/%d failed: %s", agent_name, attempt, retries, exc
                )
                if attempt < retries:
                    await asyncio.sleep(min(2 ** attempt, 8))

        self._circuit.record_failure(agent_name)
        error_tracker.record(agent_name, last_exc, recoverable=True)
        raise last_exc

    @staticmethod
    def _extract_text(response) -> str:
        result = response.result
        if result is None:
            return ""
        if hasattr(result, "artifacts") and result.artifacts:
            texts = []
            for artifact in result.artifacts:
                if artifact.parts:
                    for part in artifact.parts:
                        if hasattr(part, "text"):
                            texts.append(part.text)
            if texts:
                return "\n".join(texts)
        if hasattr(result, "parts") and result.parts:
            texts = []
            for part in result.parts:
                if hasattr(part, "text"):
                    texts.append(part.text)
            if texts:
                return "\n".join(texts)
        return str(result)

    # ------------------------------------------------------------------
    # Agent call helpers
    # ------------------------------------------------------------------

    async def _call_agent(
        self, agent_name: str, query: str
    ) -> Dict[str, Any]:
        """Call a single agent, returning result dict with timing."""
        start = datetime.now()
        try:
            with SpanContext(f"call_{agent_name}", agent=agent_name):
                summary = await self._send_to_agent(agent_name, query)
            duration = round((datetime.now() - start).total_seconds(), 2)
            record_agent_call(agent_name, "success", duration)
            return {
                "status": "success",
                "summary": summary,
                "duration": duration,
            }
        except Exception as exc:
            duration = round((datetime.now() - start).total_seconds(), 2)
            record_agent_call(agent_name, "error", duration)
            logger.error("%s failed: %s", agent_name.upper(), exc)
            return {
                "status": "error",
                "error": str(exc),
                "duration": duration,
            }

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

    async def run_pipeline(
        self,
        location: Dict[str, Any],
        horizon_days: int = 7,
        image_name: Optional[str] = None,
        user_query: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Run the full SWIM pipeline with parallel execution where possible."""
        trace_id = new_trace_id()
        set_trace_context(trace_id, "orchestrator")
        pipeline_start = datetime.now()
        _pip = pipeline_in_progress()
        _pip.__enter__()
        results: Dict[str, Any] = {}
        errors: list = []
        execution_time: Dict[str, float] = {}

        loc_str = (
            f"{location.get('name', 'unknown')} "
            f"({location.get('latitude')}, {location.get('longitude')})"
        )

        # ---- Stage 1: HOMOGEN (must run first) ----
        homogen_query = user_query or f"Run harmonization pipeline for water quality data near {loc_str}"
        results["homogen"] = await self._call_agent("homogen", homogen_query)
        execution_time["homogen"] = results["homogen"]["duration"]
        if results["homogen"]["status"] != "success":
            errors.append("homogen")

        homogen_ctx = results["homogen"].get("summary", "unavailable")[:200]

        # ---- RAG context injection ----
        rag_context = ""
        try:
            from swim.rag.query_router import retrieve_context
            base_query = user_query or f"water quality bloom risk {location.get('name', '')}"
            rag_context = retrieve_context(base_query, top_k=3)
        except Exception as exc:
            logger.debug("RAG context retrieval skipped: %s", exc)

        # ---- Stage 2: CALIBRO + VISIOS in parallel ----
        calibro_query = (
            f"Calibrate satellite data and assess water quality at {loc_str}. "
            f"HOMOGEN context: {homogen_ctx}"
        )
        if rag_context:
            calibro_query = f"Relevant context:\n{rag_context}\n\n{calibro_query}"

        parallel_tasks = [self._call_agent("calibro", calibro_query)]

        if image_name:
            parallel_tasks.append(self._call_agent("visios", f"Analyze image: {image_name}"))
        else:
            # Return skipped result immediately
            async def _skipped():
                return {"status": "skipped", "reason": "no image provided", "duration": 0.0}
            parallel_tasks.append(_skipped())

        calibro_result, visios_result = await asyncio.gather(*parallel_tasks)

        results["calibro"] = calibro_result
        results["visios"] = visios_result
        execution_time["calibro"] = calibro_result["duration"]
        execution_time["visios"] = visios_result["duration"]
        if calibro_result["status"] != "success":
            errors.append("calibro")
        if visios_result["status"] not in ("success", "skipped"):
            errors.append("visios")

        # ---- Stage 3: PREDIKT ----
        lake_name = location.get("name", "unknown")
        predikt_query = f"Predict bloom probability for {lake_name} with {horizon_days}-day horizon"
        if rag_context:
            predikt_query = f"Relevant context:\n{rag_context}\n\n{predikt_query}"
        results["predikt"] = await self._call_agent("predikt", predikt_query)
        execution_time["predikt"] = results["predikt"]["duration"]
        if results["predikt"]["status"] != "success":
            errors.append("predikt")

        # ---- Stage 4: Calibrated risk fusion ----
        risk = compute_calibrated_risk_fusion(results)

        pipeline_end = datetime.now()
        output = {
            "risk_assessment": risk,
            "agent_results": results,
            "execution_summary": {
                "time": round((pipeline_end - pipeline_start).total_seconds(), 2),
                "per_agent": execution_time,
                "errors": errors,
                "success_rate": f"{((4 - len(errors)) / 4) * 100:.0f}%",
                "trace_id": trace_id,
            },
            "metadata": {
                "started": pipeline_start.isoformat(),
                "ended": pipeline_end.isoformat(),
                "location": location,
                "protocol": "a2a",
            },
        }

        # Record metrics
        _pip.__exit__(None, None, None)
        pipeline_status = "success" if not errors else "partial"
        record_pipeline(
            status=pipeline_status,
            duration=output["execution_summary"]["time"],
            location=location.get("name", ""),
            risk_score=risk.get("score", 0.0),
            risk_level=risk.get("level", ""),
        )

        # Push notifications & alerting
        await self._notify_webhooks(output)
        await alert_manager.process_pipeline_result(output)

        # Persist to database
        try:
            db.record_pipeline_run(output)
        except Exception as exc:
            logger.warning("Failed to persist pipeline run: %s", exc)

        return output

    # ------------------------------------------------------------------
    # Webhooks
    # ------------------------------------------------------------------

    async def _notify_webhooks(self, result: Dict[str, Any]):
        """POST pipeline result to registered webhook URLs."""
        if not self._webhooks:
            return
        payload = json.dumps(result, default=str).encode()
        for url in self._webhooks:
            try:
                await self._http.post(
                    url,
                    content=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=10,
                )
                logger.info("Webhook notified: %s", url)
            except Exception as exc:
                logger.warning("Webhook failed for %s: %s", url, exc)

    async def close(self):
        await self._http.aclose()
