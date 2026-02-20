# swim/observability/error_tracking.py

"""Centralized error tracking and aggregation for agent pipelines."""

import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class AgentError:
    agent: str
    error_type: str
    message: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    trace_id: str = ""
    recoverable: bool = True


class ErrorTracker:
    """Tracks errors across agents and provides aggregated diagnostics."""

    def __init__(self, max_history: int = 500):
        self._errors: List[AgentError] = []
        self._counts: Dict[str, int] = defaultdict(int)
        self._max_history = max_history

    def record(self, agent: str, error: Exception, trace_id: str = "", recoverable: bool = True):
        entry = AgentError(
            agent=agent,
            error_type=type(error).__name__,
            message=str(error)[:500],
            trace_id=trace_id,
            recoverable=recoverable,
        )
        self._errors.append(entry)
        self._counts[f"{agent}:{entry.error_type}"] += 1
        if len(self._errors) > self._max_history:
            self._errors = self._errors[-self._max_history:]
        logger.warning(
            "Error recorded: %s in %s — %s", entry.error_type, agent, entry.message[:100]
        )

    def get_summary(self) -> Dict[str, Any]:
        return {
            "total_errors": len(self._errors),
            "by_agent": self._agent_breakdown(),
            "by_type": dict(self._counts),
            "recent": [
                {"agent": e.agent, "type": e.error_type, "message": e.message[:80], "time": e.timestamp}
                for e in self._errors[-10:]
            ],
        }

    def _agent_breakdown(self) -> Dict[str, int]:
        breakdown: Dict[str, int] = defaultdict(int)
        for e in self._errors:
            breakdown[e.agent] += 1
        return dict(breakdown)

    def get_agent_health(self, agent: str) -> Dict[str, Any]:
        agent_errors = [e for e in self._errors if e.agent == agent]
        recent = agent_errors[-5:] if agent_errors else []
        return {
            "agent": agent,
            "total_errors": len(agent_errors),
            "recent_errors": len([e for e in agent_errors if e.timestamp > datetime.now().isoformat()[:10]]),
            "last_error": recent[-1].message[:100] if recent else None,
            "healthy": len(agent_errors) == 0 or all(e.recoverable for e in recent),
        }

    def clear(self):
        self._errors.clear()
        self._counts.clear()


# Global singleton
error_tracker = ErrorTracker()
