# swim/agents/orchestrator/a2a_server.py

"""A2A server for the SWIM orchestrator — exposes the full pipeline as an A2A skill."""

import asyncio
import json
import logging
import re

import uvicorn
from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils import new_agent_text_message

from swim.agents.orchestrator.agent_card import agent_card
from swim.agents.orchestrator.a2a_orchestrator import SWIMOrchestrator
from swim.agents.predikt.config import GERMAN_LAKES
from swim.shared.a2a_config import AGENT_PORTS
from swim.shared.sanitize import sanitize_query

logger = logging.getLogger(__name__)


def _parse_location(query: str) -> dict:
    """Extract lake name or coordinates from the query text."""
    query_lower = query.lower()
    for lake_name, meta in GERMAN_LAKES.items():
        if lake_name.lower() in query_lower:
            return {"name": lake_name, "latitude": meta["lat"], "longitude": meta["lon"]}

    # Try to find lat/lon numbers
    coords = re.findall(r"(-?\d+\.?\d*)", query)
    if len(coords) >= 2:
        lat, lon = float(coords[0]), float(coords[1])
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return {"name": "custom_location", "latitude": lat, "longitude": lon}

    # Default to Bodensee
    meta = GERMAN_LAKES["Bodensee"]
    return {"name": "Bodensee", "latitude": meta["lat"], "longitude": meta["lon"]}


class OrchestratorExecutor(AgentExecutor):
    """Wraps SWIMOrchestrator as an A2A executor."""

    def __init__(self):
        self._orchestrator = SWIMOrchestrator()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    query += part.text

        if not query:
            query = "Assess HAB risk in Bodensee"

        try:
            query = sanitize_query(query)
        except ValueError as exc:
            event_queue.enqueue_event(new_agent_text_message(f"Input rejected: {exc}"))
            return

        logger.info("Orchestrator processing: %s", query[:120])

        location = _parse_location(query)

        # Extract horizon
        horizon = 7
        h_match = re.search(r"(\d+)\s*(?:day|d)\b", query, re.IGNORECASE)
        if h_match:
            h_val = int(h_match.group(1))
            if h_val in (3, 7, 14):
                horizon = h_val

        # Extract image name
        image_name = None
        img_match = re.search(r"image[:\s]+(\S+\.(?:jpg|jpeg|png|bmp))", query, re.IGNORECASE)
        if img_match:
            image_name = img_match.group(1)

        result = await self._orchestrator.run_pipeline(
            location=location,
            horizon_days=horizon,
            image_name=image_name,
            user_query=query,
        )

        summary = (
            f"SWIM Pipeline Complete\n"
            f"Location: {location.get('name')}\n"
            f"Risk Level: {result['risk_assessment']['level'].upper()}\n"
            f"Risk Score: {result['risk_assessment']['score']}\n"
            f"Confidence: {result['risk_assessment']['confidence']}\n"
            f"Recommendation: {result['risk_assessment']['recommendation']}\n"
            f"Execution Time: {result['execution_summary']['time']}s\n\n"
            f"Full result:\n{json.dumps(result, indent=2, default=str)}"
        )

        event_queue.enqueue_event(new_agent_text_message(summary))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


def main():
    handler = DefaultRequestHandler(
        agent_executor=OrchestratorExecutor(),
        task_store=InMemoryTaskStore(),
    )
    app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    port = AGENT_PORTS["orchestrator"]
    logger.info("Starting Orchestrator A2A server on port %d", port)
    uvicorn.run(app.build(), host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
