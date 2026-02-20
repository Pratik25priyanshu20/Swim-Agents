# swim/agents/predikt/a2a_server.py

"""A2A server wrapping the PREDIKT LangGraph agent."""

import asyncio
import logging
import os

import uvicorn
from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils import new_agent_text_message

from langchain_core.messages import HumanMessage, AIMessage

from swim.agents.predikt.agent_card import agent_card
from swim.shared.a2a_config import AGENT_PORTS

logger = logging.getLogger(__name__)


def _build_graph():
    from swim.agents.predikt.predikt_langagent_graph import create_predikt_graph, engine
    return create_predikt_graph(), engine


def _initial_state(engine_obj, query: str) -> dict:
    return {
        "messages": [HumanMessage(content=query)],
        "current_prediction": None,
        "active_lakes": [],
        "prediction_horizon": 7,
        "agent_name": "PREDIKT",
        "task_type": "prediction",
        "requires_handoff": False,
        "handoff_target": None,
        "data_quality_score": 0.85,
        "using_ml_model": engine_obj.get_model_info().get("using_ml_model", False),
        "confidence_level": 0.85,
    }


class PrediktAgentExecutor(AgentExecutor):
    """Wraps the PREDIKT LangGraph as an A2A executor."""

    def __init__(self):
        self._graph = None
        self._engine = None
        self._thread_counter = 0

    def _ensure_graph(self):
        if self._graph is None:
            self._graph, self._engine = _build_graph()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._ensure_graph()

        query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    query += part.text

        if not query:
            query = "Forecast all lakes"

        logger.info("PREDIKT processing: %s", query[:120])

        self._thread_counter += 1
        config = {"configurable": {"thread_id": f"a2a_predikt_{self._thread_counter}"}}
        state = _initial_state(self._engine, query)

        result = await asyncio.to_thread(
            self._graph.invoke,
            state,
            config,
        )

        ai_messages = [m for m in result.get("messages", []) if isinstance(m, AIMessage)]
        text = ai_messages[-1].content if ai_messages else "Prediction complete."

        event_queue.enqueue_event(new_agent_text_message(text))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


def main():
    from swim.shared.health import PREDIKT_HEALTH

    handler = DefaultRequestHandler(
        agent_executor=PrediktAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    starlette_app = a2a_app.build()
    starlette_app.routes.append(PREDIKT_HEALTH)
    port = AGENT_PORTS["predikt"]
    logger.info("Starting PREDIKT A2A server on port %d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
