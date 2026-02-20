# swim/agents/homogen/a2a_server.py

"""A2A server wrapping the HOMOGEN LangGraph agent."""

import asyncio
import logging

import uvicorn
from typing_extensions import override

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.utils import new_agent_text_message

from langchain_core.messages import HumanMessage

from swim.agents.homogen.agent_card import agent_card
from swim.shared.a2a_config import AGENT_PORTS

logger = logging.getLogger(__name__)


def _get_graph():
    from swim.agents.homogen.homogen_agent_graph import app
    return app


class HomogenAgentExecutor(AgentExecutor):
    """Wraps the HOMOGEN LangGraph as an A2A executor."""

    def __init__(self):
        self._graph = None

    def _ensure_graph(self):
        if self._graph is None:
            self._graph = _get_graph()

    @override
    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        self._ensure_graph()

        query = ""
        if context.message and context.message.parts:
            for part in context.message.parts:
                if hasattr(part, "text"):
                    query += part.text

        if not query:
            query = "Run the harmonization pipeline"

        logger.info("HOMOGEN processing: %s", query[:120])

        result = await asyncio.to_thread(
            self._graph.invoke,
            {"messages": [HumanMessage(content=query)]},
        )

        messages = result.get("messages", [])
        text = messages[-1].content if messages else "Harmonization complete."

        event_queue.enqueue_event(new_agent_text_message(text))

    @override
    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("cancel not supported")


def main():
    from swim.shared.health import HOMOGEN_HEALTH

    handler = DefaultRequestHandler(
        agent_executor=HomogenAgentExecutor(),
        task_store=InMemoryTaskStore(),
    )
    a2a_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=handler,
    )
    starlette_app = a2a_app.build()
    starlette_app.routes.append(HOMOGEN_HEALTH)
    port = AGENT_PORTS["homogen"]
    logger.info("Starting HOMOGEN A2A server on port %d", port)
    uvicorn.run(starlette_app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
