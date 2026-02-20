# swim/agents/orchestrator/agent_card.py

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication
from swim.shared.a2a_config import get_agent_url

skills = [
    AgentSkill(
        id="full_pipeline",
        name="Full SWIM Pipeline",
        description="Run the complete SWIM pipeline: HOMOGEN -> CALIBRO -> VISIOS -> PREDIKT -> Risk Fusion. Returns a unified risk assessment combining all agents.",
        tags=["pipeline", "orchestration", "risk"],
        examples=[
            "Assess HAB risk in Bodensee",
            "Run full pipeline for Chiemsee with 7-day forecast",
            "Analyze bloom risk at 47.5, 9.2",
        ],
    ),
]

agent_card = AgentCard(
    name="SWIM Orchestrator",
    description="Master orchestrator for the SWIM platform. Coordinates HOMOGEN, CALIBRO, VISIOS, and PREDIKT agents via A2A protocol and produces a unified risk assessment.",
    url=get_agent_url("orchestrator"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=skills,
    authentication=AgentAuthentication(schemes=["public"]),
)
