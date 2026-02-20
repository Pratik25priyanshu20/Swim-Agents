# swim/agents/visios/agent_card.py

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication
from swim.shared.a2a_config import get_agent_url

skills = [
    AgentSkill(
        id="image_analysis",
        name="Single Image Analysis",
        description="Analyze a single lake photograph for harmful algal bloom indicators using vision models.",
        tags=["image", "vision", "bloom-detection"],
        examples=["Analyze image lake_constance.jpg", "Check this photo for blooms"],
    ),
    AgentSkill(
        id="batch_analysis",
        name="Batch Image Analysis",
        description="Summarize bloom detection results across all available lake images.",
        tags=["batch", "summary", "images"],
        examples=["Summarize all images", "Batch analysis of lake photos"],
    ),
    AgentSkill(
        id="location_risk",
        name="Location Bloom Risk",
        description="Assess bloom risk at a geographic location based on visual evidence.",
        tags=["risk", "location", "geospatial"],
        examples=["Check bloom risk at 47.5, 9.2"],
    ),
]

agent_card = AgentCard(
    name="VISIOS",
    description="Visual bloom detection agent for the SWIM platform. Analyzes lake photographs using vision models to detect harmful algal bloom indicators.",
    url=get_agent_url("visios"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=skills,
    authentication=AgentAuthentication(schemes=["public"]),
)
