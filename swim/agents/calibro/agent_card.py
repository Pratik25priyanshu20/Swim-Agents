# swim/agents/calibro/agent_card.py

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication
from swim.shared.a2a_config import get_agent_url

skills = [
    AgentSkill(
        id="satellite_calibration",
        name="Satellite Calibration",
        description="Calibrate and analyze satellite-derived water quality indices (chlorophyll-a, turbidity) for a given location.",
        tags=["satellite", "calibration", "remote-sensing"],
        examples=["Calibrate satellite data for Bodensee", "Get satellite metrics at 47.5, 9.2"],
    ),
    AgentSkill(
        id="trend_analysis",
        name="Water Quality Trend Analysis",
        description="Perform statistical trend analysis on water quality parameters over time.",
        tags=["trend", "statistics", "time-series"],
        examples=["Analyze chlorophyll trends for Chiemsee", "Show turbidity trend"],
    ),
    AgentSkill(
        id="bloom_risk_satellite",
        name="Satellite Bloom Risk Assessment",
        description="Assess harmful algal bloom risk using satellite-derived indicators.",
        tags=["habs", "risk", "satellite"],
        examples=["Assess bloom risk from satellite for Bodensee"],
    ),
    AgentSkill(
        id="lake_discovery",
        name="Discover Nearby Lakes",
        description="Find water bodies near a given geographic coordinate using OpenStreetMap.",
        tags=["discovery", "geospatial", "osm"],
        examples=["Find lakes near 47.5, 9.2"],
    ),
]

agent_card = AgentCard(
    name="CALIBRO",
    description="Satellite calibration agent for the SWIM platform. Processes Sentinel-2 satellite imagery to derive water quality metrics and bloom risk indicators.",
    url=get_agent_url("calibro"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=skills,
    authentication=AgentAuthentication(schemes=["public"]),
)
