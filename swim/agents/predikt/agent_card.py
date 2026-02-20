# swim/agents/predikt/agent_card.py

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication
from swim.shared.a2a_config import get_agent_url

skills = [
    AgentSkill(
        id="predict_lake",
        name="Predict Single Lake",
        description="Predict harmful algal bloom probability for a specific German lake with a given forecast horizon.",
        tags=["prediction", "lake", "forecast"],
        examples=["Predict bloom for Bodensee", "Forecast Chiemsee 7 days"],
    ),
    AgentSkill(
        id="forecast_all",
        name="Forecast All Lakes",
        description="Run bloom forecasts for all monitored German lakes simultaneously.",
        tags=["prediction", "batch", "forecast"],
        examples=["Forecast all lakes", "Run predictions for all German lakes"],
    ),
    AgentSkill(
        id="risk_analysis",
        name="Risk Factor Analysis",
        description="Provide detailed risk factor breakdown for a lake's bloom prediction, including contributing environmental parameters.",
        tags=["risk", "analysis", "factors"],
        examples=["Analyze risk factors for Bodensee", "What drives bloom risk in Mueritz?"],
    ),
]

agent_card = AgentCard(
    name="PREDIKT",
    description="HAB forecasting agent for the SWIM platform. Predicts harmful algal bloom probability using LSTM, SARIMA, ensemble, and rule-based models.",
    url=get_agent_url("predikt"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=skills,
    authentication=AgentAuthentication(schemes=["public"]),
)
