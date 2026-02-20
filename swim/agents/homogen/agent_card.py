# swim/agents/homogen/agent_card.py

from a2a.types import AgentCard, AgentSkill, AgentCapabilities, AgentAuthentication
from swim.shared.a2a_config import get_agent_url

skills = [
    AgentSkill(
        id="run_pipeline",
        name="Run Harmonization Pipeline",
        description="Ingest and harmonize raw water quality data from CSV, Excel, and API sources into standardized parquet format.",
        tags=["etl", "harmonization", "data"],
        examples=["Run the harmonization pipeline", "Harmonize all raw data"],
    ),
    AgentSkill(
        id="query_lake_data",
        name="Query Lake Data",
        description="Query harmonized water quality data for a specific lake or region.",
        tags=["query", "lake", "water-quality"],
        examples=["Show data for Bodensee", "What lakes are available?"],
    ),
    AgentSkill(
        id="habs_assessment",
        name="HABs Data Assessment",
        description="Summarize harmful algal bloom indicators from the harmonized dataset.",
        tags=["habs", "assessment", "summary"],
        examples=["Summarize HAB indicators", "Give me a HABs assessment"],
    ),
]

agent_card = AgentCard(
    name="HOMOGEN",
    description="Data harmonization agent for the SWIM platform. Ingests multi-source water quality data and produces standardized datasets.",
    url=get_agent_url("homogen"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=skills,
    authentication=AgentAuthentication(schemes=["public"]),
)

# German-language variant
agent_card_de = AgentCard(
    name="HOMOGEN",
    description="Datenharmonisierungsagent fuer die SWIM-Plattform. Importiert Wasserqualitaetsdaten aus verschiedenen Quellen und erzeugt standardisierte Datensaetze.",
    url=get_agent_url("homogen"),
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(),
    skills=[
        AgentSkill(
            id="run_pipeline",
            name="Harmonisierungspipeline ausfuehren",
            description="Rohdaten zur Wasserqualitaet aus CSV-, Excel- und API-Quellen importieren und harmonisieren.",
            tags=["etl", "harmonisierung", "daten"],
            examples=["Harmonisierungspipeline starten", "Alle Rohdaten harmonisieren"],
        ),
        AgentSkill(
            id="query_lake_data",
            name="Seedaten abfragen",
            description="Harmonisierte Wasserqualitaetsdaten fuer einen bestimmten See abfragen.",
            tags=["abfrage", "see", "wasserqualitaet"],
            examples=["Zeige Daten fuer Bodensee", "Welche Seen sind verfuegbar?"],
        ),
        AgentSkill(
            id="habs_assessment",
            name="HAB-Bewertung",
            description="Indikatoren fuer schaedliche Algenbueten aus dem harmonisierten Datensatz zusammenfassen.",
            tags=["habs", "bewertung", "zusammenfassung"],
            examples=["HAB-Indikatoren zusammenfassen", "Gib mir eine HAB-Bewertung"],
        ),
    ],
    authentication=AgentAuthentication(schemes=["public"]),
)
