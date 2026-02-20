# swim/orchestrator/swim_platform_graph.py

from langgraph.graph import StateGraph, END
from typing import TypedDict, Dict, Any, Optional, List
from datetime import datetime
from langchain_core.messages import HumanMessage
import logging

# === Import Agents ===
from swim.agents.homogen.homogen_agent_graph import app as homogen_app
from swim.agents.calibro.calibro_agent_graph import app as calibro_app
from swim.agents.visios.visios_agent_graph import app as visios_app
from swim.agents.predikt.predikt_langagent_graph import app as predikt_app
from swim.agents.predikt.predikt_agent import PrediktAgent

# === Setup Logging ===
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
logger = logging.getLogger("SWIM-Orchestrator")

# === Shared State ===
class OrchestratorState(TypedDict):
    location: Dict[str, float]
    timestamp: Optional[str]
    image_name: Optional[str]
    user_query: Optional[str]
    results: Dict[str, Any]
    errors: List[str]
    execution_time: Dict[str, float]
    pipeline_start: Optional[str]
    pipeline_end: Optional[str]

# === Agent Initialization ===
visios = visios_app()
predikt = PrediktAgent()

# === Agent Nodes ===

async def homogen_node(state: OrchestratorState) -> OrchestratorState:
    start = datetime.now()
    logger.info("🌊 HOMOGEN started...")

    try:
        messages = []
        if state.get("user_query"):
            messages.append(HumanMessage(content=state["user_query"]))
        result = homogen_app.invoke({"messages": messages})
        last_msg = result["messages"][-1] if result.get("messages") else None

        state["results"]["homogen"] = {
            "status": "success",
            "data": result,
            "summary": last_msg.content if last_msg else "Harmonization complete",
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ HOMOGEN completed.")
    except Exception as e:
        logger.error(f"❌ HOMOGEN failed: {e}", exc_info=True)
        state["errors"].append("homogen")
        state["results"]["homogen"] = {
            "status": "error",
            "error": str(e),
            "fallback": "Using cached harmonized data"
        }
    finally:
        state["execution_time"]["homogen"] = round((datetime.now() - start).total_seconds(), 2)
    return state

async def calibro_node(state: OrchestratorState) -> OrchestratorState:
    start = datetime.now()
    logger.info("🛰️ CALIBRO started...")

    try:
        messages = []
        if state["results"].get("homogen", {}).get("status") == "success":
            messages.append(HumanMessage(content=f"Harmonized data from HOMOGEN available for {state['location']}"))
        result = calibro_app.invoke({"messages": messages})
        state["results"]["calibro"] = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ CALIBRO completed.")
    except Exception as e:
        logger.error(f"❌ CALIBRO failed: {e}", exc_info=True)
        state["errors"].append("calibro")
        state["results"]["calibro"] = {
            "status": "error",
            "error": str(e),
            "fallback": "Using uncalibrated satellite data"
        }
    finally:
        state["execution_time"]["calibro"] = round((datetime.now() - start).total_seconds(), 2)
    return state

async def visios_node(state: OrchestratorState) -> OrchestratorState:
    """
    VISIOS: Visual image analysis (LangGraph version)
    - Detects bloom risk in uploaded images
    - Integrates with HOMOGEN and CALIBRO context
    """
    agent_start = datetime.now()
    logger.info("👁️ Starting VISIOS agent...")

    try:
        messages = []

        if state.get("image_name"):
            messages.append(HumanMessage(content=f"Analyze image: {state['image_name']}"))

        if state.get("user_query"):
            messages.append(HumanMessage(content=state["user_query"]))

        result = visios_app.invoke({"messages": messages})

        last_message = result["messages"][-1] if result.get("messages") else None

        state["results"]["visios"] = {
            "status": "success",
            "data": result,
            "summary": last_message.content if last_message else "Image analysis complete",
            "timestamp": datetime.now().isoformat()
        }

        logger.info("✅ VISIOS completed successfully")

    except Exception as e:
        logger.error(f"❌ VISIOS failed: {e}", exc_info=True)
        state["errors"].append("visios")
        state["results"]["visios"] = {
            "status": "error",
            "error": str(e),
            "fallback": "Visual analysis unavailable"
        }

    finally:
        duration = (datetime.now() - agent_start).total_seconds()
        state["execution_time"]["visios"] = round(duration, 2)

    return state

async def predikt_node(state: OrchestratorState) -> OrchestratorState:
    start = datetime.now()
    logger.info("🔮 PREDIKT started...")

    try:
        result = predikt.predict_bloom_probability(location=state["location"], horizon_days=7)
        state["results"]["predikt"] = {
            "status": "success",
            "data": result,
            "timestamp": datetime.now().isoformat()
        }
        logger.info("✅ PREDIKT completed.")
    except Exception as e:
        logger.error(f"❌ PREDIKT failed: {e}", exc_info=True)
        state["errors"].append("predikt")
        state["results"]["predikt"] = {
            "status": "error",
            "error": str(e),
            "fallback": "Using historical bloom statistics"
        }
    finally:
        state["execution_time"]["predikt"] = round((datetime.now() - start).total_seconds(), 2)
    return state

async def risk_node(state: OrchestratorState) -> OrchestratorState:
    logger.info("⚖️ RISK FUSION started...")
    scores, evidence = [], []

    weights = {"predikt": 0.4, "calibro": 0.3, "visios": 0.2, "homogen": 0.1}
    results = state["results"]

    if results.get("predikt", {}).get("status") == "success":
        p = results["predikt"]["data"].get("bloom_probability", 0)
        scores.append((p, weights["predikt"]))
        evidence.append(f"PREDIKT forecast: {p:.1%}")

    if results.get("calibro", {}).get("status") == "success":
        chl = results["calibro"]["data"].get("chlorophyll_a", 0)
        chl_score = min(chl / 50, 1.0)
        scores.append((chl_score, weights["calibro"]))
        evidence.append(f"Satellite chlorophyll-a: {chl:.1f} µg/L")

    if results.get("visios", {}).get("status") == "success":
        vis = results["visios"]["data"].get("bloom_probability", 0)
        scores.append((vis, weights["visios"]))
        evidence.append(f"Visual detection: {vis:.1%}")

    if results.get("homogen", {}).get("status") == "success":
        quality = results["homogen"]["data"].get("quality_score", 0.5)
        scores.append((quality, weights["homogen"]))
        evidence.append(f"Data quality score: {quality:.1%}")

    if not scores:
        logger.warning("⚠️ No valid agent results for risk assessment.")
        state["results"]["risk"] = {
            "level": "unknown", "score": 0.0, "confidence": "0%",
            "evidence": ["No valid data sources"], "recommendation": "Increase monitoring"
        }
        return state

    avg = sum(v * w for v, w in scores) / sum(w for _, w in scores)
    level = "low"
    if avg > 0.75: level = "critical"
    elif avg > 0.5: level = "high"
    elif avg > 0.3: level = "moderate"

    recommendation = {
        "low": "No action needed",
        "moderate": "Continue standard monitoring",
        "high": "Increase monitoring frequency",
        "critical": "Issue immediate public advisory"
    }[level]

    state["results"]["risk"] = {
        "level": level,
        "score": round(avg, 3),
        "confidence": f"{min(0.9, 0.4 + len(scores)*0.15):.0%}",
        "evidence": evidence,
        "recommendation": recommendation,
        "sources_used": len(scores),
        "timestamp": datetime.now().isoformat()
    }
    state["pipeline_end"] = datetime.now().isoformat()
    logger.info(f"✅ RISK FUSION complete → Level: {level.upper()}")
    return state


# === BUILD GRAPH ===
workflow = StateGraph(OrchestratorState)
workflow.set_entry_point("homogen")
workflow.add_node("homogen", homogen_node)
workflow.add_node("calibro", calibro_node)
workflow.add_node("visios", visios_node)
workflow.add_node("predikt", predikt_node)
workflow.add_node("risk", risk_node)
workflow.add_edge("homogen", "calibro")
workflow.add_edge("calibro", "visios")
workflow.add_edge("visios", "predikt")
workflow.add_edge("predikt", "risk")
workflow.add_edge("risk", END)
app = workflow.compile()

# === STATE INIT + RUNNER ===
def initialize_state(location: Dict[str, float], image_name: Optional[str], user_query: Optional[str]) -> OrchestratorState:
    return {
        "location": location,
        "timestamp": datetime.now().isoformat(),
        "image_name": image_name,
        "user_query": user_query,
        "results": {},
        "errors": [],
        "execution_time": {},
        "pipeline_start": datetime.now().isoformat(),
        "pipeline_end": None
    }

def format_output(state: OrchestratorState) -> Dict[str, Any]:
    return {
        "risk_assessment": state["results"].get("risk", {}),
        "agent_results": {k: v for k, v in state["results"].items() if k != "risk"},
        "execution_summary": {
            "time": sum(state["execution_time"].values()),
            "per_agent": state["execution_time"],
            "errors": state["errors"],
            "success_rate": f"{((4 - len(state['errors'])) / 4) * 100:.0f}%",
        },
        "metadata": {
            "started": state["pipeline_start"],
            "ended": state["pipeline_end"],
            "location": state["location"]
        }
    }

# === MAIN RUNNER ===
if __name__ == "__main__":
    import asyncio

    async def run():
        state = initialize_state(
            location={"latitude": 47.6597, "longitude": 9.1755},
            image_name="lake_constance_sample.jpg",
            user_query="Assess HAB risk in Lake Constance"
        )
        result = await app.ainvoke(state)
        output = format_output(result)

        print("\n" + "="*60)
        print("🎯 SWIM Platform Pipeline Complete")
        print(f"Risk Level: {output['risk_assessment']['level'].upper()}")
        print(f"Score: {output['risk_assessment']['score']}")
        print(f"Confidence: {output['risk_assessment']['confidence']}")
        print("Recommendation:", output['risk_assessment']['recommendation'])

    asyncio.run(run())