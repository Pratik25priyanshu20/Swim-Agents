# swim/agents/predikt/predikt_langgraph_agent.py
"""
PREDIKT Agent using LangGraph for multi-agent integration
"""

import os
from datetime import datetime, timedelta
from typing import TypedDict, Annotated, List, Dict, Any, Literal, Optional
import operator
from pathlib import Path
from dotenv import load_dotenv

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.tools import tool

from swim.agents.predikt.predikt_agent import PrediktAgent

load_dotenv()


# ============================================================
# STATE DEFINITION (Compatible with multi-agent system)
# ============================================================

class PrediktState(TypedDict):
    """State for PREDIKT agent - compatible with broader agent system"""
    
    # Message history
    messages: Annotated[List[BaseMessage], operator.add]
    
    # Agent-specific state
    current_prediction: Optional[Dict[str, Any]]
    active_lakes: List[str]
    prediction_horizon: int
    
    # Metadata for multi-agent coordination
    agent_name: str
    task_type: str  # "prediction", "analysis", "comparison", "report"
    requires_handoff: bool
    handoff_target: Optional[str]  # e.g., "HOMOGEN", "CALIBRO", "VISIOS"
    
    # Data quality and status
    data_quality_score: float
    using_ml_model: bool
    confidence_level: float


# ============================================================
# PREDICTION ENGINE (Same as before)
# ============================================================

class PrediktEngine:
    """Core prediction engine wired to real model artifacts."""
    
    def __init__(self):
        self.version = "4.0-LangGraph"
        self.status = "operational"
        self.agent = PrediktAgent()
        self.GERMAN_LAKES = self.agent.GERMAN_LAKES
        self.prediction_history = []
    
    def predict_bloom_probability(
        self,
        location: Dict[str, Any],
        horizon_days: int = 7
    ) -> Dict[str, Any]:
        """Make HAB prediction"""
        
        lake_name = location['name']
        
        result = self.agent.predict_bloom_probability(location, horizon_days=horizon_days)
        
        # Log prediction
        self.prediction_history.append({
            "location": lake_name,
            "horizon_days": horizon_days,
            "probability": result["bloom_probability"],
            "risk_level": result["risk_level"],
            "is_ml": result.get("is_real_ml", False),
            "timestamp": result["predicted_at"]
        })
        
        return result
    
    def get_model_info(self) -> Dict:
        info = self.agent.get_agent_info()
        return {
            "version": self.version,
            "using_ml_model": info.get("model_type") != "rule_based",
            "model_type": info.get("model_type", "rule_based"),
            "accuracy": info.get("model_config", {}).get("base_accuracy", 0.82),
            "is_real_ml": info.get("model_type") != "rule_based",
        }


# Initialize engine
engine = PrediktEngine()


# ============================================================
# TOOLS (for LangGraph ToolNode)
# ============================================================

@tool
def predict_single_lake(lake_name: str, forecast_days: int = 7) -> str:
    """
    Predict HAB risk for a specific German lake.
    
    Args:
        lake_name: Name of the lake (Bodensee, Chiemsee, etc.)
        forecast_days: Number of days ahead (3, 7, or 14)
    
    Returns:
        Detailed prediction with risk level and environmental factors
    """
    if lake_name not in engine.GERMAN_LAKES:
        available = ", ".join(engine.GERMAN_LAKES.keys())
        return f"❌ Lake '{lake_name}' not found. Available: {available}"
    
    if forecast_days not in [3, 7, 14]:
        return "❌ Forecast days must be 3, 7, or 14"
    
    lake_info = engine.GERMAN_LAKES[lake_name]
    location = {"name": lake_name, "latitude": lake_info["lat"], "longitude": lake_info["lon"]}
    
    result = engine.predict_bloom_probability(location, forecast_days)
    
    ml_badge = "🧠 REAL ML" if result['is_real_ml'] else "📊 Rule-Based"
    
    output = f"""🔮 **HAB Prediction for {lake_name}** {ml_badge}

**Risk Level:** {result['risk_level'].upper()} ({result['bloom_probability']:.1%})
**Forecast:** {result['prediction_horizon_days']} days (valid until {result['valid_until'][:10]})
**Confidence:** {result['confidence']:.1%} ± {result['uncertainty']:.1%}

**Contributing Factors:**
"""
    
    for i, factor in enumerate(result['contributing_factors'][:3], 1):
        output += f"  {i}. {factor['factor']}: {factor['value']:.2f} {factor['unit']} ({factor['status']})\n"
    
    cond = result['current_conditions']
    def _fmt_value(val: Any, unit: str) -> str:
        if val is None:
            return f"n/a {unit}"
        return f"{float(val):.1f} {unit}"

    chl = _fmt_value(cond.get('chlorophyll_a'), "µg/L")
    temp = _fmt_value(cond.get('water_temperature'), "°C")
    turb = _fmt_value(cond.get('turbidity'), "NTU")
    output += f"""
**Current Conditions:**
  • Chlorophyll-a: {chl}
  • Water Temp: {temp}
  • Turbidity: {turb}

🎯 Model: {result['model_used']} ({result['base_accuracy']:.1%} accuracy)
⏰ Generated: {result['predicted_at'][:19]}
"""
    
    return output


@tool
def forecast_all_lakes(forecast_days: int = 7) -> str:
    """
    Generate HAB forecast for all monitored German lakes.
    
    Args:
        forecast_days: Forecast horizon (3, 7, or 14 days)
    
    Returns:
        System-wide forecast summary
    """
    if forecast_days not in [3, 7, 14]:
        return "❌ Forecast days must be 3, 7, or 14"
    
    all_predictions = {}
    
    for lake_name, lake_info in engine.GERMAN_LAKES.items():
        location = {"name": lake_name, "latitude": lake_info["lat"], "longitude": lake_info["lon"]}
        prediction = engine.predict_bloom_probability(location, forecast_days)
        all_predictions[lake_name] = prediction
    
    # Calculate summary
    probs = [p['bloom_probability'] for p in all_predictions.values()]
    risk_dist = {"low": 0, "moderate": 0, "high": 0, "critical": 0}
    for pred in all_predictions.values():
        risk_dist[pred['risk_level']] += 1
    
    high_risk = [name for name, p in all_predictions.items() if p['risk_level'] in ['high', 'critical']]
    
    output = f"""🗺️  **German Lakes HAB Forecast ({forecast_days}-day)**

**Summary:**
  • Total Lakes: {len(all_predictions)}
  • Avg Risk: {sum(probs)/len(probs):.1%}
  • Range: {min(probs):.1%} - {max(probs):.1%}

**Risk Distribution:**
"""
    
    for level in ["low", "moderate", "high", "critical"]:
        count = risk_dist[level]
        emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}[level]
        output += f"  {emoji} {level.capitalize()}: {count} lakes\n"
    
    if high_risk:
        output += f"\n⚠️  **High-Risk Alerts:** {', '.join(high_risk)}\n"
    
    output += "\n**Lake-by-Lake:**\n"
    for name, pred in sorted(all_predictions.items(), key=lambda x: x[1]['bloom_probability'], reverse=True):
        emoji = {"low": "🟢", "moderate": "🟡", "high": "🟠", "critical": "🔴"}[pred['risk_level']]
        output += f"  {emoji} {name}: {pred['bloom_probability']:.1%}\n"
    
    return output


@tool
def compare_horizons(lake_name: str) -> str:
    """
    Compare 3, 7, and 14-day forecast horizons for a lake.
    
    Args:
        lake_name: Name of the lake to analyze
    
    Returns:
        Comparative analysis across time horizons
    """
    if lake_name not in engine.GERMAN_LAKES:
        return f"❌ Lake '{lake_name}' not found"
    
    lake_info = engine.GERMAN_LAKES[lake_name]
    location = {"name": lake_name, "latitude": lake_info["lat"], "longitude": lake_info["lon"]}
    
    predictions = {}
    for horizon in [3, 7, 14]:
        predictions[horizon] = engine.predict_bloom_probability(location, horizon)
    
    output = f"📊 **Horizon Comparison - {lake_name}**\n\n"
    
    for h in [3, 7, 14]:
        p = predictions[h]
        output += f"""**{h}-Day:**
  Risk: {p['risk_level'].upper()} ({p['bloom_probability']:.1%})
  Confidence: {p['confidence']:.1%}
  Uncertainty: ±{p['uncertainty']:.1%}

"""
    
    # Trend analysis
    prob_3 = predictions[3]['bloom_probability']
    prob_14 = predictions[14]['bloom_probability']
    
    if prob_14 > prob_3 + 0.15:
        output += "📈 **Trend:** Rising risk - prepare mitigation\n"
    elif prob_3 > prob_14 + 0.15:
        output += "📉 **Trend:** Declining risk - continue monitoring\n"
    else:
        output += "➡️  **Trend:** Stable conditions\n"
    
    return output


@tool
def analyze_risk_factors(lake_name: str, forecast_days: int = 7) -> str:
    """
    Deep analysis of environmental risk factors for a lake.
    
    Args:
        lake_name: Lake to analyze
        forecast_days: Forecast horizon (3, 7, or 14)
    
    Returns:
        Detailed environmental factor breakdown
    """
    if lake_name not in engine.GERMAN_LAKES:
        return f"❌ Lake '{lake_name}' not found"
    
    lake_info = engine.GERMAN_LAKES[lake_name]
    location = {"name": lake_name, "latitude": lake_info["lat"], "longitude": lake_info["lon"]}
    
    result = engine.predict_bloom_probability(location, forecast_days)
    
    output = f"""🔬 **Risk Factor Analysis - {lake_name}**

**Overall:** {result['risk_level'].upper()} ({result['bloom_probability']:.1%})

**Primary Factors:**
"""
    
    for i, factor in enumerate(result['contributing_factors'], 1):
        output += f"""
{i}. **{factor['factor']}**
   Value: {factor['value']:.2f} {factor['unit']}
   Status: {factor['status'].upper()}
   Weight: {factor['importance']:.1%}
"""
    
    cond = result['current_conditions']
    output += f"""
**Complete Environmental Profile:**
  • Chlorophyll-a: {cond['chlorophyll_a']:.1f} µg/L
  • Water Temp: {cond['water_temperature']:.1f}°C
  • Turbidity: {cond['turbidity']:.1f} NTU
  • pH: {cond['ph']:.1f}
  • Dissolved O₂: {cond['dissolved_oxygen']:.1f} mg/L
  • Total Nitrogen: {cond['total_nitrogen']:.2f} mg/L
  • Total Phosphorus: {cond['total_phosphorus']:.3f} mg/L
  • Wind Speed: {cond['wind_speed']:.1f} m/s
"""
    
    return output


@tool
def get_system_status() -> str:
    """
    Get PREDIKT agent system status and model information.
    
    Returns:
        System health, model type, and performance metrics
    """
    info = engine.get_model_info()
    
    output = f"""⚙️  **PREDIKT System Status**

**Version:** {info['version']}
**Status:** OPERATIONAL
**Model Type:** {info['model_type']}
**Using Real ML:** {'✅ YES' if info['using_ml_model'] else '❌ NO'}
**Base Accuracy:** {info['accuracy']:.1%}

**Monitored Lakes:** {len(engine.GERMAN_LAKES)}
  • {', '.join(engine.GERMAN_LAKES.keys())}

**Total Predictions Made:** {len(engine.prediction_history)}
"""
    
    if info['using_ml_model']:
        model_type = info['model_type']
        if model_type == "lstm":
            output += """
✅ **LSTM MODEL ACTIVE**
  • Architecture: Compact LSTM with attention pooling
  • Optimized for multivariate time series
"""
        elif model_type == "sarima":
            output += """
✅ **SARIMA MODEL ACTIVE**
  • Seasonal ARIMA with exogenous variables
  • Optimized for trend/seasonality baselines
"""
        elif model_type == "ensemble":
            output += """
✅ **ENSEMBLE MODEL ACTIVE**
  • Random Forest + Gradient Boosting
  • Optimized for fast tabular inference
"""
        else:
            output += f"""
✅ **ML MODEL ACTIVE**
  • Model type: {model_type}
"""
    else:
        output += """
📊 **RULE-BASED MODE**
  • Using environmental parameter analysis
  • Train a model to enable ML inference
"""
    
    return output


@tool
def request_data_from_homogen(lake_name: str, date_range: str = "last_30_days") -> str:
    """
    Request harmonized data from HOMOGEN agent (for multi-agent coordination).
    
    Args:
        lake_name: Lake to get data for
        date_range: Time period needed
    
    Returns:
        Data request status (triggers handoff to HOMOGEN)
    """
    return f"""📡 **Data Request to HOMOGEN**

Requesting harmonized data for: {lake_name}
Period: {date_range}
Purpose: HAB prediction model input

Status: HANDOFF_REQUIRED → HOMOGEN Agent
"""


@tool
def request_satellite_calibration(lake_name: str) -> str:
    """
    Request calibrated satellite data from CALIBRO agent.
    
    Args:
        lake_name: Lake to get satellite data for
    
    Returns:
        Calibration request status (triggers handoff to CALIBRO)
    """
    return f"""🛰️  **Satellite Data Request to CALIBRO**

Requesting calibrated EO data for: {lake_name}
Parameters needed: Chlorophyll-a, turbidity, temperature
Purpose: Ground-truth validation for HAB prediction

Status: HANDOFF_REQUIRED → CALIBRO Agent
"""


# Tools list
tools = [
    predict_single_lake,
    forecast_all_lakes,
    compare_horizons,
    analyze_risk_factors,
    get_system_status,
    request_data_from_homogen,
    request_satellite_calibration
]


# ============================================================
# LANGGRAPH NODE FUNCTIONS
# ============================================================

def call_model(state: PrediktState) -> PrediktState:
    """Main agent reasoning node"""
    
    messages = state["messages"]
    
    # Add system message if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        system_prompt = f"""You are PREDIKT, an AI agent specializing in harmful algal bloom (HAB) prediction for German lakes.

**Current Configuration:**
- Model: {engine.get_model_info()['model_type']}
- Using ML: {engine.get_model_info()['using_ml_model']}
- Accuracy: {engine.get_model_info()['accuracy']:.1%}

**Your Role in Multi-Agent System:**
- Primary: Predict HAB risks (3/7/14 day forecasts)
- Coordination: Request data from HOMOGEN, CALIBRO when needed
- Integration: Provide predictions to main orchestrator agent

**Autonomous Behavior:**
- "predict X" → immediately call predict_single_lake(X, 7)
- "forecast all" → call forecast_all_lakes()
- "compare X" → call compare_horizons(X)
- Be proactive, don't ask unnecessary clarifications

**Available Lakes:** {', '.join(engine.GERMAN_LAKES.keys())}
"""
        messages = [SystemMessage(content=system_prompt)] + messages
    
    # Initialize LLM with tools
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-2.5-flash",
        temperature=0.1,
        google_api_key=os.getenv("GEMINI_API_KEY")
    ).bind_tools(tools)
    
    # Get response
    response = llm.invoke(messages)
    
    # Update state
    model_info = engine.get_model_info()
    return {
        "messages": messages + [response],
        "agent_name": state.get("agent_name", "PREDIKT"),
        "using_ml_model": model_info.get("using_ml_model", False),
        "confidence_level": state.get("confidence_level", 0.85)
    }


def should_continue(state: PrediktState) -> Literal["tools", "end"]:
    """Decide whether to use tools or end"""
    
    last_message = state["messages"][-1]
    
    # Check if model wants to use tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


def process_tool_output(state: PrediktState) -> PrediktState:
    """Process tool results and update state"""
    
    # Check if handoff is needed (to HOMOGEN or CALIBRO)
    last_messages = state["messages"][-3:]
    
    requires_handoff = False
    handoff_target = None
    
    for msg in last_messages:
        if hasattr(msg, "content"):
            content_str = str(msg.content)
            if "HANDOFF_REQUIRED → HOMOGEN" in content_str:
                requires_handoff = True
                handoff_target = "HOMOGEN"
            elif "HANDOFF_REQUIRED → CALIBRO" in content_str:
                requires_handoff = True
                handoff_target = "CALIBRO"
    
    return {
        "messages": state["messages"],
        "requires_handoff": requires_handoff,
        "handoff_target": handoff_target,
        "agent_name": "PREDIKT"
    }


# ============================================================
# BUILD LANGGRAPH
# ============================================================

def create_predikt_graph():
    """Create the PREDIKT agent graph"""
    
    # Initialize graph
    workflow = StateGraph(PrediktState)
    
    # Add nodes
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("process", process_tool_output)
    
    # Set entry point
    workflow.set_entry_point("agent")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END
        }
    )
    
    # Tool output goes to processing
    workflow.add_edge("tools", "process")
    
    # After processing, go back to agent (for multi-turn)
    workflow.add_edge("process", "agent")
    
    # Compile with memory
    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    
    return app


# ============================================================
# CLI INTERFACE
# ============================================================

class PrediktCLI:
    """Command-line interface for PREDIKT agent"""
    
    def __init__(self):
        self.graph = create_predikt_graph()
        self.config = {"configurable": {"thread_id": "predikt_session_1"}}
        self.llm_enabled = bool(os.getenv("GEMINI_API_KEY")) and os.getenv("PREDIKT_LLM_DISABLED") != "1"
        
        # Initialize state
        self.state = {
            "messages": [],
            "current_prediction": None,
            "active_lakes": [],
            "prediction_horizon": 7,
            "agent_name": "PREDIKT",
            "task_type": "prediction",
            "requires_handoff": False,
            "handoff_target": None,
            "data_quality_score": 0.85,
            "using_ml_model": engine.get_model_info().get("using_ml_model", False),
            "confidence_level": 0.85
        }
    
    def send_message(self, user_input: str) -> str:
        """Send message and get response"""
        if not self.llm_enabled:
            return self._fallback_response(user_input)

        # Add user message to state
        self.state["messages"].append(HumanMessage(content=user_input))

        try:
            # Invoke graph
            result = self.graph.invoke(self.state, self.config)

            # Update state
            self.state = result

            # Get last AI message
            ai_messages = [m for m in result["messages"] if isinstance(m, AIMessage)]
            if ai_messages:
                return ai_messages[-1].content
        except Exception as exc:
            return self._fallback_response(user_input, error=str(exc))

        return "No response generated"

    def _fallback_response(self, user_input: str, error: Optional[str] = None) -> str:
        """Simple intent parser when LLM is unavailable."""
        text = user_input.strip()
        lowered = text.lower()

        if error:
            header = "⚠️ LLM unavailable, using local fallback.\n"
        else:
            header = ""

        if lowered in {"help", "?"}:
            return header + (
                "Try: 'predict Bodensee', 'forecast all', 'compare Bodensee', or 'status'."
            )

        if lowered.startswith("predict "):
            lake = text[len("predict "):].strip()
            return header + predict_single_lake.invoke(
                {"lake_name": lake, "forecast_days": 7}
            )

        if "forecast all" in lowered:
            return header + forecast_all_lakes.invoke({"forecast_days": 7})

        if lowered.startswith("compare "):
            lake = text[len("compare "):].strip()
            return header + compare_horizons.invoke({"lake_name": lake})

        if "status" in lowered:
            return header + get_system_status.invoke({})

        return header + "Try: 'predict Bodensee', 'forecast all', 'compare Bodensee', or 'status'."
    
    def reset(self):
        """Reset conversation"""
        self.state["messages"] = []
        self.config = {"configurable": {"thread_id": f"predikt_session_{datetime.now().timestamp()}"}}


def main():
    """Main CLI loop"""
    
    print("=" * 70)
    print("🔮  PREDIKT - HAB Prediction Agent (LangGraph)")
    print("=" * 70)
    
    # Show model status
    model_info = engine.get_model_info()
    print(f"\n🤖 Model: {model_info['model_type']}")
    print(f"🎯 Accuracy: {model_info['accuracy']:.1%}")
    if model_info.get('using_ml_model'):
        print(f"✅ Using {model_info['model_type'].upper()} model")
    else:
        print("📊 Using rule-based fallback")
    
    print("\n" + "=" * 70)
    print("💬 PREDIKT ready! Ask about HAB forecasts.\n")
    
    cli = PrediktCLI()
    
    while True:
        try:
            user_input = input("🧠 You > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['exit', 'quit', 'bye']:
                print("\n👋 Goodbye!")
                break
            
            if user_input.lower() == 'reset':
                cli.reset()
                print("\n🔄 Conversation reset!\n")
                continue
            
            if user_input.lower() == 'help':
                print("""
📖 **Commands:**
  • predict [lake] - 7-day forecast
  • predict [lake] for [3/7/14] days
  • forecast all - All lakes overview
  • compare [lake] - Compare horizons
  • risk factors for [lake]
  • status - System health
  • reset - Clear conversation
  • exit - Quit
""")
                continue
            
            print()
            response = cli.send_message(user_input)
            print(f"🔮 PREDIKT > {response}\n")
        
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()
