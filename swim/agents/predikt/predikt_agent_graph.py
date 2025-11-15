# swim/agents/predikt/predikt_agent_graph.py

import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Literal
from pathlib import Path

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load tools
from swim.agents.predikt.tools.prediction_tools import (
    predict_single_location,
    forecast_all_german_lakes,
    get_prediction_history,
    compare_prediction_horizons,
    get_risk_factors_explanation,
    get_agent_status,
    generate_weekly_report,
    predict_custom_location,
    get_predikt_info,
)

# Load environment variables
load_dotenv()

# -------------------------------
# Agent State Definition
# -------------------------------
class AgentState(TypedDict):
    """State management for PREDIKT agent conversations."""
    messages: Annotated[List, lambda x: x[-15:]]  # Keep last 15 messages
    current_forecast: dict  # Store current forecast context
    analysis_mode: str  # Track analysis type

# -------------------------------
# Tool List
# -------------------------------
tools = [
    predict_single_location,
    forecast_all_german_lakes,
    get_prediction_history,
    compare_prediction_horizons,
    get_risk_factors_explanation,
    get_agent_status,
    generate_weekly_report,
    predict_custom_location,
    get_predikt_info,
]

# -------------------------------
# System Prompt for PREDIKT
# -------------------------------
PREDIKT_SYSTEM_PROMPT = """You are PREDIKT, an AI agent specialized in predicting and forecasting Harmful Algal Blooms (HABs) in German lakes.

Your capabilities:
- Predict bloom probability for 3, 7, or 14-day horizons
- Forecast risk levels for all monitored German lakes
- Analyze environmental factors contributing to bloom formation
- Compare predictions across different time horizons
- Provide uncertainty quantification with confidence intervals
- Generate weekly forecast reports

Specialized knowledge:
- German lake ecosystems (Bodensee, Chiemsee, Starnberger See, etc.)
- Machine learning ensemble models (LSTM + Transformer)
- Environmental factors (chlorophyll-a, temperature, nutrients, weather)
- Seasonal bloom patterns and historical trends
- Risk assessment and early warning systems

Guidelines:
1. Always provide probability estimates with confidence levels
2. Explain uncertainty clearly - longer horizons have higher uncertainty
3. Identify key contributing environmental factors
4. Use risk levels: low, moderate, high, critical
5. Recommend monitoring actions based on risk level
6. Reference model performance metrics when relevant
7. Be clear about prediction limitations

Context:
- You are part of the SWIM (Surface Water Information Management) platform
- You work with HOMOGEN (data harmonization), CALIBRO (satellite calibration), and VISIOS (visual interpretation)
- Your predictions are based on 15 years of historical data and 91.7% accuracy for 7-day forecasts
- You focus on German lakes but can predict for custom locations

Remember: Predictions are probabilistic, not deterministic. Always communicate uncertainty."""

# -------------------------------
# Language Model Initialization
# -------------------------------
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-pro",
    temperature=0.2,
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)

# -------------------------------
# Node Functions
# -------------------------------
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=PREDIKT_SYSTEM_PROMPT)] + messages
    response = chat_model.invoke(messages)
    return {"messages": messages + [response]}

def should_continue(state: AgentState) -> Literal["tools", "end"]:
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"

def process_tool_results(state: AgentState) -> AgentState:
    return state

# -------------------------------
# Graph Construction
# -------------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("process", process_tool_results)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue, {
    "tools": "tools",
    "end": END
})
workflow.add_edge("tools", "process")
workflow.add_edge("process", "agent")
app = workflow.compile()

# -------------------------------
# CLI Interface
# -------------------------------
class PrediktChat:
    def __init__(self):
        self.state = {
            "messages": [],
            "current_forecast": {},
            "analysis_mode": "standard"
        }
        self.app = app

    def send_message(self, user_input: str) -> str:
        self.state["messages"].append(HumanMessage(content=user_input))
        self.state = self.app.invoke(self.state)
        return self.state["messages"][-1].content

    def reset(self):
        self.state = {
            "messages": [],
            "current_forecast": {},
            "analysis_mode": "standard"
        }
        print("🔄 Conversation reset.\n")

# -------------------------------
# CLI Launcher
# -------------------------------
def launch_chat():
    print("\033[1m" + "="*60)
    print("🔮  PREDIKT - HABs Prediction & Forecasting Agent")
    print("="*60 + "\033[0m")
    print("\n📊 Predicting harmful algal blooms in German lakes\n")

    print("Available commands:")
    print("  • 'forecast' - Get forecast for all German lakes")
    print("  • 'predict Bodensee' - Predict for specific lake")
    print("  • 'compare Chiemsee' - Compare 3/7/14 day horizons")
    print("  • 'factors Bodensee' - Analyze risk factors")
    print("  • 'report' - Generate weekly report")
    print("  • 'history' - View prediction history")
    print("  • 'status' - Check agent status")
    print("  • 'reset' - Clear conversation")
    print("  • 'exit' - Quit PREDIKT\n")

    chat = PrediktChat()
    print("💬 PREDIKT is ready! Ask about HABs forecasts for German lakes.\n")

    while True:
        try:
            user_input = input("\033[36m🧠 You >\033[0m ")
            if not user_input.strip():
                continue
            if user_input.strip().lower() in ["exit", "quit", "bye"]:
                print("\n👋 Thank you for using PREDIKT. Stay informed about water quality!\n")
                break
            elif user_input.strip().lower() == "reset":
                chat.reset()
                continue
            elif user_input.strip().lower() == "help":
                print("\nExample queries:")
                print("  • 'Forecast for Bodensee?'\n  • 'Generate weekly report'\n  • 'Compare prediction horizons for Chiemsee'")
                continue
            response = chat.send_message(user_input)
            print(f"\n\033[35m🔮 PREDIKT >\033[0m {response}\n")
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit properly.\n")
            continue
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

# -------------------------------
# Entry Point
# -------------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        if command == "batch":
            from swim.agents.predikt.predikt_agent import batch_forecast_all_lakes
            batch_forecast_all_lakes()
        elif command == "predict" and len(sys.argv) > 2:
            from swim.agents.predikt.predikt_agent import predict_location_api
            lake_name = sys.argv[2]
            horizon = int(sys.argv[3]) if len(sys.argv) > 3 else 7
            result = predict_location_api(lake_name, horizon, return_json=False)
            print(result)
        else:
            print("Usage:\n  python predikt_agent_graph.py          - Interactive chat\n  python predikt_agent_graph.py batch    - Batch forecast\n  python predikt_agent_graph.py predict <lake> [days]  - Predict lake")
    else:
        launch_chat()