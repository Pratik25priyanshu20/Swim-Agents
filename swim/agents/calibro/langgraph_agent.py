#swim/agents/calibro/langgraph_agent.py
'''
import os
from dotenv import load_dotenv
from typing import TypedDict, Optional, Annotated, List

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from swim.agents.calibro.tools import (
    list_uploaded_csvs,
    summarize_uploaded_csv,
    filter_by_lake,
    filter_by_date_range,
    plot_index_timeseries,
    summarize_quality_metrics
)

# -----------------------------
# Load environment & LLM
# -----------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not found in .env")

chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=gemini_api_key,
    temperature=0.4
)

# -----------------------------
# 1. Agent State Definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    selected_csv: Optional[str]
    lake_name: Optional[str]

# -----------------------------
# 2. Tools
# -----------------------------
@tool
def scan_all_files(query: str = "") -> str:
    """Scan and list all CSV files in the data directory."""
    return list_uploaded_csvs()

@tool
def auto_summarize(query: str = "") -> str:
    """Auto-summarize all CSV files found in the data directory."""
    file_list = list_uploaded_csvs()
    lines = file_list.splitlines()
    if not lines:
        return "⚠️ No files found."
    summaries = []
    for line in lines:
        if line.startswith("\u2022 "):
            fname = line.replace("\u2022 ", "").strip()
            summaries.append(summarize_uploaded_csv(fname))
    return "\n\n".join(summaries)

@tool
def auto_plot_turbidity(query: str = "") -> str:
    """Plot turbidity index for all lakes from the calibrated data."""
    fname = "CALIBRO_Agent_2020_2025_Alle_Seen_Kalibrierte_Satelliten_Daten_Output.csv"
    return plot_index_timeseries(fname, "turbidity_index")

@tool
def auto_summarize_calibration(query: str = "") -> str:
    """Summarize calibration quality metrics from the calibration output file."""
    fname = "CALIBRO_Agent_2023_Seen_Satelliten_Kalibrierung_Output.csv"
    return summarize_quality_metrics(fname)

@tool
def filter_data_by_lake(file_name: str, lake_name: str) -> str:
    """Filter CSV data by a specific lake name."""
    return filter_by_lake(file_name, lake_name)

@tool
def filter_data_by_date(file_name: str, start_date: str, end_date: str) -> str:
    """Filter CSV data by date range."""
    return filter_by_date_range(file_name, start_date, end_date)

# -----------------------------
# 3. Bind tools to model
# -----------------------------
tools = [
    scan_all_files,
    auto_summarize,
    auto_plot_turbidity,
    auto_summarize_calibration,
    filter_data_by_lake,
    filter_data_by_date,
]

model_with_tools = chat_model.bind_tools(tools)

# -----------------------------
# 4. Define agent node
# -----------------------------
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]
    response = model_with_tools.invoke(messages)
    return {"messages": messages + [response]}

# -----------------------------
# 5. Define conditional logic
# -----------------------------
def should_continue(state: AgentState) -> str:
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "tools"
    return END

# -----------------------------
# 6. Build LangGraph
# -----------------------------
tool_node = ToolNode(tools)

workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# -----------------------------
# 7. CLI Chat Loop
# -----------------------------
def launch_chat():
    print("\033[1m\n🌊 CALIBRO LangGraph Agent (Gemini) is ready!\033[0m")
    print("Type your question. Type 'exit' to quit.")

    state: AgentState = {
        "messages": [
            HumanMessage(content="""You are CALIBRO, a specialized agent for analyzing satellite data and in-situ measurements for lake water quality monitoring. 
You help users with:
- Analyzing chlorophyll-a and turbidity data
- Processing satellite calibration results
- Filtering and visualizing lake data
- Quality metrics assessment
Be helpful, concise, and technical when needed.""")
        ],
        "selected_csv": None,
        "lake_name": None
    }

    while True:
        try:
            user_input = input("\n\033[94m🗨️  You > \033[0m")
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Exiting CALIBRO Agent. Stay blue 💧")
                break

            state["messages"].append(HumanMessage(content=user_input))
            state = app.invoke(state)

            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage):
                print(f"\n\033[92m📊 Response:\033[0m\n{last_message.content}")

        except KeyboardInterrupt:
            print("\n👋 Interrupted by user. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

# -----------------------------
# 8. Entry Point
# -----------------------------
if __name__ == "__main__":
    launch_chat()
    

'''

# swim/agents/calibro/langgraph_agent.py
"""
Thin launcher for the CALIBRO LangGraph Agent.

The actual LangGraph flow and tools are defined in:
    swim/agents/calibro/calibro_agent_graph.py
"""

from swim.agents.calibro.calibro_agent_graph import launch_chat

if __name__ == "__main__":
    launch_chat()