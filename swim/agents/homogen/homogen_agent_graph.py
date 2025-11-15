# swim/agents/homogen/homogen_agent_graph.py

import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Optional

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from swim.agents.homogen.tools import (
    run_homogen_pipeline,
    summarize_harmonized_data,
    load_csv,
    show_columns,
    validate_sample,
    compute_bbox,
    get_lake_quality,
    get_habs_summary,
    list_available_lakes,
    detect_outliers,
)

# Load environment
load_dotenv()

# -----------------------------
# Agent State Definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]  # Must be BaseMessage list

# -----------------------------
# Tool List
# -----------------------------
tools = [
    run_homogen_pipeline,
    summarize_harmonized_data,
    load_csv,
    show_columns,
    validate_sample,
    compute_bbox,
    get_lake_quality,
    get_habs_summary,
    list_available_lakes,
    detect_outliers,
]

# -----------------------------
# Model Setup
# -----------------------------
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-pro",
    temperature=0.4,
    google_api_key=os.getenv("GEMINI_API_KEY")
).bind_tools(tools)

# -----------------------------
# LangGraph Node Functions
# -----------------------------
def call_model(state: AgentState) -> AgentState:
    messages = state["messages"]

    if not all(isinstance(m, BaseMessage) for m in messages):
        raise TypeError("All elements in 'messages' must be BaseMessage instances")

    response = chat_model.invoke(messages)
    return {"messages": messages + [response]}

def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END

def process_tool_results(state: AgentState) -> AgentState:
    return state

# -----------------------------
# LangGraph Workflow
# -----------------------------
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

# -----------------------------
# CLI Launcher
# -----------------------------
def launch_chat():
    print("\033[1m\n🤖 HOMOGEN LangGraph Agent Activated!\033[0m")
    print("💬 Ask anything about harmonized lake data or run the pipeline.\n(Type 'exit' to quit)\n")

    state: AgentState = {"messages": []}
    while True:
        try:
            user_input = input("🧠 You > ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("👋 HOMOGEN shutting down.\n")
                break

            # Add HumanMessage
            human_msg = HumanMessage(content=user_input)
            state["messages"].append(human_msg)

            # Invoke LangGraph
            state = app.invoke(state)

            # Display response
            response_msg = state["messages"][-1]
            if isinstance(response_msg, AIMessage):
                print(f"\n🤖 HOMOGEN > {response_msg.content}\n")

        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    launch_chat()