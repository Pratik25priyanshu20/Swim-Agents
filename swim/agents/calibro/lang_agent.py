# swim/agents/calibro/lang_agent.py

import os
import sys
from dotenv import load_dotenv
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

from swim.agents.calibro.core_pipeline import run_calibro_pipeline
from swim.agents.calibro.tools import (
    list_lakes,
    summarize_all_lakes,
    list_uploaded_csvs,
    summarize_uploaded_csv,
    filter_by_lake,
    filter_by_date_range,
    plot_index_timeseries,
    summarize_quality_metrics
)
from swim.agents.calibro.tool_schemas import (
    FilterByLakeInput,
    FilterByDateRangeInput,
    PlotIndexInput,
    SummarizeQualityInput
)

# ------------------------
# Load environment
# ------------------------
load_dotenv()
gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("❌ GEMINI_API_KEY not set in .env")

# ------------------------
# Initialize LLM
# ------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.4,
    google_api_key=gemini_api_key
)

# ------------------------
# Register Tools
# ------------------------
tools = [
    Tool.from_function(
        func=lambda input: run_calibro_pipeline(select_lake=input),
        name="run_calibro_pipeline",
        description="Runs the CALIBRO satellite chlorophyll and turbidity analysis pipeline for selected lakes. Input should be a lake name like 'Chiemsee' or 'ALL'."
    ),
    Tool.from_function(
        func=list_lakes,
        name="list_lakes",
        description="Lists all lakes available for satellite analysis."
    ),
    Tool.from_function(
        func=summarize_all_lakes,
        name="summarize_all_lakes",
        description="Summarizes data coverage and stats for all processed lakes."
    ),
    Tool.from_function(
        func=list_uploaded_csvs,
        name="list_uploaded_csvs",
        description="Lists all uploaded CSV files in the data/ folder."
    ),
    Tool.from_function(
        func=summarize_uploaded_csv,
        name="summarize_uploaded_csv",
        description="Summarizes a selected CSV file from the data/ folder. Input should be the exact file name."
    ),
    Tool.from_function(
        func=filter_by_lake,
        name="filter_by_lake",
        description="Filters a CSV file by lake name.",
        args_schema=FilterByLakeInput
    ),
    Tool.from_function(
        func=filter_by_date_range,
        name="filter_by_date_range",
        description="Filters a CSV file by a date range.",
        args_schema=FilterByDateRangeInput
    ),
    Tool.from_function(
        func=plot_index_timeseries,
        name="plot_index_timeseries",
        description="Plots a time series for a given index (e.g., 'turbidity_index', 'chlorophyll_index') from a CSV file.",
        args_schema=PlotIndexInput
    ),
    Tool.from_function(
        func=summarize_quality_metrics,
        name="summarize_quality_metrics",
        description="Summarizes quality metrics like RMSE and R² from a calibrated CSV.",
        args_schema=SummarizeQualityInput
    )
]

# ------------------------
# Agent Setup
# ------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="openai-functions",
    verbose=True
)

# ------------------------
# CLI Interface
# ------------------------
def launch_chat():
    print("\033[1m\n🌊 CALIBRO Agent Activated!\033[0m")
    print("🛰️  Ask me to run satellite analysis or summarize lake insights.")
    print("📁  You can also analyze uploaded calibration or ESA/DLR CSVs.")
    print("💬  Type a command or question, or type \033[1m'exit'\033[0m to quit.\n")

    while True:
        try:
            user_input = input("\033[94m🗨️  You > \033[0m")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("\n👋 \033[1mCALIBRO Agent shutting down. Stay blue! 💧\033[0m\n")
                sys.exit(0)

            response = agent.run(user_input)
            print(f"\n\033[92m📊 CALIBRO Agent Response:\033[0m\n{response}\n")

        except KeyboardInterrupt:
            print("\n\n👋 \033[1mSession interrupted. Goodbye!\033[0m\n")
            sys.exit(0)
        except Exception as e:
            print(f"\n⚠️  \033[91mError:\033[0m {e}\n")

# ------------------------
# Main Entry Point
# ------------------------
if __name__ == "__main__":
    launch_chat()