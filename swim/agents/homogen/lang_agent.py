# swim/agents/homogen/lang_agent.py

import os
import sys
from dotenv import load_dotenv
from pathlib import Path
from langchain.tools import Tool
from langchain.agents import initialize_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# Import all tools
from swim.agents.homogen.core_pipeline import HOMOGENPipeline
from swim.agents.homogen.tools import (
    load_csv,
    show_columns,
    validate_sample,
    compute_bbox,
    summarize_harmonized_data,
    get_lake_quality,
    get_habs_summary,
    list_available_lakes
)

# ------------------------
# Load environment
# ------------------------
load_dotenv()

# ------------------------
# Initialize LLM
# ------------------------
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    temperature=0.4,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

# ------------------------
# Initialize pipeline
# ------------------------
project_root = Path(__file__).resolve().parents[3]
pipeline = HOMOGENPipeline(project_root)

# ------------------------
# Tool: Run HOMOGEN pipeline
# ------------------------
def run_homogen_pipeline(input: str = "") -> str:
    """Runs the HOMOGEN harmonization pipeline for all data sources."""
    pipeline.run_pipeline()
    return "Pipeline run complete. Check 'data/harmonized/' for output."

# ------------------------
# Register all tools
# ------------------------
tools = [
    Tool.from_function(
        run_homogen_pipeline,
        name="run_homogen_pipeline",
        description="Runs the HOMOGEN harmonization pipeline for all data sources."
    ),
    Tool.from_function(
        summarize_harmonized_data,
        name="summarize_harmonized_data",
        description="Summarizes record count and quality of harmonized datasets."
    ),
    Tool.from_function(
        load_csv,
        name="load_csv",
        description="Loads a CSV file and returns the number of rows."
    ),
    Tool.from_function(
        show_columns,
        name="show_columns",
        description="Displays column names in the CSV file."
    ),
    Tool.from_function(
        validate_sample,
        name="validate_sample",
        description="Checks missing values in the given CSV file."
    ),
    Tool.from_function(
        compute_bbox,
        name="compute_bbox",
        description="Computes the bounding box from latitude and longitude columns."
    ),
    Tool.from_function(
        get_lake_quality,
        name="get_lake_quality",
        description="Returns water quality statistics (e.g., turbidity, pH) for a given lake or station name."
    ),
    Tool.from_function(
        get_habs_summary,
        name="get_habs_summary",
        description="Returns HABs (algae bloom) indicators such as bloom probability, toxin levels, and bloom status for a given lake."
    ),
    Tool.from_function(
        list_available_lakes,
        name="list_available_lakes",
        description="Lists all available lake names in the dataset."
    )
]

# ------------------------
# Agent Setup
# ------------------------
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description",
    verbose=True
)

# ------------------------
# CLI Chat
# ------------------------
def launch_chat():
    print("\033[1m\n🤖  HOMOGEN Agent Activated!\033[0m")
    print("🧠  Your intelligent assistant for harmonizing, validating, and analyzing environmental data.")
    print("💬  Type a command or question, or type \033[1m'exit'\033[0m to quit.\n")

    while True:
        try:
            user_input = input("\033[94m🗨️  You > \033[0m")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("\n👋 \033[1mHOMOGEN Agent shutting down. Stay green! 🌿\033[0m\n")
                sys.exit(0)

            response = agent.run(user_input)
            print(f"\n\033[92m📊 HOMOGEN Agent Response:\033[0m\n{response}\n")

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