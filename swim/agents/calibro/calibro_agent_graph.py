# swim/agents/calibro/calibro_agent_graph.py


import os
import time
from dotenv import load_dotenv
from typing import TypedDict, Optional, Annotated, List
import pandas as pd

from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_google_genai import ChatGoogleGenerativeAI

from swim.agents.calibro.core_pipeline import run_calibro_pipeline

from swim.agents.calibro.calibro_core import CalibroCore
from swim.agents.calibro.tools import (
    list_uploaded_csvs,
    summarize_uploaded_csv,
    filter_by_lake,
    filter_by_date_range,
    plot_index_timeseries,
    summarize_quality_metrics,
)


from swim.agents.calibro.enhanced_tools.trend_analysis import AdvancedTrendAnalyzer
from swim.agents.calibro.enhanced_tools.lake_finder import EnhancedWaterBodyProcessor
from swim.agents.calibro.enhanced_tools.bloom_risk import AdvancedRiskAssessment

# Load Gemini API Key
load_dotenv()
chat_model = ChatGoogleGenerativeAI(
    model="models/gemini-2.5-flash",
    google_api_key=os.getenv("GEMINI_API_KEY"),
    temperature=0.4,
)

# Initialize enhanced processors
trend_analyzer = AdvancedTrendAnalyzer()
lake_processor = EnhancedWaterBodyProcessor()
risk_assessor = AdvancedRiskAssessment()

# -----------------------------
# Agent State Definition
# -----------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    selected_csv: Optional[str]
    lake_name: Optional[str]
    processing_errors: Optional[List[str]]

# -----------------------------
# Enhanced Tool Definitions
# -----------------------------
@tool
def scan_all_files(query: str = "") -> str:
    """Scan and list all CSV files in the data directory."""
    try:
        return list_uploaded_csvs()
    except Exception as e:
        return f"⚠️ Error scanning files: {str(e)}"

@tool
def summarize_all_csvs(query: str = "") -> str:
    """Summarize all CSV files found in the data directory."""
    try:
        file_list = list_uploaded_csvs().splitlines()
        summaries = []
        for line in file_list:
            fname = line.replace("• ", "").strip()
            if fname:
                summary = summarize_uploaded_csv(fname)
                summaries.append(summary)
        return "\n\n".join(summaries) if summaries else "⚠️ No CSVs found."
    except Exception as e:
        return f"⚠️ Error summarizing CSVs: {str(e)}"

@tool
def plot_turbidity_index(query: str = "") -> str:
    """Plot turbidity index for all lakes from the calibrated data."""
    try:
        return plot_index_timeseries(
            "CALIBRO_Agent_2020_2025_Alle_Seen_Kalibrierte_Satelliten_Daten_Output.csv", 
            "turbidity_index"
        )
    except Exception as e:
        return f"⚠️ Error plotting turbidity: {str(e)}"

@tool
def summarize_calibration_metrics(query: str = "") -> str:
    """Summarize calibration quality metrics from the calibration output file."""
    try:
        return summarize_quality_metrics(
            "CALIBRO_Agent_2023_Seen_Satelliten_Kalibrierung_Output.csv"
        )
    except Exception as e:
        return f"⚠️ Error summarizing metrics: {str(e)}"

@tool
def filter_by_lake_tool(file_name: str, lake_name: str) -> str:
    """Filter a given CSV by lake name."""
    try:
        return filter_by_lake(file_name, lake_name)
    except Exception as e:
        return f"⚠️ Error filtering by lake: {str(e)}"

@tool
def filter_by_date_tool(file_name: str, start_date: str, end_date: str) -> str:
    """Filter a CSV file by date range (format: YYYY-MM-DD)."""
    try:
        return filter_by_date_range(file_name, start_date, end_date)
    except Exception as e:
        return f"⚠️ Error filtering by date: {str(e)}"

# -----------------------------
# NEW ENHANCED TOOLS
# -----------------------------
@tool
def analyze_water_quality_trends(csv_file: str, lake_name: str) -> str:
    """
    Analyze temporal trends in water quality metrics (chlorophyll-a and turbidity).
    Performs statistical regression analysis to detect increasing, decreasing, or stable trends.
    
    Args:
        csv_file: Name of the CSV file in outputs directory
        lake_name: Name of the lake to analyze
    """
    try:
        df = pd.read_csv(f"outputs/{csv_file}")
        
        # Ensure required columns exist
        if 'Chl_a_mg_m3' not in df.columns and 'Chl_mg_m3' in df.columns:
            df['Chl_a_mg_m3'] = df['Chl_mg_m3']
        if 'Turbidity_FNU' not in df.columns and 'Turbidity_FNU_median' in df.columns:
            df['Turbidity_FNU'] = df['Turbidity_FNU_median']
        
        result = trend_analyzer.analyze_water_quality_trends(df, lake_name)
        
        # Format output nicely
        output = f"📊 Trend Analysis for {lake_name}\n"
        output += f"Analysis Date: {result.get('trend_analysis_date', 'N/A')}\n\n"
        
        if 'chlorophyll_a' in result:
            chl = result['chlorophyll_a']
            if 'error' not in chl:
                output += f"🌿 Chlorophyll-a Trend: {chl['trend']}\n"
                output += f"   Slope: {chl['slope']:.4f} mg/m³ per day\n"
                output += f"   R²: {chl['r_squared']:.3f}, p-value: {chl['p_value']:.3f}\n\n"
        
        if 'turbidity' in result:
            turb = result['turbidity']
            if 'error' not in turb:
                output += f"💧 Turbidity Trend: {turb['trend']}\n"
                output += f"   Slope: {turb['slope']:.4f} FNU per day\n"
                output += f"   R²: {turb['r_squared']:.3f}, p-value: {turb['p_value']:.3f}\n"
        
        return output
    except FileNotFoundError:
        return f"⚠️ File not found: {csv_file}. Check outputs directory."
    except Exception as e:
        return f"⚠️ Error analyzing trends: {str(e)}"

@tool
def discover_nearby_lakes(location: str, radius_km: float = 50) -> str:
    """
    Discover water bodies near a specified location using satellite imagery and OpenStreetMap.
    
    Args:
        location: Location name or address (e.g., "Heidelberg, Germany")
        radius_km: Search radius in kilometers (default: 50)
    """
    try:
        results = lake_processor.discover_water_bodies_by_location(location, radius_km)
        
        if not results:
            return f"⚠️ No water bodies found within {radius_km}km of {location}"
        
        output = f"🌊 Found {len(results)} water bodies near {location}:\n\n"
        
        for i, body in enumerate(results[:10], 1):  # Limit to top 10
            output += f"{i}. {body['name']}\n"
            output += f"   Source: {body['source']}\n"
            output += f"   Area: {body['area_ha']:.1f} hectares\n"
            output += f"   Centroid: ({body['centroid'][1]:.4f}, {body['centroid'][0]:.4f})\n\n"
        
        return output
    except Exception as e:
        return f"⚠️ Error discovering lakes: {str(e)}"

@tool
def assess_bloom_risk(csv_file: str, lake_name: str) -> str:
    """
    Comprehensive harmful algal bloom (HAB) risk assessment.
    Analyzes chlorophyll-a levels, trends, seasonal patterns, and variability.
    
    Args:
        csv_file: Name of the CSV file in outputs directory
        lake_name: Name of the lake to assess
    """
    try:
        df = pd.read_csv(f"outputs/{csv_file}")
        
        # Ensure required columns exist
        if 'Chl_a_mg_m3' not in df.columns and 'Chl_mg_m3' in df.columns:
            df['Chl_a_mg_m3'] = df['Chl_mg_m3']
        
        assessment = risk_assessor.assess_bloom_risk(df, lake_name)
        
        if 'error' in assessment:
            return f"⚠️ {assessment['error']}"
        
        # Format output
        output = f"🚨 Bloom Risk Assessment for {lake_name}\n"
        output += f"Assessment Date: {assessment['assessment_date']}\n\n"
        
        output += f"⚠️ Overall Risk Level: {assessment['overall_risk_level']}\n"
        output += f"📊 Risk Score: {assessment['risk_score']:.2f}/4.0\n\n"
        
        # Risk factors
        if 'risk_factors' in assessment:
            factors = assessment['risk_factors']
            
            if 'chlorophyll_a' in factors:
                chl = factors['chlorophyll_a']
                output += f"🌿 Current Chlorophyll-a: {chl['current_level']} mg/m³ ({chl['risk_level']} risk)\n"
            
            if 'trend_risk' in factors:
                trend = factors['trend_risk']
                output += f"📈 Trend Risk: {trend['risk_level']} - {trend.get('reason', 'N/A')}\n"
            
            if 'seasonal_risk' in factors:
                seasonal = factors['seasonal_risk']
                output += f"📅 Seasonal Risk: {seasonal['risk_level']} - {seasonal.get('reason', 'N/A')}\n"
        
        # Recommendations
        output += "\n💡 Recommendations:\n"
        for i, rec in enumerate(assessment.get('recommendations', []), 1):
            output += f"   {i}. {rec}\n"
        
        output += f"\n🔍 Recommended Monitoring: {assessment.get('monitoring_frequency', 'Weekly')}\n"
        
        return output
    except FileNotFoundError:
        return f"⚠️ File not found: {csv_file}. Check outputs directory."
    except Exception as e:
        return f"⚠️ Error assessing bloom risk: {str(e)}"

@tool
def batch_process_lakes(csv_file: str) -> str:
    """
    Batch process all lakes in a CSV file to generate comprehensive reports.
    Includes trend analysis and risk assessment for each lake.
    
    Args:
        csv_file: Name of the CSV file containing multiple lakes
    """
    try:
        df = pd.read_csv(f"outputs/{csv_file}")
        
        if 'lake' not in df.columns and 'lake_name' not in df.columns:
            return "⚠️ CSV must contain 'lake' or 'lake_name' column"
        
        lake_col = 'lake' if 'lake' in df.columns else 'lake_name'
        unique_lakes = df[lake_col].unique()
        
        output = f"🔄 Batch Processing {len(unique_lakes)} lakes from {csv_file}\n\n"
        
        for lake in unique_lakes[:5]:  # Limit to 5 lakes to avoid timeout
            lake_df = df[df[lake_col] == lake]
            output += f"{'='*60}\n"
            output += f"Lake: {lake} ({len(lake_df)} observations)\n"
            output += f"{'='*60}\n\n"
            
            # Quick stats
            if 'Chl_a_mg_m3' in lake_df.columns:
                output += f"Avg Chlorophyll-a: {lake_df['Chl_a_mg_m3'].mean():.2f} mg/m³\n"
            if 'Turbidity_FNU' in lake_df.columns:
                output += f"Avg Turbidity: {lake_df['Turbidity_FNU'].mean():.2f} FNU\n"
            
            output += "\n"
        
        output += f"\n✅ Processed {min(5, len(unique_lakes))} lakes. Use individual tools for detailed analysis."
        
        return output
    except Exception as e:
        return f"⚠️ Error in batch processing: {str(e)}"

@tool
def validate_calibration_quality(csv_file: str) -> str:
    """
    Validate the quality of calibrated satellite data.
    Checks for data completeness, reasonable ranges, and quality flags.
    
    Args:
        csv_file: Name of the calibrated CSV file to validate
    """
    try:
        df = pd.read_csv(f"outputs/{csv_file}")
        
        output = f"✅ Validation Report for {csv_file}\n\n"
        
        # Check 1: Completeness
        total_rows = len(df)
        output += f"📊 Total Observations: {total_rows}\n"
        
        # Check 2: Missing values
        if total_rows > 0:
            missing_pct = (df.isnull().sum() / total_rows * 100)
            output += f"\n📉 Missing Data:\n"
            for col in missing_pct[missing_pct > 0].index:
                output += f"   {col}: {missing_pct[col]:.1f}%\n"
        
        # Check 3: Value ranges
        range_checks = []
        if 'Chl_a_mg_m3' in df.columns:
            chl_ok = df['Chl_a_mg_m3'].between(0, 200).sum()
            range_checks.append(f"Chlorophyll-a in range [0-200]: {chl_ok}/{len(df)} ({chl_ok/len(df)*100:.1f}%)")
        
        if 'Turbidity_FNU' in df.columns:
            turb_ok = df['Turbidity_FNU'].between(0, 300).sum()
            range_checks.append(f"Turbidity in range [0-300]: {turb_ok}/{len(df)} ({turb_ok/len(df)*100:.1f}%)")
        
        if range_checks:
            output += f"\n✓ Range Validation:\n"
            for check in range_checks:
                output += f"   {check}\n"
        
        # Check 4: Quality flags (if present)
        qa_cols = [col for col in df.columns if 'QA' in col or 'quality' in col.lower()]
        if qa_cols:
            output += f"\n🏁 Quality Flags Present: {', '.join(qa_cols)}\n"
        
        # Check 5: Temporal coverage
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            date_range = (df['date'].max() - df['date'].min()).days
            output += f"\n📅 Temporal Coverage: {date_range} days\n"
            output += f"   From: {df['date'].min().strftime('%Y-%m-%d')}\n"
            output += f"   To: {df['date'].max().strftime('%Y-%m-%d')}\n"
        
        output += f"\n{'='*50}\n"
        output += "✅ Validation Complete\n"
        
        return output
    except Exception as e:
        return f"⚠️ Error validating data: {str(e)}"

# -----------------------------
# Bind All Tools to LLM
# -----------------------------
tools = [
    # Basic tools
    scan_all_files,
    summarize_all_csvs,
    plot_turbidity_index,
    summarize_calibration_metrics,
    filter_by_lake_tool,
    filter_by_date_tool,
    # Enhanced tools
    analyze_water_quality_trends,
    discover_nearby_lakes,
    assess_bloom_risk,
    batch_process_lakes,
    validate_calibration_quality,
]

model_with_tools = chat_model.bind_tools(tools)

# -----------------------------
# LangGraph Node Logic with Error Recovery
# -----------------------------
def call_model(state: AgentState) -> AgentState:
    """Call model with error handling"""
    try:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": state["messages"] + [response]}
    except Exception as e:
        error_msg = AIMessage(content=f"⚠️ Error calling model: {str(e)}")
        errors = state.get("processing_errors", []) or []
        errors.append(str(e))
        return {
            "messages": state["messages"] + [error_msg],
            "processing_errors": errors
        }

def should_continue(state: AgentState) -> str:
    """Determine next step with error checking"""
    last = state["messages"][-1]
    
    # Check for too many errors
    errors = state.get("processing_errors", [])
    if errors and len(errors) > 3:
        return END
    
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools)

# -----------------------------
# Enhanced LangGraph Workflow
# -----------------------------
workflow = StateGraph(AgentState)
workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)
workflow.set_entry_point("agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", "agent")
app = workflow.compile()

# -----------------------------
# Enhanced Chat Loop
# -----------------------------
def launch_chat():
    print("\n" + "="*70)
    print("🚀 CALIBRO Agent v2.0 - Enhanced Satellite Calibration System")
    print("="*70)
    print("\n📋 Available Capabilities:")
    print("  • Satellite data calibration and validation")
    print("  • Water quality trend analysis")
    print("  • Harmful algal bloom risk assessment")
    print("  • Nearby lake discovery")
    print("  • Batch processing of multiple lakes")
    print("  • Data quality validation")
    print("\n💡 Try: 'analyze trends for Bodensee' or 'find lakes near Heidelberg'")
    print("="*70 + "\n")
    
    state: AgentState = {
        "messages": [HumanMessage(content="""
You are CALIBRO v2.0, an advanced satellite calibration expert for lake water quality monitoring.

Your enhanced capabilities include:
1. Satellite data calibration with confidence metrics
2. Statistical trend analysis (chlorophyll-a, turbidity)
3. Harmful algal bloom risk assessment
4. Water body discovery using satellite imagery
5. Batch processing and quality validation

Help users understand water quality through satellite data analysis.
Be precise, provide confidence levels, and explain technical terms when needed.
""")],
        "selected_csv": None,
        "lake_name": None,
        "processing_errors": []
    }

    conversation_count = 0
    max_conversations = 50  # Prevent infinite loops

    while conversation_count < max_conversations:
        try:
            user_input = input("\n🧠 You > ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit", "bye"]:
                print("\n👋 CALIBRO: Goodbye! Keep monitoring those lakes!")
                break
            
            # Add user message
            state["messages"].append(HumanMessage(content=user_input))
            
            # Invoke agent
            state = app.invoke(state)
            
            # Get last message
            last = state["messages"][-1]
            
            if isinstance(last, AIMessage):
                print(f"\n📊 CALIBRO > {last.content}")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n👋 Interrupted. Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("💡 Try rephrasing your question or type 'exit' to quit.")
            import traceback
            traceback.print_exc()

# -----------------------------
# Entry Point
# -----------------------------
if __name__ == "__main__":
    launch_chat()