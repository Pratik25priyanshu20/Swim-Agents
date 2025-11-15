# swim/agents/visios/visios_agent_graph.py
# ENHANCED VERSION WITH FULL INTEGRATION

import os
from dotenv import load_dotenv
from typing import List, TypedDict, Annotated, Literal
from pathlib import Path
import time

from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_google_genai import ChatGoogleGenerativeAI

# Load enhanced tools
from swim.agents.visios.tools.image_tools import (
    list_visios_images,
    analyze_image_by_name,
    summarize_all_images,
    get_image_metadata,
    get_analysis_history,
    generate_full_report,
    check_bloom_risk_at_location,
)

# Load environment variables
load_dotenv()

# ========================================
# Agent State Definition
# ========================================
class AgentState(TypedDict):
    """Enhanced state management for VISIOS conversations."""
    messages: Annotated[List[BaseMessage], add_messages]
    current_task: str
    analysis_context: dict
    images_processed: int
    errors: List[str]

# ========================================
# Enhanced Tool List
# ========================================
tools = [
    list_visios_images,
    analyze_image_by_name,
    summarize_all_images,
    get_image_metadata,
    get_analysis_history,
    generate_full_report,
    check_bloom_risk_at_location,
]

# ========================================
# System Prompt for VISIOS
# ========================================
VISIOS_SYSTEM_PROMPT = """You are VISIOS, an advanced AI agent specialized in visual analysis of surface water for Harmful Algal Blooms (HABs).

🎯 Core Mission:
Empower users to contribute to water quality monitoring through image-based reporting, providing instant AI-powered analysis and actionable insights.

🧠 Technical Capabilities:
- Vision Transformer + EfficientNet ensemble (93.7% accuracy)
- Trained on 150,000+ user photos and 50,000+ expert annotations
- Real-time bloom classification: Clear Water / Possible Bloom / Likely Bloom / Severe Bloom
- GPS metadata extraction and geolocation
- Integration with CALIBRO (satellite) and HOMOGEN (sensor) data

🇩🇪 German Lakes Specialization:
- Optimized for Bodensee, Chiemsee, Müritz, Starnberger See, Ammersee, and 15+ other lakes
- Cultural and regulatory context awareness
- German environmental standards compliance

💬 Communication Guidelines:
1. **Be Clear & Accessible**: Explain technical findings in user-friendly language
2. **Safety First**: Always prioritize public health in recommendations
3. **Educate**: Help users understand HABs, risks, and prevention
4. **Encourage Participation**: Thank users for contributions to water monitoring
5. **Show Confidence Levels**: Be transparent about analysis uncertainty
6. **Link Context**: Reference satellite/sensor data when available

⚠️ Critical Safety Rules:
- For "Severe Bloom" detections, emphasize immediate danger and emergency response
- Always recommend keeping people and pets away from suspected blooms
- Suggest contacting local environmental agencies for confirmed blooms
- Never downplay potential health risks

🔗 Integration Points:
- **CALIBRO**: Validate visual findings with satellite chlorophyll-a data
- **HOMOGEN**: Cross-reference with in-situ sensor measurements
- **User Community**: Build trust through transparent, accurate analysis

Remember: Every image uploaded is a valuable data point. Treat each analysis with care and precision."""

# ========================================
# Language Model Initialization
# ========================================
chat_model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    temperature=0.3,
    google_api_key=os.getenv("GEMINI_API_KEY")
)

model_with_tools = chat_model.bind_tools(tools)

# ========================================
# Node Functions with Error Handling
# ========================================
def call_model(state: AgentState) -> AgentState:
    """
    Main reasoning node - Gemini processes queries and selects tools.
    Enhanced with error recovery and context tracking.
    """
    messages = state["messages"]
    
    # Add system prompt if new conversation
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=VISIOS_SYSTEM_PROMPT)] + messages
    
    try:
        response = model_with_tools.invoke(messages)
        return {
            "messages": messages + [response],
            "current_task": state.get("current_task"),
            "analysis_context": state.get("analysis_context", {}),
            "images_processed": state.get("images_processed", 0)
        }
    except Exception as e:
        error_msg = AIMessage(content=f"⚠️ Error processing request: {str(e)}. Please try rephrasing your question.")
        errors = state.get("errors", []) or []
        errors.append(str(e))
        return {
            "messages": messages + [error_msg],
            "errors": errors
        }


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """
    Routing function with error handling.
    Decides whether to call tools or end conversation.
    """
    # Check for excessive errors
    errors = state.get("errors", [])
    if errors and len(errors) > 5:
        return "end"
    
    last_message = state["messages"][-1]
    
    # Check for tool calls
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def process_tool_results(state: AgentState) -> AgentState:
    """
    Post-process tool execution results.
    Track statistics and update context.
    """
    # Update images processed count
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls'):
        for tool_call in last_message.tool_calls:
            if tool_call['name'] in ['analyze_image_by_name', 'summarize_all_images']:
                images_processed = state.get("images_processed", 0) + 1
                state["images_processed"] = images_processed
    
    return state

# ========================================
# Graph Construction
# ========================================
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("agent", call_model)
workflow.add_node("tools", ToolNode(tools))
workflow.add_node("process", process_tool_results)

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

# Tools go through processing back to agent
workflow.add_edge("tools", "process")
workflow.add_edge("process", "agent")

# Compile the graph
app = workflow.compile()

# ========================================
# Enhanced Chat Interface
# ========================================
class VisiosChat:
    """Interactive chat interface with enhanced features."""
    
    def __init__(self):
        self.state = {
            "messages": [],
            "current_task": None,
            "analysis_context": {},
            "images_processed": 0,
            "errors": []
        }
        self.app = app
        self.session_start = time.time()
    
    def send_message(self, user_input: str) -> str:
        """Send message and get response."""
        self.state["messages"].append(HumanMessage(content=user_input))
        
        try:
            result = self.app.invoke(self.state)
            self.state = result
            
            last_message = result["messages"][-1]
            return last_message.content
        except Exception as e:
            error_response = f"⚠️ System error: {str(e)}. Please try again."
            self.state["messages"].append(AIMessage(content=error_response))
            return error_response
    
    def get_stats(self) -> dict:
        """Get session statistics."""
        session_time = time.time() - self.session_start
        return {
            "images_processed": self.state.get("images_processed", 0),
            "messages_exchanged": len(self.state["messages"]),
            "session_duration_min": round(session_time / 60, 1),
            "errors": len(self.state.get("errors", []))
        }
    
    def reset(self):
        """Reset conversation state."""
        stats = self.get_stats()
        print(f"\n🔄 Session Stats:")
        print(f"   • Images Analyzed: {stats['images_processed']}")
        print(f"   • Messages: {stats['messages_exchanged']}")
        print(f"   • Duration: {stats['session_duration_min']} min")
        print(f"\n✅ Conversation reset.\n")
        
        self.state = {
            "messages": [],
            "current_task": None,
            "analysis_context": {},
            "images_processed": 0,
            "errors": []
        }
        self.session_start = time.time()

# ========================================
# CLI Launcher with Premium UX
# ========================================
def launch_chat():
    """Interactive CLI for VISIOS agent with enhanced user experience."""
    
    # Fancy banner
    print("\n" + "="*75)
    print("🖼️  " + " "*25 + "VISIOS" + " "*25 + "🖼️")
    print("   Visual Interpretation & Scene Analysis Agent for HABs Detection")
    print("="*75)
    
    print("\n🌊 Analyzing surface water images for Harmful Algal Blooms")
    print("🇩🇪 Specialized for German Lakes • 93.7% Accuracy • Real-time Analysis\n")
    
    # Show capabilities
    print("📋 Core Capabilities:")
    print("  ✓ Image-based bloom detection (Vision Transformer + EfficientNet)")
    print("  ✓ GPS metadata extraction and geolocation")
    print("  ✓ Safety recommendations and risk assessment")
    print("  ✓ Integration with satellite (CALIBRO) and sensor (HOMOGEN) data")
    print("  ✓ Batch processing and comprehensive reporting\n")
    
    # Quick commands
    print("⚡ Quick Commands:")
    print("  • 'list' or 'show images' - View available images")
    print("  • 'analyze <filename>' - Analyze specific image")
    print("  • 'summary' or 'report' - Get batch analysis summary")
    print("  • 'history' - View analysis history")
    print("  • 'help' - Show detailed help")
    print("  • 'stats' - Show session statistics")
    print("  • 'reset' - Clear conversation")
    print("  • 'exit' - Quit VISIOS\n")
    
    print("="*75 + "\n")
    
    chat = VisiosChat()
    conversation_count = 0
    max_conversations = 100
    
    # Welcome message
    print("💬 VISIOS: Hello! I'm ready to analyze water quality images.")
    print("           Upload photos and I'll detect harmful algal blooms.\n")
    
    while conversation_count < max_conversations:
        try:
            user_input = input("\033[36m🧠 You > \033[0m").strip()
            
            if not user_input:
                continue
            
            # Handle special commands
            if user_input.lower() in ["exit", "quit", "bye", "goodbye"]:
                stats = chat.get_stats()
                print("\n" + "="*75)
                print("📊 Final Session Summary:")
                print(f"   • Images Analyzed: {stats['images_processed']}")
                print(f"   • Messages Exchanged: {stats['messages_exchanged']}")
                print(f"   • Session Duration: {stats['session_duration_min']} minutes")
                print("\n👋 Thank you for using VISIOS!")
                print("   Stay safe around water and keep monitoring! 🌊")
                print("="*75 + "\n")
                break
            
            elif user_input.lower() == "reset":
                chat.reset()
                continue
            
            elif user_input.lower() == "stats":
                stats = chat.get_stats()
                print(f"\n📊 Session Statistics:")
                print(f"   • Images Processed: {stats['images_processed']}")
                print(f"   • Messages: {stats['messages_exchanged']}")
                print(f"   • Duration: {stats['session_duration_min']} min")
                print(f"   • Errors: {stats['errors']}\n")
                continue
            
            elif user_input.lower() == "help":
                print("\n" + "="*75)
                print("📖 VISIOS Help Guide")
                print("="*75)
                print("\n🎯 Main Functions:")
                print("  1. Image Analysis - Detect HABs in uploaded photos")
                print("  2. GPS Extraction - Get location data from image metadata")
                print("  3. Batch Processing - Analyze all images at once")
                print("  4. Risk Assessment - Evaluate bloom severity and safety")
                print("  5. Reporting - Generate comprehensive analysis reports")
                
                print("\n💬 Example Queries:")
                print("  • 'What images are available?'")
                print("  • 'Analyze lake_photo_01.jpg'")
                print("  • 'Show me a summary of all bloom detections'")
                print("  • 'What's the GPS location for image.jpg?'")
                print("  • 'Check bloom risk near 47.5°N, 9.5°E'")
                print("  • 'Generate a full report'")
                
                print("\n🔗 Integration:")
                print("  • CALIBRO - Validates findings with satellite data")
                print("  • HOMOGEN - Cross-references with sensor measurements")
                print("  • SWIM Platform - Part of comprehensive monitoring system")
                
                print("\n⚠️ Safety Notes:")
                print("  • Severe blooms require immediate action")
                print("  • Keep people and pets away from suspected blooms")
                print("  • Contact local authorities for confirmed HABs")
                print("="*75 + "\n")
                continue
            
            # Send to agent
            response = chat.send_message(user_input)
            print(f"\n\033[32m🤖 VISIOS >\033[0m {response}\n")
            
            conversation_count += 1
            
        except KeyboardInterrupt:
            print("\n\n⚠️  Interrupted. Type 'exit' to quit properly or press Enter to continue.\n")
            continue
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")
            print("💡 Try rephrasing your question or type 'help' for guidance.\n")
            continue
    
    if conversation_count >= max_conversations:
        print("\n⚠️  Maximum conversation limit reached. Please restart VISIOS.\n")

# ========================================
# Batch Processing Mode
# ========================================
def batch_analyze_directory(image_dir: Path = None, output_path: Path = None):
    """
    Batch analyze all images without interactive chat.
    Optimized for automated processing pipelines.
    
    Args:
        image_dir: Directory with images (default: data/visios_images)
        output_path: Report output path (default: outputs/visios/batch_report.md)
    """
    from swim.agents.visios.visios_agent import VisiosAgent
    
    print("="*75)
    print("🔄 VISIOS Batch Processing Mode")
    print("="*75 + "\n")
    
    agent = VisiosAgent(image_dir=image_dir)
    
    print(f"📂 Processing: {agent.image_dir}")
    print(f"🖼️  Images found: {len(agent.list_images())}\n")
    
    if not agent.list_images():
        print("⚠️  No images found. Please add images to the directory.\n")
        return None
    
    # Generate report
    start_time = time.time()
    report = agent.generate_report(output_format="markdown")
    processing_time = time.time() - start_time
    
    # Save report
    if output_path is None:
        output_path = agent.output_dir / f"batch_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Batch analysis complete!")
    print(f"⏱️  Processing time: {processing_time:.2f} seconds")
    print(f"📄 Report saved: {output_path}\n")
    
    # Show summary
    summary = agent.summarize_batch()
    print("📊 Quick Summary:")
    print(f"   • Total Images: {summary['statistics']['total_images']}")
    print(f"   • Avg Bloom Score: {summary['statistics']['average_bloom_score']:.1%}")
    print(f"   • High Risk Locations: {len(summary['high_risk_locations'])}")
    print(f"   • Geotagged Images: {summary['geotagged_images']}\n")
    
    return report

# ========================================
# API-style Function
# ========================================
def analyze_single_image_api(image_path: str, return_json: bool = True):
    """
    Programmatic API for single image analysis.
    
    Args:
        image_path: Path to image file
        return_json: Return dict (True) or JSON string (False)
    
    Returns:
        Analysis results
    """
    from swim.agents.visios.visios_agent import VisiosAgent
    import json
    
    agent = VisiosAgent()
    image_name = Path(image_path).name
    result = agent.analyze_image(image_name)
    
    if return_json:
        return result
    else:
        return json.dumps(result, indent=2)

# ========================================
# Entry Point with Multiple Modes
# ========================================
if __name__ == "__main__":
    import sys
    from datetime import datetime
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == "batch":
            # Batch processing mode
            batch_analyze_directory()
            
        elif command == "analyze" and len(sys.argv) > 2:
            # Single image analysis mode
            image_name = sys.argv[2]
            result = analyze_single_image_api(image_name, return_json=False)
            print("\n" + "="*75)
            print("🖼️  VISIOS Single Image Analysis")
            print("="*75)
            print(result)
            print("="*75 + "\n")
            
        elif command == "help":
            print("\n" + "="*75)
            print("🖼️  VISIOS Command Line Interface")
            print("="*75)
            print("\nUsage:")
            print("  python visios_agent_graph.py              # Interactive chat mode")
            print("  python visios_agent_graph.py batch        # Batch analysis mode")
            print("  python visios_agent_graph.py analyze <image.jpg>  # Single image")
            print("  python visios_agent_graph.py help         # Show this message")
            print("\n" + "="*75 + "\n")
        else:
            print(f"⚠️  Unknown command: {command}")
            print("💡 Try: python visios_agent_graph.py help\n")
    else:
        # Default: interactive chat mode
        launch_chat()