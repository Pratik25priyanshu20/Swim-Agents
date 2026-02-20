# swim/observability/dag_visualizer.py

"""Visualize the SWIM agent pipeline DAG and execution traces."""

from typing import Any, Dict, List, Optional


PIPELINE_DAG = {
    "nodes": [
        {"id": "homogen", "label": "HOMOGEN", "type": "agent"},
        {"id": "calibro", "label": "CALIBRO", "type": "agent"},
        {"id": "visios", "label": "VISIOS", "type": "agent"},
        {"id": "predikt", "label": "PREDIKT", "type": "agent"},
        {"id": "risk", "label": "Risk Fusion", "type": "computation"},
    ],
    "edges": [
        {"from": "homogen", "to": "calibro"},
        {"from": "homogen", "to": "visios"},
        {"from": "calibro", "to": "predikt"},
        {"from": "visios", "to": "predikt"},
        {"from": "predikt", "to": "risk"},
        {"from": "calibro", "to": "risk"},
        {"from": "visios", "to": "risk"},
        {"from": "homogen", "to": "risk"},
    ],
}


def get_pipeline_dag() -> Dict[str, Any]:
    """Return the pipeline DAG definition for visualization."""
    return PIPELINE_DAG


def format_execution_trace(result: Dict[str, Any]) -> str:
    """Format a pipeline execution result into a readable trace string."""
    lines = ["SWIM Pipeline Execution Trace", "=" * 40]

    exec_summary = result.get("execution_summary", {})
    per_agent = exec_summary.get("per_agent", {})

    for agent in ["homogen", "calibro", "visios", "predikt"]:
        agent_result = result.get("agent_results", {}).get(agent, {})
        status = agent_result.get("status", "unknown")
        duration = per_agent.get(agent, 0)
        icon = "OK" if status == "success" else ("SKIP" if status == "skipped" else "FAIL")
        lines.append(f"  [{icon}] {agent.upper():10s} {duration:6.2f}s")

    risk = result.get("risk_assessment", {})
    lines.append("-" * 40)
    lines.append(f"  Risk: {risk.get('level', '?').upper()} ({risk.get('score', 0):.3f})")
    lines.append(f"  Total: {exec_summary.get('time', 0):.2f}s")
    return "\n".join(lines)


def build_mermaid_diagram(result: Optional[Dict[str, Any]] = None) -> str:
    """Generate a Mermaid flowchart of the pipeline, optionally annotated with results."""
    lines = ["graph TD"]

    status_style = {"success": ":::green", "error": ":::red", "skipped": ":::gray"}

    for node in PIPELINE_DAG["nodes"]:
        nid = node["id"]
        label = node["label"]
        if result:
            agent_result = result.get("agent_results", {}).get(nid, {})
            st = agent_result.get("status", "")
            suffix = status_style.get(st, "")
            duration = result.get("execution_summary", {}).get("per_agent", {}).get(nid, 0)
            lines.append(f'    {nid}["{label}<br/>{duration:.1f}s"]{suffix}')
        else:
            shape = f'["{label}"]' if node["type"] == "agent" else f'{{"{label}"}}'
            lines.append(f"    {nid}{shape}")

    for edge in PIPELINE_DAG["edges"]:
        lines.append(f"    {edge['from']} --> {edge['to']}")

    lines.extend([
        "    classDef green fill:#2d6a4f,stroke:#1b4332,color:#fff",
        "    classDef red fill:#9d0208,stroke:#6a040f,color:#fff",
        "    classDef gray fill:#6c757d,stroke:#495057,color:#fff",
    ])
    return "\n".join(lines)
