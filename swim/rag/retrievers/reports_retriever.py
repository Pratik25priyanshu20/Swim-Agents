# swim/rag/retrievers/reports_retriever.py

"""Retriever for historical HAB reports and prediction outcomes."""

import json
import logging
from pathlib import Path
from typing import List

from swim.rag.document_processor import Document, knowledge_base
from swim.shared.paths import OUTPUT_DIR

logger = logging.getLogger(__name__)


def load_historical_reports(max_reports: int = 50) -> List[Document]:
    """Ingest historical pipeline reports from outputs/ as RAG documents."""
    docs = []
    report_dir = OUTPUT_DIR / "main_agent"
    if not report_dir.exists():
        return docs

    for report_path in sorted(report_dir.glob("run_*.json"), reverse=True)[:max_reports]:
        try:
            with open(report_path, "r", encoding="utf-8") as f:
                report = json.load(f)
            location = report.get("location", {})
            lake = location.get("name", "unknown")
            agents = report.get("agents", {})

            summary_parts = [f"Pipeline report for {lake} at {report.get('timestamp', '?')}."]
            for agent_name, agent_data in agents.items():
                status = agent_data.get("status", "?")
                summary_parts.append(f"{agent_name}: {status}")

            text = " ".join(summary_parts)
            docs.append(Document(
                text=text,
                source=report_path.name,
                category="report",
                metadata={"lake": lake},
            ))
        except Exception as exc:
            logger.warning("Failed to parse report %s: %s", report_path.name, exc)

    return docs


def load_report_documents():
    """Add historical reports to the global knowledge base."""
    docs = load_historical_reports()
    if docs:
        knowledge_base.add_documents(docs)
        logger.info("Added %d historical reports to knowledge base", len(docs))
