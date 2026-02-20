# swim/rag/query_router.py

"""Route queries to the appropriate retriever based on intent."""

import re
from typing import List, Optional

from swim.rag.document_processor import Document, knowledge_base, load_lake_knowledge


def ensure_knowledge_loaded():
    """Initialize the knowledge base if not already loaded."""
    if not knowledge_base.documents:
        docs = load_lake_knowledge()
        knowledge_base.add_documents(docs)
        knowledge_base.build_index()


def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant context for a query, formatted as a string."""
    ensure_knowledge_loaded()
    docs = knowledge_base.query(query, top_k=top_k)
    if not docs:
        return ""
    parts = []
    for i, doc in enumerate(docs, 1):
        parts.append(f"[{i}] ({doc.category}) {doc.text}")
    return "\n".join(parts)


def classify_intent(query: str) -> str:
    """Classify the user query into an intent category."""
    q = query.lower()
    if any(w in q for w in ["policy", "regulation", "directive", "law", "standard"]):
        return "policy"
    if any(w in q for w in ["climate", "weather", "temperature", "precipitation"]):
        return "climate"
    if any(w in q for w in ["report", "history", "past", "previous"]):
        return "report"
    if any(w in q for w in ["lake", "bodensee", "chiemsee", "müritz", "ammersee", "starnberg"]):
        return "lake_info"
    return "general"
