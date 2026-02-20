# swim/rag/ingest.py

"""High-level ingestion pipeline for the SWIM RAG knowledge base."""

import logging
from pathlib import Path
from typing import Any, Dict, List

from swim.rag.document_processor import (
    KNOWLEDGE_DIR,
    INDEX_PATH,
    Document,
    knowledge_base,
    load_lake_knowledge,
)
from swim.rag.file_loader import load_file, load_directory

logger = logging.getLogger(__name__)

_retrievers_loaded = False


def _ensure_retrievers_loaded():
    """Load all built-in retrievers once."""
    global _retrievers_loaded
    if _retrievers_loaded:
        return
    _retrievers_loaded = True

    # Lake knowledge
    lake_docs = load_lake_knowledge()
    knowledge_base.add_documents(lake_docs)

    # Domain retrievers
    from swim.rag.retrievers.climate_retriever import load_climate_documents
    from swim.rag.retrievers.policy_retriever import load_policy_documents
    from swim.rag.retrievers.reports_retriever import load_report_documents

    load_climate_documents()
    load_policy_documents()
    load_report_documents()

    # User-uploaded files from data/knowledge/
    if KNOWLEDGE_DIR.is_dir():
        user_docs = load_directory(KNOWLEDGE_DIR)
        if user_docs:
            knowledge_base.add_documents(user_docs)

    logger.info(
        "Retrievers loaded: %d total documents in knowledge base",
        len(knowledge_base.documents),
    )


def ingest_file(path: Path, category: str = "uploaded") -> int:
    """Load a file, add to knowledge base, rebuild index, and save.

    Returns the number of chunks added.
    """
    _ensure_retrievers_loaded()
    docs = load_file(path, category=category)
    if not docs:
        return 0
    knowledge_base.add_documents(docs)
    knowledge_base.build_index()
    knowledge_base.save()
    logger.info("Ingested %d chunks from %s", len(docs), path.name)
    return len(docs)


def ingest_uploaded_bytes(
    filename: str, content_bytes: bytes, category: str = "uploaded"
) -> int:
    """Save raw bytes to data/knowledge/ and ingest the resulting file.

    Returns the number of chunks added.
    """
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = KNOWLEDGE_DIR / filename
    dest.write_bytes(content_bytes)
    return ingest_file(dest, category=category)


def get_kb_stats() -> Dict[str, Any]:
    """Return knowledge base statistics."""
    _ensure_retrievers_loaded()
    docs = knowledge_base.documents
    categories: Dict[str, int] = {}
    sources: Dict[str, int] = {}
    for doc in docs:
        categories[doc.category] = categories.get(doc.category, 0) + 1
        sources[doc.source] = sources.get(doc.source, 0) + 1

    return {
        "total_documents": len(docs),
        "categories": categories,
        "sources": sources,
        "index_built": knowledge_base.embeddings is not None,
        "index_path": str(INDEX_PATH),
        "knowledge_dir": str(KNOWLEDGE_DIR),
    }


def reset_kb():
    """Clear all documents and the index."""
    global _retrievers_loaded
    knowledge_base.documents.clear()
    knowledge_base.embeddings = None
    _retrievers_loaded = False
    if INDEX_PATH.exists():
        INDEX_PATH.unlink()
    logger.info("Knowledge base reset")


def query_kb(question: str, top_k: int = 3) -> List[Dict[str, Any]]:
    """Query the knowledge base and return structured results."""
    _ensure_retrievers_loaded()
    if knowledge_base.embeddings is None and knowledge_base.documents:
        knowledge_base.build_index()

    from swim.rag.embedding import embed_query
    import numpy as np

    if knowledge_base.embeddings is None or not knowledge_base.documents:
        return []

    q_vec = embed_query(question)
    norms = np.linalg.norm(knowledge_base.embeddings, axis=1, keepdims=True) + 1e-8
    normed = knowledge_base.embeddings / norms
    q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
    scores = normed @ q_norm
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for i in top_idx:
        doc = knowledge_base.documents[i]
        results.append({
            "text": doc.text,
            "source": doc.source,
            "category": doc.category,
            "score": round(float(scores[i]), 4),
        })
    return results
