# swim/rag/document_processor.py

"""Document ingestion and chunking for the SWIM RAG knowledge base."""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from swim.rag.embedding import embed_texts
from swim.shared.paths import PROJECT_ROOT

logger = logging.getLogger(__name__)

KNOWLEDGE_DIR = PROJECT_ROOT / "data" / "knowledge"
INDEX_PATH = PROJECT_ROOT / "data" / "rag_index.npz"


@dataclass
class Document:
    text: str
    source: str
    category: str  # "policy", "climate", "report", "lake_info"
    metadata: Dict = field(default_factory=dict)


class KnowledgeBase:
    """In-memory vector store for lake-specific knowledge."""

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: Optional[np.ndarray] = None

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        self.embeddings = None  # invalidate cache

    def build_index(self):
        """Compute embeddings for all documents."""
        if not self.documents:
            logger.warning("No documents to index")
            return
        texts = [d.text for d in self.documents]
        self.embeddings = embed_texts(texts)
        logger.info("Indexed %d documents (dim=%d)", len(texts), self.embeddings.shape[1])

    def query(self, question: str, top_k: int = 5) -> List[Document]:
        """Retrieve the top-k most relevant documents for a question."""
        if self.embeddings is None or len(self.documents) == 0:
            return []
        from swim.rag.embedding import embed_query
        q_vec = embed_query(question)
        # Cosine similarity
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-8
        normed = self.embeddings / norms
        q_norm = q_vec / (np.linalg.norm(q_vec) + 1e-8)
        scores = normed @ q_norm
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [self.documents[i] for i in top_idx]

    def save(self, path: Path = INDEX_PATH):
        """Persist the knowledge base to disk."""
        if self.embeddings is None:
            self.build_index()
        meta = [
            {"text": d.text, "source": d.source, "category": d.category, "metadata": d.metadata}
            for d in self.documents
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            embeddings=self.embeddings,
            meta=json.dumps(meta),
        )
        logger.info("Knowledge base saved to %s", path)

    def load(self, path: Path = INDEX_PATH):
        """Load a previously saved knowledge base."""
        if not path.exists():
            logger.warning("No knowledge base found at %s", path)
            return
        data = np.load(path, allow_pickle=True)
        self.embeddings = data["embeddings"]
        meta = json.loads(str(data["meta"]))
        self.documents = [Document(**m) for m in meta]
        logger.info("Loaded %d documents from %s", len(self.documents), path)


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Split text into overlapping chunks."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunks.append(" ".join(words[start:end]))
        start = end - overlap
    return chunks


def load_lake_knowledge() -> List[Document]:
    """Load built-in lake knowledge for the German lakes database."""
    from swim.agents.predikt.config import GERMAN_LAKES

    docs = []
    for name, meta in GERMAN_LAKES.items():
        text = (
            f"{name} is a {meta['trophic_status']} lake in {meta.get('region', 'Germany')}. "
            f"Area: {meta.get('area_km2', '?')} km², max depth: {meta.get('depth_max_m', '?')} m, "
            f"mean depth: {meta.get('depth_mean_m', '?')} m, volume: {meta.get('volume_km3', '?')} km³, "
            f"elevation: {meta.get('elevation_m', '?')} m. "
            f"Coordinates: {meta['lat']}, {meta['lon']}."
        )
        docs.append(Document(text=text, source="config", category="lake_info", metadata={"lake": name}))

    # Add general HAB knowledge
    docs.append(Document(
        text=(
            "Harmful Algal Blooms (HABs) are caused by cyanobacteria that produce toxins dangerous to humans and animals. "
            "Key drivers include water temperature above 20°C, high phosphorus and nitrogen levels, calm wind conditions, "
            "and high solar radiation. Chlorophyll-a concentrations above 20 µg/L indicate elevated bloom risk. "
            "German water quality standards (Badegewässerverordnung) require monitoring during bathing season (May-September)."
        ),
        source="domain_knowledge",
        category="policy",
    ))
    docs.append(Document(
        text=(
            "The EU Bathing Water Directive (2006/7/EC) classifies water quality as excellent, good, sufficient, or poor. "
            "Cyanobacteria must be monitored when visual inspection suggests bloom formation. "
            "WHO guidelines recommend a multi-tiered alert system: Level 1 (20,000 cells/mL), "
            "Level 2 (100,000 cells/mL), Level 3 (visible scum). "
            "German state agencies (LANUV, LfU) publish weekly bathing water quality reports."
        ),
        source="domain_knowledge",
        category="policy",
    ))

    return docs


# Global singleton
knowledge_base = KnowledgeBase()
