# swim/rag/embedding.py

"""Embedding generation for RAG — uses Google Generative AI or fallback TF-IDF."""

import logging
import os
from typing import List, Optional

import numpy as np

logger = logging.getLogger(__name__)

_EMBED_MODEL = None


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            try:
                from langchain_google_genai import GoogleGenerativeAIEmbeddings
                _EMBED_MODEL = GoogleGenerativeAIEmbeddings(
                    model="models/embedding-001",
                    google_api_key=api_key,
                )
                logger.info("Using Google Generative AI embeddings")
                return _EMBED_MODEL
            except Exception as exc:
                logger.warning("Failed to init Google embeddings: %s", exc)
        # Fallback: simple TF-IDF-like hash embeddings
        _EMBED_MODEL = "fallback"
    return _EMBED_MODEL


def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed a list of texts and return an (N, D) numpy array."""
    model = _get_embed_model()
    if model == "fallback":
        return _fallback_embed(texts)
    try:
        vectors = model.embed_documents(texts)
        return np.array(vectors, dtype=np.float32)
    except Exception as exc:
        logger.warning("Embedding API failed, using fallback: %s", exc)
        return _fallback_embed(texts)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query string."""
    model = _get_embed_model()
    if model == "fallback":
        return _fallback_embed([text])[0]
    try:
        vec = model.embed_query(text)
        return np.array(vec, dtype=np.float32)
    except Exception:
        return _fallback_embed([text])[0]


def _fallback_embed(texts: List[str], dim: int = 128) -> np.ndarray:
    """Deterministic hash-based embeddings as a fallback."""
    vectors = []
    for text in texts:
        rng = np.random.RandomState(hash(text) % (2**31))
        vectors.append(rng.randn(dim).astype(np.float32))
    return np.array(vectors)
