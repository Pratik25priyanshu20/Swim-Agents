# swim/rag/file_loader.py

"""Load uploaded files (PDF, TXT, MD, CSV) into Document objects for RAG ingestion."""

import logging
from pathlib import Path
from typing import List

from swim.rag.document_processor import Document, chunk_text

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".csv"}


def load_file(path: Path, category: str = "uploaded") -> List[Document]:
    """Read a file and return chunked Document objects.

    Supports PDF (via PyMuPDF), plain text/markdown, and CSV (via pandas).
    """
    path = Path(path)
    ext = path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {SUPPORTED_EXTENSIONS}")

    if ext == ".pdf":
        text = _read_pdf(path)
    elif ext == ".csv":
        text = _read_csv(path)
    else:
        text = path.read_text(encoding="utf-8", errors="replace")

    if not text.strip():
        logger.warning("Empty content from %s", path.name)
        return []

    chunks = chunk_text(text)
    docs = [
        Document(
            text=chunk,
            source=path.name,
            category=category,
            metadata={"file_path": str(path)},
        )
        for chunk in chunks
    ]
    logger.info("Loaded %d chunks from %s", len(docs), path.name)
    return docs


def load_directory(dir_path: Path) -> List[Document]:
    """Bulk-load all supported files from a directory."""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        logger.warning("Directory does not exist: %s", dir_path)
        return []

    docs: List[Document] = []
    for f in sorted(dir_path.iterdir()):
        if f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(load_file(f))
            except Exception as exc:
                logger.warning("Failed to load %s: %s", f.name, exc)
    logger.info("Loaded %d total chunks from %s", len(docs), dir_path)
    return docs


def _read_pdf(path: Path) -> str:
    """Extract text from a PDF using PyMuPDF (fitz)."""
    import fitz  # PyMuPDF

    doc = fitz.open(str(path))
    pages = []
    for page in doc:
        pages.append(page.get_text())
    doc.close()
    return "\n".join(pages)


def _read_csv(path: Path) -> str:
    """Convert a CSV into a text representation (one row per line)."""
    import pandas as pd

    df = pd.read_csv(path, low_memory=False)
    lines = [", ".join(f"{col}: {val}" for col, val in row.items()) for _, row in df.iterrows()]
    return "\n".join(lines)
