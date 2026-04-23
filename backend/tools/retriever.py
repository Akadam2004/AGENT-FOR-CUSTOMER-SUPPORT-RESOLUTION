"""
Knowledge retrieval utility backed by SentenceTransformers + FAISS.
"""

from pathlib import Path
from typing import List, Optional, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL_NAME = "models/all-MiniLM-L6-v2"
TOP_K_DEFAULT = 2
CHUNK_MAX_CHARS = 400

_model: Optional[SentenceTransformer] = None
_index: Optional[faiss.IndexFlatIP] = None
_chunks: Optional[list[str]] = None


def _knowledge_file_path() -> Path:
    # tools/retriever.py -> backend/data/knowledge.txt
    return Path(__file__).resolve().parent.parent / "data" / "knowledge.txt"


def _split_long_text(text: str, max_chars: int = CHUNK_MAX_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    words = text.split()
    parts: List[str] = []
    current: List[str] = []
    current_len = 0
    for word in words:
        next_len = current_len + len(word) + (1 if current else 0)
        if next_len > max_chars and current:
            parts.append(" ".join(current).strip())
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = next_len

    if current:
        parts.append(" ".join(current).strip())
    return [p for p in parts if p]


def _load_chunks() -> List[str]:
    raw_text = _knowledge_file_path().read_text(encoding="utf-8").strip()
    if not raw_text:
        return []

    # Use paragraph blocks first, then split oversized blocks.
    paragraphs = [block.strip() for block in raw_text.split("\n\n") if block.strip()]
    chunks: List[str] = []
    for paragraph in paragraphs:
        chunks.extend(_split_long_text(paragraph))
    return chunks


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _build_index(chunks: List[str]) -> faiss.IndexFlatIP:
    model = _get_model()
    vectors = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    embeddings = np.asarray(vectors, dtype=np.float32)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def _ensure_retriever_initialized() -> Tuple[faiss.IndexFlatIP, List[str]]:
    global _index, _chunks
    if _index is not None and _chunks is not None:
        return _index, _chunks

    chunks = _load_chunks()
    if not chunks:
        # Keep an empty retriever state with a valid but empty index.
        _chunks = []
        _index = faiss.IndexFlatIP(384)
        return _index, _chunks

    _chunks = chunks
    _index = _build_index(chunks)
    return _index, _chunks


def get_relevant_context(query: str) -> List[str]:
    """
    Return the top 2 most relevant chunks for the query.
    """
    query_text = query.strip()
    if not query_text:
        return []

    index, chunks = _ensure_retriever_initialized()
    if not chunks or index.ntotal == 0:
        return []

    model = _get_model()
    query_vector = model.encode(
        [query_text],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_embedding = np.asarray(query_vector, dtype=np.float32)

    top_k = min(TOP_K_DEFAULT, len(chunks))
    _, indices = index.search(query_embedding, top_k)
    return [chunks[i] for i in indices[0] if 0 <= i < len(chunks)]
