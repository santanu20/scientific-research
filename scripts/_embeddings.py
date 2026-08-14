"""Semantic embedding ranking using fastembed (BGE models via ONNX).

Replaces TF-IDF cosine similarity for paper-query relevance scoring.
Uses BAAI/bge-base-en-v1.5 — a high-quality general-purpose embedding model
that works well for scientific text without requiring torch/GPU.

Falls back to TF-IDF if fastembed unavailable.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger("scientific_research.embeddings")

_model = None
# Pluggable embedding model — default bge-base-en-v1.5 (768-dim, general).
# Override for domain-specific ranking, e.g. SPECTER-class scientific models
# when available in fastembed's catalog: SCIENTIFIC_RESEARCH_EMBED_MODEL=<name>.
# Cache keys embed the model name, so switching models never mixes vectors.
import os as _os

_model_name = _os.environ.get(
    "SCIENTIFIC_RESEARCH_EMBED_MODEL", "BAAI/bge-base-en-v1.5"
)
_cache_dir = Path.home() / ".cache" / "scientific_research" / "embeddings"


def _get_model():
    """Lazily load the BGE embedding model (singleton)."""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(model_name=_model_name)
        log.info("Loaded embedding model: %s", _model_name)
    return _model


def is_available() -> bool:
    """Check if semantic embeddings are available (without loading model)."""
    try:
        from fastembed import TextEmbedding  # noqa: F401

        return True
    except ImportError:
        return False


def _cache_key(text: str) -> str:
    """Content-hash cache key for a text embedding."""
    return hashlib.sha256(f"{_model_name}|{text[:500]}".encode()).hexdigest()[:32]


def _get_cached(key: str) -> np.ndarray | None:
    """Get cached embedding."""
    cache_file = _cache_dir / f"{key}.npy"
    if cache_file.exists():
        try:
            return np.load(cache_file)
        except Exception:
            pass
    return None


def _put_cached(key: str, emb: np.ndarray) -> None:
    """Cache an embedding."""
    try:
        _cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(_cache_dir / f"{key}.npy", emb)
    except Exception as e:
        log.debug("Embedding cache write failed: %s", e)


def embed_texts(texts: list[str], use_cache: bool = True) -> np.ndarray | None:
    """Embed a list of texts using BGE model.

    Returns (n_texts, dim) array or None if unavailable.
    Uses content-hash disk cache to avoid recomputation.
    """
    if not texts:
        return None

    try:
        model = _get_model()
    except Exception as e:
        log.warning("Embedding model load failed: %s — falling back to TF-IDF", e)
        return None

    results = []
    uncached_indices = []
    uncached_texts = []

    for i, text in enumerate(texts):
        if use_cache:
            key = _cache_key(text)
            cached = _get_cached(key)
            if cached is not None:
                results.append((i, cached))
                continue
        uncached_indices.append(i)
        uncached_texts.append(text)

    if uncached_texts:
        try:
            new_embs = list(model.embed(uncached_texts))
            for idx, emb in zip(uncached_indices, new_embs):
                emb_arr = np.array(emb, dtype=np.float32)
                results.append((idx, emb_arr))
                if use_cache:
                    _put_cached(_cache_key(texts[idx]), emb_arr)
        except Exception as e:
            log.warning("Embedding computation failed: %s", e)
            return None

    results.sort(key=lambda x: x[0])
    return np.array([emb for _, emb in results]) if results else None


def semantic_rank(
    query: str,
    papers: list[dict],
    min_score: float = 0.3,
) -> list[tuple[dict, float]]:
    """Rank papers by semantic similarity to query.

    Returns list of (paper, score) tuples sorted by score (descending).
    Score is cosine similarity in [0, 1] range for BGE models.
    Falls back to empty list if embeddings unavailable.
    """
    if not papers:
        return []

    # Build paper texts
    texts = [query]
    for p in papers:
        text = " ".join(
            filter(
                None,
                [
                    p.get("title", "") or "",
                    (p.get("abstract") or "")[:500],
                    (p.get("key_finding") or "")[:200],
                ],
            )
        )
        texts.append(text)

    # Embed all texts
    embeddings = embed_texts(texts)
    if embeddings is None:
        return []

    # Compute cosine similarity
    query_emb = embeddings[0:1]
    paper_embs = embeddings[1:]

    # Normalize for cosine similarity
    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
    paper_norms = paper_embs / (
        np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-10
    )

    scores = (paper_norms @ query_norm.T).flatten()

    # Sort by score
    ranked = list(zip(papers, scores))
    ranked.sort(key=lambda x: -x[1])

    # Filter by minimum score
    ranked = [(p, s) for p, s in ranked if s >= min_score]

    return ranked


def embed_paper(query: str, papers: list[dict]) -> list[tuple[dict, float]]:
    """Semantic ranking wrapper that integrates with _filter_topical.

    Returns ranked papers with scores. If embeddings unavailable,
    returns empty list (caller falls back to TF-IDF).
    """
    try:
        return semantic_rank(query, papers, min_score=0.25)
    except Exception as e:
        log.warning("Semantic ranking failed: %s — falling back to TF-IDF", e)
        return []
