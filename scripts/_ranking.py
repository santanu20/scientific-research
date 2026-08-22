#!/usr/bin/env python3
"""Multi-criteria paper ranking for scientific-research skill.

Inspired by:
  - ASReview (TF-IDF + classifier for systematic review screening)
  - arxiv-sanity-lite (Karpathy: TF-IDF cosine similarity)
  - Inciteful / Connected Papers (citation network centrality)

Scores papers on 4 criteria, combines with weights, applies diversity filter.

Criteria (each normalized to [0, 1]):
  1. Semantic relevance (30%): TF-IDF cosine between query and title+abstract
  2. Citation influence (25%): log-normalized citation count
  3. Network centrality (25%): bibliographic coupling degree (shared references)
  4. Recency (20%): exponential decay favoring recent papers

Then applies:
  - Diversity filter: max 3 papers per first author, max 5 per venue
  - Recency balance: ensure ≥20% papers from last 5 years
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter
from typing import Any

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

log = logging.getLogger("scientific_research.ranking")

# Weights for each criterion (must sum to 1.0)
# Citation influence weighted highest — a foundational paper with 5000 citations
# should never be excluded regardless of age. Recency is a small boost, not a penalty.
# Venue quality uses OpenAlex source h_index (Nature=500+ vs low-IF journal=10).
# Context-dominant weighting: similarity to THE QUERY plus IDF-weighted
# query-facet coverage carry 60%. Popularity signals only differentiate
# WITHIN the pool the query already retrieved — they can never pull an
# off-topic paper above an on-topic one. No field vocabulary anywhere:
# every signal is derived from query x pool at runtime.
WEIGHTS = {
    "semantic": 0.40,
    "facet": 0.20,
    "citation": 0.15,
    "network": 0.10,
    "recency": 0.05,
    "venue": 0.10,
}

# Dynamic relevance floor (pool-relative, not absolute):
RELEVANT_SEM_FRACTION = 0.25  # of the pool's best semantic score
RELEVANT_MEDIAN_FRACTION = 0.70  # and of the pool's MEDIAN (kills polysemy stragglers)
RELEVANT_FACET_FRACTION = 0.50  # or covers half the query's facets
MIN_USEFUL_CORPUS = 3  # weak-context papers excluded once this many strong exist

# Diversity caps
MAX_PER_FIRST_AUTHOR = 3
MAX_PER_VENUE = 5

# Recency: ensure at least this fraction from last N years
MIN_RECENT_FRACTION = 0.20
RECENT_YEARS = 5


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def _first_author_name(paper: Any) -> str:
    """Extract first author's last name for dedup."""
    authors = getattr(paper, "authors", None) or []
    if authors and isinstance(authors, list) and len(authors) > 0:
        name = authors[0].get("name", "") if isinstance(authors[0], dict) else str(authors[0])
        parts = name.split()
        if parts:
            return parts[-1].lower()
    return "unknown"


def _venue_name(paper: Any) -> str:
    """Extract venue for dedup."""
    venue = getattr(paper, "venue", None) or ""
    return venue.lower().strip()[:50] if venue else "unknown"


def semantic_relevance_scores(query: str, papers: list) -> np.ndarray:
    """Semantic relevance scoring — BGE embeddings (primary) or TF-IDF (fallback).

    Primary: BGE-base-en-v1.5 embeddings via fastembed (ONNX, no torch).
    Fallback: TF-IDF cosine similarity (ASReview/arxiv-sanity-lite style).

    Includes query expansion with domain synonyms.

    Returns array of [0, 1] scores.
    """
    if not papers:
        return np.array([])

    # Try BGE embeddings first (SOTA — semantic, not just keyword matching)
    try:
        import numpy as _np_bge

        expanded_query = _expand_query(query)
        texts = [expanded_query]
        for p in papers:
            title = getattr(p, "title", "") or ""
            abstract = getattr(p, "abstract", "") or ""
            texts.append(title + " " + abstract)  # full text (chunk-pooled downstream)

        from _embeddings import embed_text_full

        _vecs = [embed_text_full(t) for t in texts]
        if any(v is None for v in _vecs):
            embeddings = None
        else:
            import numpy as _np

            embeddings = _np.array(_vecs)
        if embeddings is not None and embeddings.shape[0] == len(texts):
            query_emb = embeddings[0:1]
            paper_embs = embeddings[1:]
            # Cosine similarity
            q_norm = query_emb / (_np_bge.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-10)
            p_norm = paper_embs / (_np_bge.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-10)
            sims = (p_norm @ q_norm.T).flatten()
            # ABSOLUTE cosine similarity [0, 1] — no pool-max rescale,
            # so a weak match stays weak even in a poor pool.
            sims = np.clip(sims, 0.0, 1.0)
            log.info("Using BGE embeddings for semantic relevance (%d papers)", len(papers))
            return sims
    except Exception as e:
        log.debug("BGE embeddings unavailable, using TF-IDF: %s", e)

    # Fallback: TF-IDF cosine similarity
    # Expand query with synonyms and abbreviations
    expanded_query = _expand_query(query)

    # Build text corpus: expanded query + all papers
    texts = [expanded_query.lower().strip()]
    for p in papers:
        title = getattr(p, "title", "") or ""
        abstract = getattr(p, "abstract", "") or ""
        texts.append((title + " " + abstract).lower().strip())

    try:
        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts)
        # Cosine similarity between query (row 0) and each paper (rows 1+)
        sims = cosine_similarity(matrix[0:1], matrix[1:]).flatten()
        return np.clip(sims, 0.0, 1.0)
    except ValueError as e:
        log.warning("TF-IDF failed: %s — using uniform scores", e)
        return np.ones(len(papers)) / max(len(papers), 1)


def query_facet_scores(query: str, papers: list) -> np.ndarray:
    """IDF-weighted coverage of the query's own content terms.

    Fully dynamic context scoring: facets ARE the query terms; IDF comes
    from the retrieved pool itself. Returns [0, 1] per paper.
    """
    if not papers:
        return np.array([])
    try:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

        stop = ENGLISH_STOP_WORDS
    except ImportError:  # pragma: no cover - sklearn is a hard dep
        stop = frozenset()

    tokens = {
        tok
        for tok in re.findall(r"[a-z0-9]+", query.lower())
        if len(tok) >= 3 and tok not in stop
    }
    if not tokens:
        return np.zeros(len(papers))

    docs = [
        ((getattr(p, "title", "") or "") + " " + (getattr(p, "abstract", "") or "")).lower()
        for p in papers
    ]
    n_docs = max(len(docs), 1)
    idf = {
        tok: math.log((n_docs + 1) / (sum(1 for d in docs if tok in d) + 1)) + 1.0
        for tok in tokens
    }
    total_idf = sum(idf.values()) or 1.0
    return np.array(
        [sum(idf[tok] for tok in tokens if tok in doc) / total_idf for doc in docs]
    )


# Domain synonym map for query expansion
# _SYNONYM_GROUPS removed — static domain vocabulary (dynamic-context mandate).


def _expand_query(query: str) -> str:
    """Return the query unchanged (dynamic-context mandate, 2026-08-22).

    The former 16 hardcoded domain synonym groups expanded only queries
    in those domains — asymmetric static behavior. Semantic similarity,
    facet coverage and the relevance floor own recall now.
    """
    return query


def citation_influence_scores(papers: list) -> np.ndarray:
    """Citation influence combining total citations + citation velocity.

    70% weight: log(1 + total_citations) — rewards foundational papers
    30% weight: log(1 + citations_per_year) — rewards high-impact recent work

    This ensures a 2013 paper with 1377 citations (106/yr) still ranks
    high, but a 2023 paper with 556 citations (185/yr) also gets credit
    for its velocity.
    """
    if not papers:
        return np.array([])
    current_year = time.gmtime().tm_year
    raw = np.array([_safe_float(getattr(p, "citation_count", 0)) for p in papers])
    years = np.array([max(1, current_year - _safe_float(getattr(p, "year", current_year))) for p in papers])
    velocity = raw / years

    log_total = np.log1p(raw)
    log_velocity = np.log1p(velocity)

    combined = 0.7 * log_total + 0.3 * log_velocity
    if combined.max() > 0:
        return combined / combined.max()
    return np.zeros(len(papers))


def network_centrality_scores(papers: list) -> np.ndarray:
    """Combined network centrality: bibliographic coupling + co-citation + PageRank.

    Inspired by Inciteful / Connected Papers:
    - Coupling: papers that share references (cite the same sources)
    - Co-citation: papers cited together by other corpus papers
    - PageRank: papers cited by other important papers in the corpus

    Returns average of the 3 sub-scores, normalized to [0, 1].
    """
    if not papers:
        return np.array([])

    coupling = _bibliographic_coupling_scores(papers)
    cocitation = _co_citation_scores(papers)
    pagerank = _pagerank_scores(papers)

    # Average of 3 sub-scores
    combined = (coupling + cocitation + pagerank) / 3.0
    if combined.max() > 0:
        return combined / combined.max()
    return np.zeros(len(papers))


def _bibliographic_coupling_scores(papers: list) -> np.ndarray:
    """Bibliographic coupling: papers that cite the same references.

    For each pair of papers, count shared referenced_works entries.
    """
    n = len(papers)
    ref_sets = []
    for p in papers:
        refs = getattr(p, "referenced_works", None) or []
        ref_sets.append(set(refs) if refs else set())

    scores = np.zeros(n)
    for i in range(n):
        for j in range(i + 1, n):
            shared = len(ref_sets[i] & ref_sets[j])
            if shared > 0:
                scores[i] += shared
                scores[j] += shared

    # Reward papers with many references (connect to more papers)
    for i, refs in enumerate(ref_sets):
        if len(refs) > 0:
            scores[i] += math.log1p(len(refs)) * 0.5

    if scores.max() > 0:
        return scores / scores.max()
    return np.zeros(n)


def _co_citation_scores(papers: list) -> np.ndarray:
    """Co-citation: papers cited together by other papers in the corpus.

    If paper C's referenced_works contains both paper A and paper B
    (where A, B, C are all in the corpus), then A and B are co-cited.

    This is the within-corpus approximation — only counts co-citations
    from papers that are themselves in the corpus. Still captures the
    local citation structure effectively.
    """
    n = len(papers)

    # Build set of corpus paper IDs (DOIs or openalex_ids)
    corpus_ids = set()
    id_to_idx = {}
    for i, p in enumerate(papers):
        pid = getattr(p, "doi", None) or getattr(p, "openalex_id", None) or ""
        if pid:
            corpus_ids.add(pid)
            id_to_idx[pid] = i
        # Also add openalex_id if different from doi
        oa_id = getattr(p, "openalex_id", None)
        if oa_id and oa_id != pid:
            corpus_ids.add(oa_id)
            id_to_idx[oa_id] = i

    if not corpus_ids:
        return np.zeros(n)

    # For each paper, find which corpus papers it cites
    co_cite_matrix = np.zeros((n, n))
    for c_idx, p in enumerate(papers):
        refs = getattr(p, "referenced_works", None) or []
        if not refs:
            continue
        # Find which refs are in our corpus
        cited_in_corpus = []
        for ref in refs:
            # Try matching by DOI or openalex_id
            ref_str = str(ref)
            if ref_str in corpus_ids:
                cited_idx = id_to_idx.get(ref_str)
                if cited_idx is not None and cited_idx != c_idx:
                    cited_in_corpus.append(cited_idx)

        # For each pair of cited papers, increment co-citation
        for i_pos, a_idx in enumerate(cited_in_corpus):
            for b_idx in cited_in_corpus[i_pos + 1 :]:
                co_cite_matrix[a_idx, b_idx] += 1
                co_cite_matrix[b_idx, a_idx] += 1

    # Sum co-citation counts per paper
    scores = co_cite_matrix.sum(axis=1)

    if scores.max() > 0:
        return scores / scores.max()
    return np.zeros(n)


def _pagerank_scores(papers: list, damping: float = 0.85, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
    """PageRank on the within-corpus citation network (numpy vectorized).

    A paper is important if it's cited by other important papers.
    Uses matrix power iteration — fully vectorized with numpy.

    Graph: paper A → paper B if A cites B (B is in A's referenced_works).
    """
    n = len(papers)
    if n == 0:
        return np.array([])
    if n == 1:
        return np.array([1.0])

    # Build set of corpus paper IDs
    corpus_ids = set()
    id_to_idx = {}
    for i, p in enumerate(papers):
        pid = getattr(p, "doi", None) or getattr(p, "openalex_id", None) or ""
        if pid:
            corpus_ids.add(pid)
            id_to_idx[pid] = i
        oa_id = getattr(p, "openalex_id", None)
        if oa_id and oa_id != pid:
            corpus_ids.add(oa_id)
            id_to_idx[oa_id] = i

    # Build adjacency matrix: A[i][j] = 1 if paper i cites paper j
    adj = np.zeros((n, n))
    for i, p in enumerate(papers):
        refs = getattr(p, "referenced_works", None) or []
        for ref in refs:
            ref_str = str(ref)
            j = id_to_idx.get(ref_str)
            if j is not None and j != i:
                adj[i][j] = 1.0

    # Compute out-degrees
    out_degree = adj.sum(axis=1)

    # Build transition matrix (row-stochastic) — vectorized
    trans = np.zeros((n, n))
    nonzero = out_degree > 0
    trans[nonzero] = adj[nonzero] / out_degree[nonzero, np.newaxis]

    # Handle dangling nodes: redistribute their score uniformly
    dangling = ~nonzero
    if dangling.any():
        trans[dangling] = 1.0 / n

    # Power iteration (fully vectorized via matrix multiplication)
    scores = np.ones(n) / n
    trans_T = trans.T  # transpose for column-stochastic
    for iteration in range(max_iter):
        new_scores = damping * (trans_T @ scores) + (1 - damping) / n
        diff = np.abs(new_scores - scores).sum()
        scores = new_scores
        if diff < tol:
            log.debug("PageRank converged in %d iterations", iteration + 1)
            break

    # Normalize to [0, 1]
    if scores.max() > 0:
        return scores / scores.max()
    return np.zeros(n)


def venue_quality_scores(papers: list) -> np.ndarray:
    """Venue/journal quality based on OpenAlex source h_index.

    Fetches h_index for each unique venue via OpenAlex Sources API.
    High-IF journals (Nature, Science) have h_index 500+ → score 1.0.
    Low-IF journals have h_index 10-20 → score 0.3.
    Papers without venue data get 0.5 (neutral).

    Caches results to avoid repeated API calls.
    """
    if not papers:
        return np.array([])

    import json as _json
    import urllib.error
    import urllib.request

    # Extract source IDs from raw_metadata
    source_ids = {}
    venue_names = {}
    for p in papers:
        raw = getattr(p, "raw_metadata", None)
        if not raw:
            # Try dict access if p is a dict-like
            raw = p.get("raw_metadata") if hasattr(p, "get") else None
        if not raw:
            continue

        # Try to get source from primary_location
        loc = raw.get("primary_location", {}) or {}
        source = loc.get("source", {}) or {}
        src_id = source.get("id", "")
        src_name = source.get("display_name", "") or getattr(p, "venue", "") or ""

        if src_name:
            venue_names[id(p)] = src_name
        if src_id:
            source_ids[id(p)] = src_id

    # Batch-fetch h_index for unique source IDs
    h_index_cache: dict[str, int] = {}
    unique_src_ids = set(source_ids.values())

    for src_url in unique_src_ids:
        # Convert "https://openalex.org/S12345" to API URL
        src_id_short = src_url.split("/")[-1] if src_url else ""
        if not src_id_short:
            continue
        api_url = f"https://api.openalex.org/sources/{src_id_short}?select=h_index,works_count,cited_by_count"
        try:
            req = urllib.request.Request(api_url, headers={"User-Agent": "scientific-research/1.0"})
            with urllib.request.urlopen(req, timeout=5) as resp:
                data = _json.loads(resp.read().decode())
                h_idx = data.get("h_index", 0) or 0
                h_index_cache[src_url] = h_idx
                log.debug(
                    "Venue %s: h_index=%d",
                    data.get("display_name", src_id_short),
                    h_idx,
                )
        except (urllib.error.URLError, OSError, _json.JSONDecodeError) as e:
            log.debug("Source h_index lookup failed for %s: %s", src_id_short, e)
            h_index_cache[src_url] = 0

    # Known high-impact venues (fallback if API fails)
    _HIGH_IMPACT = {
        "nature",
        "science",
        "cell",
        "nature communications",
        "nature methods",
        "nature genetics",
        "nature medicine",
        "nature biotechnology",
        "nature materials",
        "nature physics",
        "nature chemistry",
        "nature geoscience",
        "nature climate change",
        "nature energy",
        "nature catalysis",
        "science advances",
        "proceedings of the national academy of sciences",
        "pnas",
        "the lancet",
        "lancet",
        "british medical journal",
        "bmj",
        "jama",
        "new england journal of medicine",
        "nejm",
        # Biology/Genetics
        "genome research",
        "nucleic acids research",
        "genome biology",
        "cell discovery",
        "cell reports",
        "cell stem cell",
        "molecular cell",
        "annual review of biochemistry",
        "advanced science",
        # Chemistry/Physics
        "physical review letters",
        "physical review",
        "journal of the american chemical society",
        "angewandte chemie",
        "chemical science",
        # Geo/Env
        "earth and planetary science letters",
        "geochimica et cosmochimica acta",
        "journal of geophysical research",
        "geology",
        "geochemistry geophysics geosystems",
        # CS
        "ieee transactions on pattern analysis",
    }

    # Compute scores
    scores = []
    for p in papers:
        src_id = source_ids.get(id(p), "")
        venue_name = (venue_names.get(id(p), "") or getattr(p, "venue", "") or "").lower()

        h_idx = h_index_cache.get(src_id, 0)

        if h_idx > 0:
            # Use actual h_index: log(1 + h_index) normalized
            score = math.log1p(h_idx) / math.log1p(600)  # Nature h_index ~600
        elif any(h in venue_name for h in _HIGH_IMPACT):
            score = 0.9  # known high-impact venue
        elif venue_name:
            score = 0.5  # unknown venue — neutral
        else:
            score = 0.5  # no venue info

        scores.append(min(1.0, score))

    return np.array(scores)


def recency_scores(papers: list) -> np.ndarray:
    """Recent-paper boost (NOT a penalty for old papers).

    Papers from the last 3 years: 1.0 (boost)
    Papers 3-10 years old: 0.7 (slight boost)
    Papers >10 years old: 0.5 (no penalty - foundational work stays valuable)

    This ensures milestone papers from 2005 with 10000 citations are NOT
    excluded just because they're old. The 10% weight means recency
    contributes at most 0.10 to the final score.
    """
    if not papers:
        return np.array([])
    current_year = time.gmtime().tm_year
    scores = []
    for p in papers:
        year = _safe_float(getattr(p, "year", current_year))
        years_ago = max(0, current_year - year)
        if years_ago <= 3:
            scores.append(1.0)
        elif years_ago <= 10:
            scores.append(0.7)
        else:
            scores.append(0.5)
    return np.array(scores)


def rank_papers(
    query: str,
    papers: list,
    max_n: int = 50,
    weights: dict | None = None,
) -> list:
    """Rank papers by multi-criteria score, apply diversity filter + recency balance.

    Args:
        query: research query string
        papers: list of PaperRecord objects
        max_n: max papers to return
        weights: optional weight override dict

    Returns:
        List of (paper, score, score_breakdown) tuples, sorted by score descending.
    """
    if not papers:
        return []

    w = weights or WEIGHTS
    n = len(papers)
    log.info("Ranking %d papers (weights: %s)", n, w)

    # Compute scores — every signal is query-x-pool derived at runtime;
    # no static discipline tables participate in ranking.
    sem = semantic_relevance_scores(query, papers)
    facet = query_facet_scores(query, papers)
    cite = citation_influence_scores(papers)
    net = network_centrality_scores(papers)
    rec = recency_scores(papers)
    ven = venue_quality_scores(papers)

    # Combine
    combined = (
        w.get("semantic", 0.40) * sem
        + w.get("facet", 0.20) * facet
        + w.get("citation", 0.15) * cite
        + w.get("network", 0.10) * net
        + w.get("recency", 0.05) * rec
        + w.get("venue", 0.10) * ven
    )

    # Sort by combined score (descending)
    ranked_indices = np.argsort(-combined)

    # Dynamic relevance floor (pool-relative): papers with neither real
    # query-similarity nor half-facet coverage sink below every strong
    # candidate for top-N slots. Corpus size preserved — weak papers stay
    # available as fill when the pool is small.
    sem_max = float(sem.max()) if len(sem) else 0.0
    median_sem = float(np.median(sem)) if len(sem) else 0.0
    # Semantic arm must clear BOTH the best-of-pool fraction and a robust
    # median bar: polysemous strays ("morphometric" fish studies under a
    # drainage-morphometry query) clear a max-fraction that collapses in
    # tiny pools, but sit far below the on-topic median.
    sem_floor = max(
        RELEVANT_SEM_FRACTION * sem_max, RELEVANT_MEDIAN_FRACTION * median_sem
    )
    strong = ((sem >= sem_floor) & (sem_max > 0)) | (
        facet >= RELEVANT_FACET_FRACTION
    )
    n_weak = int((~strong).sum())
    n_strong = int(strong.sum())

    if n_weak and n_strong >= MIN_USEFUL_CORPUS:
        # Enough genuine material — known off-topic papers are excluded
        # outright. Filling a corpus with them corrupts synthesis more
        # than a smaller honest corpus does.
        log.info(
            "Relevance floor: %d weak-context paper(s) EXCLUDED "
            "(%d strong candidates)",
            n_weak,
            n_strong,
        )
        ordered_indices = ranked_indices[strong[ranked_indices]]
    else:
        if n_weak:
            log.info(
                "Relevance floor: %d weak-context paper(s) demoted below strong set",
                n_weak,
            )
        ordered_indices = np.concatenate(
            [
                ranked_indices[strong[ranked_indices]],
                ranked_indices[~strong[ranked_indices]],
            ]
        )

    # Apply diversity filter
    result = []
    author_counts = Counter()
    venue_counts = Counter()
    recent_count = 0
    current_year = time.gmtime().tm_year

    for idx in ordered_indices:
        p = papers[idx]
        score = float(combined[idx])
        breakdown = {
            "semantic": float(sem[idx]),
            "facet": float(facet[idx]),
            "citation": float(cite[idx]),
            "network": float(net[idx]),
            "recency": float(rec[idx]),
            "venue": float(ven[idx]),
            "combined": score,
        }

        author = _first_author_name(p)
        venue = _venue_name(p)
        year = _safe_float(getattr(p, "year", 0))
        is_recent = year >= (current_year - RECENT_YEARS)

        # Diversity caps
        if author_counts[author] >= MAX_PER_FIRST_AUTHOR:
            continue
        if venue_counts[venue] >= MAX_PER_VENUE:
            continue

        result.append((p, score, breakdown))
        author_counts[author] += 1
        venue_counts[venue] += 1
        if is_recent:
            recent_count += 1

        if len(result) >= max_n:
            break

    # Recency balance: ensure ≥20% from last 5 years
    min_recent = max(1, int(max_n * MIN_RECENT_FRACTION))
    if recent_count < min_recent and len(ranked_indices) > len(result):
        # Find recent papers not yet included
        included_ids = {p.primary_id for p, _, _ in result}
        recent_candidates = []
        for idx in ordered_indices:
            p = papers[idx]
            if p.primary_id in included_ids:
                continue
            year = _safe_float(getattr(p, "year", 0))
            if year >= (current_year - RECENT_YEARS):
                recent_candidates.append(
                    (
                        p,
                        float(combined[idx]),
                        {
                            "semantic": float(sem[idx]),
                            "citation": float(cite[idx]),
                            "network": float(net[idx]),
                            "recency": float(rec[idx]),
                            "combined": float(combined[idx]),
                        },
                    )
                )

        # Insert recent papers (replace lowest-scoring non-recent papers)
        for p, score, breakdown in recent_candidates[: min_recent - recent_count]:
            # Remove last entry if it's not recent
            while result and len(result) >= max_n:
                last_p, _, _ = result[-1]
                last_year = _safe_float(getattr(last_p, "year", 0))
                if last_year < (current_year - RECENT_YEARS):
                    result.pop()
                    break
                else:
                    break
            result.append((p, score, breakdown))

    # Re-sort by score
    result.sort(key=lambda x: -x[1])

    log.info(
        "Ranked %d → %d papers (recent: %d/%d = %.0f%%, authors: %d unique, venues: %d unique)",
        n,
        len(result),
        recent_count,
        len(result),
        recent_count / max(len(result), 1) * 100,
        len(author_counts),
        len(venue_counts),
    )

    return result


if __name__ == "__main__":
    # Quick test
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from _sources import PaperRecord

    test_papers = [
        PaperRecord(
            title="CRISPR-Cas9 gene editing in human cells",
            abstract="We demonstrate efficient gene editing using CRISPR-Cas9 in human embryonic kidney cells.",
            year=2014,
            citation_count=5000,
            doi="10.1/a",
            authors=[{"name": "J Smith"}],
            venue="Nature",
            referenced_works=["10.2/x", "10.3/y", "10.4/z"],
        ),
        PaperRecord(
            title="Off-target effects of CRISPR-Cas9",
            abstract="We identified significant off-target mutations in CRISPR-Cas9 edited cells.",
            year=2023,
            citation_count=150,
            doi="10.5/b",
            authors=[{"name": "J Jones"}],
            venue="Science",
            referenced_works=["10.1/a", "10.3/y", "10.6/w"],
        ),
        PaperRecord(
            title="A review of CRISPR applications",
            abstract="This review covers CRISPR gene editing applications in various organisms.",
            year=2020,
            citation_count=800,
            doi="10.7/c",
            authors=[{"name": "J Smith"}],
            venue="Nature Reviews",
            referenced_works=["10.1/a", "10.5/b", "10.8/v"],
        ),
    ]

    ranked = rank_papers("CRISPR gene editing off-target effects", test_papers, max_n=3)
    for p, score, breakdown in ranked:
        print(f"  [{score:.3f}] {p.title[:60]} ({p.year}, {p.citation_count} cites)")
        print(
            f"    sem={breakdown['semantic']:.2f} cite={breakdown['citation']:.2f} net={breakdown['network']:.2f} rec={breakdown['recency']:.2f}"
        )
