#!/usr/bin/env python3
"""PRISMA Phase 2 screening tool (title/abstract triage).

Pre-ranks a corpus against inclusion/exclusion criteria using keyword matching
and (optionally) sentence-transformer embeddings. Outputs PRISMA Phase 2 counts
plus a ranked list that the LLM can quickly adjudicate.

References:
    PRISMA 2020 — Page MJ et al. BMJ 2021;372:n71.
        Phase 2 = "Screening" (title/abstract review against inclusion criteria).

Embedding mode (optional):
    pip install sentence-transformers
    Uses all-MiniLM-L6-v2 by default (fast, ~80MB, Apache-2.0).
    Falls back to keyword mode if unavailable (fail-loud per §5 H2 — prints warning).
"""

from __future__ import annotations

# --- Skill venv bootstrap (shared: see _bootstrap.py) ---
if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()
# --- End bootstrap ---


import argparse
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus

log = logging.getLogger("scientific_research.screen")

# Optional embedding support — uses _embeddings.py (BGE via fastembed)
# Consolidated: no separate sentence-transformers dependency
_EMBED_OK = False
try:
    from _embeddings import is_available as _embeddings_available

    _EMBED_OK = _embeddings_available()
except ImportError:
    pass


# =============================================================================
# Keyword mode
# =============================================================================
def _score_paper_keywords(
    paper: PaperRecord, include: list[str], exclude: list[str]
) -> tuple[float, list[str], list[str]]:
    """TF-IDF-weighted keyword scoring (upgraded 2026-08-15).

    Old flat count treated 'magma' (everywhere) equal to 'thermobarometry'
    (decisive). Now each matched keyword carries its IDF weight over the
    scoring batch — same principle as the discovery co-occurrence filter.
    Falls back to flat counting when called on a single paper (no batch
    statistics available) — weights default to 1.0.
    Score = Σ idf(include matched) − 2·Σ idf(exclude matched).
    """

    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    if not text.strip():
        return (0.0, [], [])
    matched_inc = [k for k in include if k.lower() in text]
    matched_exc = [k for k in exclude if k.lower() in text]
    idf = getattr(_score_paper_keywords, "_batch_idf", {}) or {}
    w = lambda k: idf.get(k.lower(), 1.0)
    score = sum(w(k) for k in matched_inc) - 2.0 * sum(w(k) for k in matched_exc)
    return (float(score), matched_inc, matched_exc)


def set_keyword_idf(
    papers: list[PaperRecord], include: list[str], exclude: list[str]
) -> None:
    """Compute batch IDF for keywords over the corpus; attach to scorer."""
    n = max(len(papers), 1)
    idf: dict[str, float] = {}
    for k in set(include) | set(exclude):
        kl = k.lower()
        df = sum(
            1
            for p in papers
            if kl in ((p.title or "") + " " + (p.abstract or "")).lower()
        )
        idf[kl] = math.log((n + 1) / (df + 1)) + 1.0
    _score_paper_keywords._batch_idf = idf


# =============================================================================
# Embedding mode (semantic similarity, optional)
# =============================================================================
def _embed_text(text: str) -> list[float]:
    """Embed single text using BGE — lossless for long text (chunk-pooled)."""
    from _embeddings import embed_text_full

    vec = embed_text_full(text)
    return vec.tolist() if vec is not None else []


def _cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    return sum(x * y for x, y in zip(a, b))


def _score_paper_embeddings(
    paper: PaperRecord,
    query_vec: list[float],
    include_vecs: list[list[float]],
    exclude_vecs: list[list[float]],
) -> tuple[float, list[str], list[str]]:
    """Score by cosine similarity to include/exclude embeddings."""
    text = (paper.title or "") + ". " + (paper.abstract or "")
    if not text.strip():
        return (0.0, [], [])
    paper_vec = _embed_text(text)  # _embed_text pools long text losslessly
    inc_scores = [_cosine(paper_vec, iv) for iv in include_vecs]
    exc_scores = [_cosine(paper_vec, ev) for ev in exclude_vecs]
    matched_inc = [f"{s:.2f}" for s in inc_scores if s > 0.55]
    matched_exc = [f"{s:.2f}" for s in exc_scores if s > 0.65]
    primary = max(_cosine(paper_vec, query_vec), 0.0)
    score = (
        primary
        + sum(max(0.0, s - 0.4) for s in inc_scores)
        - 2.0 * sum(max(0.0, s - 0.5) for s in exc_scores)
    )
    return (float(score), matched_inc, matched_exc)


# =============================================================================
# Active-learning prioritization (ASReview ELAS-Ultra-style, reorder-only)
# =============================================================================
@dataclass
class ALResult:
    queue: list[dict]  # [{paper_id, title, p_include, al_rank, labeled}]
    n_seed: int
    n_unscreened: int
    model_info: dict
    recall_curve: list[dict]  # [{reviewed, predicted_includes, p_found}] — advisory


def prioritize_active_learning(
    papers: list[PaperRecord], labels: dict[str, str], seed_min: int = 5
) -> ALResult:
    """Prioritize unscreened papers via TF-IDF(1-2) + LinearSVC on seed labels.

    Model stack = ELAS-Ultra (ASReview LAB v2 default): TF-IDF bigrams +
    LinearSVC (Patterns 2025;6:101318). Ranking = P(include) descending:
    most-likely-relevant papers surface first → reviewer recall rises fastest
    (sensitivity-first, per O'Mara-Eves 2015 Cochrane guidance).

    SAFETY: reorder-only. No paper is excluded or dropped — every paper stays
    in the queue. Stopping is the reviewer's decision; the advisory recall
    curve shows predicted coverage, never a hard stop.

    labels: {paper_key: "include"|"exclude"} where paper_key = DOI or arXiv ID.
    "maybe" labels are ignored for training (ambiguous seed = noise).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.svm import LinearSVC

    def paper_key(p: PaperRecord) -> str:
        return (p.doi or p.arxiv_id or p.title or "").strip().lower()

    def paper_text(p: PaperRecord) -> str:
        return (p.title or "") + " " + (p.abstract or "")

    keyed = {paper_key(p): p for p in papers}
    train_x, train_y = [], []
    labeled_keys = set()
    for key, label in labels.items():
        k = key.strip().lower()
        if k not in keyed or label not in ("include", "exclude"):
            continue
        train_x.append(paper_text(keyed[k]))
        train_y.append(label)
        labeled_keys.add(k)
    n_inc = sum(1 for y in train_y if y == "include")
    n_exc = len(train_y) - n_inc
    if n_inc < seed_min or n_exc < seed_min // 2 or not train_x:
        raise ValueError(
            f"active learning needs ≥{seed_min} 'include' and ≥{seed_min // 2} "
            f"'exclude' seed labels; got include={n_inc}, exclude={n_exc}. "
            "Label more papers (labels file: one JSON object per line, "
            '{"doi": "...", "label": "include|exclude"}).'
        )
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    xs = vectorizer.fit_transform(train_x)
    clf = LinearSVC(C=1.0, class_weight="balanced", max_iter=5000, dual="auto")
    clf.fit(xs, train_y)

    unlabeled = [p for p in papers if paper_key(p) not in labeled_keys]
    queue: list[dict] = []
    if unlabeled:
        xu = vectorizer.transform([paper_text(p) for p in unlabeled])
        # decision function → probability-like score via Platt-style squash
        dec = clf.decision_function(xu)
        scores = [1.0 / (1.0 + pow(2.718281828459045, -d)) for d in dec]
        ranked = sorted(zip(unlabeled, scores), key=lambda t: -t[1])
        queue = [
            {
                "paper_id": p.primary_id,
                "doi": p.doi,
                "title": p.title,
                "p_include": round(float(s), 4),
                "al_rank": i + 1,
                "labeled": False,
            }
            for i, (p, s) in enumerate(ranked)
        ]
    # advisory recall curve: predicted cumulative includes vs reviewed count
    curve = []
    cum = 0
    for i, item in enumerate(queue, start=1):
        cum += 1 if item["p_include"] >= 0.5 else 0
        if i % 10 == 0 or i == len(queue):
            curve.append(
                {
                    "reviewed": i,
                    "predicted_includes": cum,
                    "p_found": round(
                        cum / max(1, sum(1 for q in queue if q["p_include"] >= 0.5)), 3
                    ),
                }
            )
    return ALResult(
        queue=queue,
        n_seed=len(train_x),
        n_unscreened=len(unlabeled),
        model_info={
            "classifier": "LinearSVC (balanced) on TF-IDF 1-2 grams, sublinear",
            "reference": "ASReview LAB v2 ELAS-Ultra — Patterns 2025;6:101318",
            "mode": "reorder-only; no auto-exclusion; reviewer decides stopping",
        },
        recall_curve=curve,
    )


def _al_stopping_section(seed_labels: dict[str, str], n_total: int) -> str:
    """Human-readable stopping guidance appended to the AL report."""
    order = list(seed_labels.values())
    adv = stopping_advisory(order, n_total)
    return (
        "\n## Stopping advisory (CAL u0=1 + ERV, beta posterior)\n\n"
        f"- reviewed so far: {adv['reviewed']} ({adv['relevant_found']} relevant)\n"
        f"- p̂(include): {adv['p_include_posterior']}\n"
        f"- expected relevant remaining: {adv['expected_relevant_remaining']}\n"
        f"- ERV next 100: {adv['erv_next_100']}\n"
        f"- CAL stop criterion met: {'YES' if adv['cal_stop'] else 'no'}; "
        f"ERV: {'YES' if adv['erv_stop'] else 'no'}\n"
        f"- {adv['note']}\n"
        f"- Re-run with updated --labels as you screen to refresh this advice.\n"
    )


def stopping_advisory(
    labels_in_order: list[str], n_total: int, u0: float = 1.0
) -> dict:
    """CAL- and ERV-style stopping estimates (ADVISORY ONLY — never auto-stops).

    - Posterior: beta(1,1) prior → p̂ = (r+1)/(n+2) after n reviewed, r relevant
      (Laplace smoothing; the published CAL estimator uses beta-binomial MLE —
      this simplification is documented, conservative for small n).
    - CAL-style (Callaghan & Müller-Büttner 2020): stop when expected relevant
      remaining (N_remaining · p̂) ≤ u0.
    - ERV (Extra Relevant found by Viewing next 100): 100·p̂; stop when < 1.
    labels_in_order: reviewer's screening decisions in review order
    ('include'/'exclude'/'maybe' — maybe counts as NOT relevant here).
    """
    n = len(labels_in_order)
    r = sum(1 for x in labels_in_order if x == "include")
    p = (r + 1) / (n + 2)
    remaining = max(n_total - n, 0)
    expected_remaining = remaining * p
    erv = 100.0 * p
    return {
        "reviewed": n,
        "relevant_found": r,
        "p_include_posterior": round(p, 4),
        "expected_relevant_remaining": round(expected_remaining, 2),
        "erv_next_100": round(erv, 2),
        "cal_stop": bool(expected_remaining <= u0),
        "erv_stop": bool(erv < 1.0),
        "u0": u0,
        "note": (
            "advisory only — simplified beta-posterior CAL/ERV; reviewer "
            "decides; stopping never automatic"
        ),
    }


def load_labels_file(path: Path) -> dict[str, str]:
    """Load labels JSONL: {"doi": "...", "label": "include|exclude|maybe"} per line."""
    labels: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"bad labels line {line[:60]!r}: {e}") from e
        key = obj.get("doi") or obj.get("arxiv_id") or obj.get("title")
        label = obj.get("label", "")
        if key and label:
            labels[str(key)] = str(label)
    return labels


# =============================================================================
# Corpus scoring
# =============================================================================
@dataclass
class ScoredPaper:
    paper: PaperRecord
    score: float
    matched_include: list[str]
    matched_exclude: list[str]
    recommendation: str  # 'include' | 'exclude' | 'borderline'


def screen_corpus(
    papers: list[PaperRecord],
    include: list[str],
    exclude: list[str],
    query: str | None = None,
    use_embeddings: bool = False,
    threshold_include: float = 1.0,
    threshold_exclude: float = -1.0,
    model_name: str = "all-MiniLM-L6-v2",
) -> list[ScoredPaper]:
    """Score + classify each paper. Returns list sorted by score descending."""
    scored: list[ScoredPaper] = []
    if use_embeddings and not _EMBED_OK:
        log.warning(
            "sentence-transformers not available; falling back to keyword mode. "
            "Install with: pip install sentence-transformers"
        )
        use_embeddings = False
    if use_embeddings:
        query_vec = _embed_text(query or " ".join(include))
        include_vecs = [_embed_text(s) for s in include]
        exclude_vecs = [_embed_text(s) for s in exclude]
        for p in papers:
            score, mi, me = _score_paper_embeddings(
                p, query_vec, include_vecs, exclude_vecs
            )
            rec = (
                "include"
                if score >= threshold_include
                else "exclude"
                if score <= threshold_exclude
                else "borderline"
            )
            scored.append(ScoredPaper(p, score, mi, me, rec))
    else:
        set_keyword_idf(papers, include, exclude)  # batch IDF (2026-08-15)
        for p in papers:
            score, mi, me = _score_paper_keywords(p, include, exclude)
            rec = (
                "include"
                if score >= threshold_include
                else "exclude"
                if score <= threshold_exclude
                else "borderline"
            )
            scored.append(ScoredPaper(p, score, mi, me, rec))
    scored.sort(key=lambda s: -s.score)
    return scored


# =============================================================================
# PRISMA Phase 2 counts
# =============================================================================
def prisma_phase2_counts(scored: list[ScoredPaper]) -> dict:
    n_total = len(scored)
    n_inc = sum(1 for s in scored if s.recommendation == "include")
    n_exc = sum(1 for s in scored if s.recommendation == "exclude")
    n_bord = sum(1 for s in scored if s.recommendation == "borderline")
    return {
        "n_records_screened": n_total,
        "n_records_excluded": n_exc,
        "n_records_to_eligibility": n_inc + n_bord,  # LLM decides on borderline
        "n_include_recommended": n_inc,
        "n_borderline": n_bord,
        "n_exclude_recommended": n_exc,
    }


# =============================================================================
# Rendered output for LLM
# =============================================================================
def render_screening_report(scored: list[ScoredPaper], counts: dict) -> str:
    """Compact markdown report (~50 tokens/paper). LLM adjudicates borderline."""
    out = ["# PRISMA Phase 2 Screening Report", ""]
    out.append("## Counts")
    out.append(f"- Records screened: **{counts['n_records_screened']}**")
    out.append(f"- Recommended exclude (auto): **{counts['n_exclude_recommended']}**")
    out.append(f"- Recommended include (auto): **{counts['n_include_recommended']}**")
    out.append(f"- **Borderline (LLM adjudicates)**: **{counts['n_borderline']}**")
    out.append(
        f"- Sent to Phase 3 (eligibility): **{counts['n_records_to_eligibility']}**"
    )
    out.append("")
    out.append("## Recommended INCLUDE")
    for s in [x for x in scored if x.recommendation == "include"]:
        out.append(_format_scored(s))
    out.append("\n## BORDERLINE (LLM adjudicates)")
    for s in [x for x in scored if x.recommendation == "borderline"]:
        out.append(_format_scored(s))
    out.append("\n## Recommended EXCLUDE (auto-excluded)")
    for s in [x for x in scored if x.recommendation == "exclude"][:10]:
        out.append(_format_scored(s, brief=True))
    n_exc = counts["n_exclude_recommended"]
    if n_exc > 10:
        out.append(f"\n... and {n_exc - 10} more excluded")
    return "\n".join(out)


def _format_scored(s: ScoredPaper, brief: bool = False) -> str:
    ident = s.paper.doi or s.paper.arxiv_id or s.paper.openalex_id or "?"
    line = f"- **[{s.recommendation.upper()}]** score={s.score:.2f} `{ident}` "
    line += f"**{(s.paper.title or '?')[:80]}**"
    if s.paper.year:
        line += f" ({s.paper.year})"
    if not brief:
        if s.matched_include:
            line += f"\n  - matched: *{', '.join(s.matched_include[:5])}*"
        if s.matched_exclude:
            line += f"\n  - excluded by: *{', '.join(s.matched_exclude[:5])}*"
    return line


# =============================================================================
# CLI
# =============================================================================
def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="screen",
        description="PRISMA Phase 2 screening (title/abstract triage).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("corpus", type=Path, help="input corpus.json from discover.py")
    p.add_argument("--include", default="", help="comma-separated include keywords")
    p.add_argument("--exclude", default="", help="comma-separated exclude keywords")
    p.add_argument("--query", help="original research query (for embedding mode)")
    p.add_argument(
        "--embeddings",
        action="store_true",
        help="use sentence-transformer embeddings (default: keyword)",
    )
    p.add_argument(
        "--al",
        action="store_true",
        help="active-learning prioritization (ELAS-Ultra-style TF-IDF+LinearSVC). "
        "REORDER-ONLY: never excludes; needs --labels seed file",
    )
    p.add_argument(
        "--labels",
        type=Path,
        help='labels JSONL for --al / kappa: {"doi": ..., "label": "include|exclude"} per line',
    )
    p.add_argument(
        "--model",
        default="all-MiniLM-L6-v2",
        help="embedding model name (default all-MiniLM-L6-v2)",
    )
    p.add_argument(
        "--threshold-include",
        type=float,
        default=1.0,
        help="score threshold for include recommendation",
    )
    p.add_argument(
        "--threshold-exclude",
        type=float,
        default=-1.0,
        help="score threshold for exclude recommendation (negative)",
    )
    p.add_argument(
        "-o", "--output", type=Path, default=Path("research_outputs/screened.json")
    )
    p.add_argument(
        "--report", type=Path, default=Path("research_outputs/screening_report.md")
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    level = logging.WARNING - 10 * args.verbose
    logging.basicConfig(
        level=max(level, logging.DEBUG),
        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s",
    )

    include = [s.strip() for s in args.include.split(",") if s.strip()]
    exclude = [s.strip() for s in args.exclude.split(",") if s.strip()]

    papers = load_corpus(args.corpus)
    log.info("Loaded %d papers from %s", len(papers), args.corpus)

    # Active-learning mode: reorder-only prioritization, no keyword scoring needed
    if args.al:
        if not args.labels or not args.labels.exists():
            p.error("--al requires --labels LABELS.jsonl (seed include/exclude labels)")
        al_labels = load_labels_file(args.labels)
        try:
            al = prioritize_active_learning(papers, al_labels)
        except ValueError as e:
            print(f"FATAL: {e}", file=sys.stderr)
            return 2
        payload = {
            "meta": {
                "mode": "active-learning",
                "source_corpus": str(args.corpus),
                "labels_file": str(args.labels),
                "n_seed": al.n_seed,
                "n_unscreened": al.n_unscreened,
                "model": al.model_info,
                "advisory_recall_curve": al.recall_curve,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
            "al_queue": al.queue,
            "papers": [p.to_dict() for p in papers],  # ALL papers — reorder-only
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        report_lines = [
            "# Active-Learning Screening Queue",
            "",
            f"- Seed labels: **{al.n_seed}** | Unscreened: **{al.n_unscreened}**",
            f"- Model: {al.model_info['classifier']}",
            f"- Mode: {al.model_info['mode']}",
            "",
            "## Priority queue (review top-down; P(include) descending)",
        ]
        for item in al.queue[:50]:
            report_lines.append(
                f"{item['al_rank']}. P(include)={item['p_include']:.2f} "
                f"`{item['doi'] or item['paper_id']}` **{(item['title'] or '?')[:80]}**"
            )
        if len(al.queue) > 50:
            report_lines.append(f"... and {len(al.queue) - 50} more in {args.output}")
        report_lines.append(_al_stopping_section(al_labels, len(papers)))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(report_lines))
        print(f"Wrote AL queue ({al.n_unscreened} papers) → {args.output}")
        print(f"Wrote AL report → {args.report}")
        print(
            f"\nAL mode: {al.n_unscreened} unscreened prioritized "
            f"(seed={al.n_seed}). REORDER-ONLY — no auto-exclusion."
        )
        return 0

    if not include and not exclude:
        p.error("must specify --include or --exclude keywords (or use --al --labels)")

    scored = screen_corpus(
        papers,
        include,
        exclude,
        query=args.query,
        use_embeddings=args.embeddings,
        threshold_include=args.threshold_include,
        threshold_exclude=args.threshold_exclude,
    )
    counts = prisma_phase2_counts(scored)

    # Save screened.json
    payload = {
        "meta": {
            "source_corpus": str(args.corpus),
            "include": include,
            "exclude": exclude,
            "embeddings": args.embeddings,
            "model": args.model if args.embeddings else None,
            "counts": counts,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "screened_papers": [
            {
                "paper_id": s.paper.primary_id,
                "doi": s.paper.doi,
                "arxiv_id": s.paper.arxiv_id,
                "title": s.paper.title,
                "year": s.paper.year,
                "score": s.score,
                "recommendation": s.recommendation,
                "matched_include": s.matched_include,
                "matched_exclude": s.matched_exclude,
                "paper": s.paper.to_dict(),
            }
            for s in scored
        ],
        # Also output as filtered corpus (papers key) for verify.py compatibility
        "papers": [s.paper.to_dict() for s in scored if s.recommendation != "exclude"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"Wrote {len(scored)} scored papers → {args.output}")

    # Save screening report
    args.report.write_text(render_screening_report(scored, counts))
    print(f"Wrote screening report → {args.report}")
    print(
        f"\nPRISMA Phase 2: {counts['n_records_screened']} screened → "
        f"{counts['n_include_recommended']} include, "
        f"{counts['n_borderline']} borderline, "
        f"{counts['n_exclude_recommended']} exclude"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
