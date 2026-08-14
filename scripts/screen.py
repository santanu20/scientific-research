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

# --- Self-contained skill venv bootstrap (mirrors pdf-ocr/web-search pattern) ---
import os as _bs_os, sys as _bs_sys

_SKILL_VENV = _bs_os.path.expanduser(
    "~/.config/opencode/skills/scientific-research/.venv"
)
_SKILL_VENV_PY = _bs_os.path.join(_SKILL_VENV, "bin", "python")
_REQ_IMPORTS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "sklearn",
    "httpx",
)
_REQ_INSTALLS = (
    "habanero",
    "pyalex",
    "semanticscholar",
    "arxiv",
    "numpy",
    "scipy",
    "scikit-learn",
    "httpx",
    "pytest",
    "ruff",
)
if __name__ == "__main__" and not _bs_os.environ.get(
    "SCIENTIFIC_RESEARCH_NO_SKILL_VENV"
):
    if not _bs_os.path.exists(_SKILL_VENV_PY) and not _bs_os.environ.get(
        "SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"
    ):
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(
                "Bootstrapping scientific-research skill venv (one-time setup)...\n"
            )
            _bs_sp.run(
                ["uv", "venv", _SKILL_VENV, "--python", "3.13"],
                check=True,
                capture_output=True,
            )
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True,
                capture_output=True,
            )
            _bs_sys.stderr.write("scientific-research skill venv ready.\n")
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(
                f"Failed to auto-bootstrap: {_bs_ex}\nManual: uv venv {_SKILL_VENV} --python 3.13 && uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
            )
            _bs_sys.exit(2)
    if _bs_os.path.exists(_SKILL_VENV_PY) and _bs_os.path.normpath(
        _bs_sys.prefix
    ) != _bs_os.path.normpath(_SKILL_VENV):
        _bs_os.environ["SCIENTIFIC_RESEARCH_NO_SKILL_VENV"] = "1"
        _bs_os.execv(
            _SKILL_VENV_PY,
            [_SKILL_VENV_PY, _bs_os.path.abspath(__file__)] + _bs_sys.argv[1:],
        )
    _missing = []
    for _m in _REQ_IMPORTS:
        try:
            __import__(_m)
        except ImportError:
            _missing.append(_m)
    if _missing and not _bs_os.environ.get("SCIENTIFIC_RESEARCH_NO_BOOTSTRAP"):
        # stale venv: auto-install missing deps once, then re-check
        import subprocess as _bs_sp

        try:
            _bs_sys.stderr.write(f"Installing missing deps: {', '.join(_missing)}\n")
            _bs_sp.run(
                ["uv", "pip", "install", "--python", _SKILL_VENV_PY, *_REQ_INSTALLS],
                check=True, capture_output=True,
            )
            _missing = []
            for _m in _REQ_IMPORTS:
                try:
                    __import__(_m)
                except ImportError:
                    _missing.append(_m)
        except (_bs_sp.CalledProcessError, FileNotFoundError) as _bs_ex:
            _bs_sys.stderr.write(f"Auto-install failed: {_bs_ex}\n")
    if _missing:
        _bs_sys.stderr.write(
            f"FATAL: missing required deps: {', '.join(_missing)}\nInstall: uv pip install --python {_SKILL_VENV_PY} {' '.join(_REQ_INSTALLS)}\n"
        )
        _bs_sys.exit(2)
# --- End bootstrap ---


import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord, load_corpus  # noqa: E402

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
    """Return (score, matched_include, matched_exclude).
    Score = (#include matched) − 2*(#exclude matched)."""
    text = ((paper.title or "") + " " + (paper.abstract or "")).lower()
    if not text.strip():
        return (0.0, [], [])
    matched_inc = [k for k in include if k.lower() in text]
    matched_exc = [k for k in exclude if k.lower() in text]
    score = float(len(matched_inc)) - 2.0 * float(len(matched_exc))
    return (score, matched_inc, matched_exc)


# =============================================================================
# Embedding mode (semantic similarity, optional)
# =============================================================================
def _embed_text(text: str) -> list[float]:
    """Embed single text using BGE (shared singleton from _embeddings.py)."""
    import numpy as _np
    from _embeddings import embed_texts

    emb = embed_texts([text])
    if emb is not None and emb.shape[0] > 0:
        vec = emb[0]
        norm = _np.linalg.norm(vec) + 1e-10
        return (vec / norm).tolist()
    return []


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
    text = ((paper.title or "") + ". " + (paper.abstract or ""))[:5000]
    if not text.strip():
        return (0.0, [], [])
    paper_vec = _embed_text(text)
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
        return ((p.title or "") + " " + (p.abstract or ""))[:8000]

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
