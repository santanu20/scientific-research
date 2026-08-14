"""Evaluate screening precision/recall against a human-labeled corpus.

Run after manually labeling papers in a corpus.json or extracted.json.
Produces precision, recall, F1, and confusion matrix per stage.

Labeling format (JSONL, one line per paper):
    {"doi": "10.xxx", "label": "include"}      # paper IS relevant to query
    {"doi": "10.yyy", "label": "exclude"}      # paper is NOT relevant
    {"doi": "10.zzz", "label": "maybe"}        # ambiguous, excluded from metrics

Papers in corpus but not in labels file are excluded from metrics
(reported as "unlabeled").

Usage:
    uv run python scripts/eval_screening.py \\
        research_outputs/corpus.json \\
        --query "amphibole thermometery" \\
        --labels research_outputs/labels.jsonl

Output (stdout):
    === Screening evaluation ===
    Query: amphibole thermometery
    Corpus: 79 papers (10 labeled include, 12 labeled exclude, 57 unlabeled)

    Confusion matrix:
                    screen=include    screen=exclude
    label=include        9 (TP)            1 (FN)
    label=exclude        2 (FP)           10 (TN)

    Precision: 9/11 = 0.818
    Recall:    9/10 = 0.900
    F1:        0.857
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
import sys
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger("eval_screening")


@dataclass(frozen=True, slots=True)
class _PaperAdapter:
    """Adapt dict paper record to the PaperRecord-like interface screen_paper expects."""

    title: str
    abstract: str
    doi: str
    primary_id: str
    raw_metadata: dict

    @classmethod
    def from_dict(cls, d: dict) -> _PaperAdapter:
        return cls(
            title=d.get("title", "") or "",
            abstract=d.get("abstract", "") or "",
            doi=d.get("doi", "") or "",
            primary_id=d.get("paper_id", "") or d.get("doi", "") or "",
            raw_metadata=d.get("raw_metadata", {}) or {},
        )


def _load_papers(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "papers" in data:
        return data["papers"]
    if "extractions" in data:
        return data["extractions"]
    raise ValueError(f"unknown format: {path} (expected corpus.json or extracted.json)")


def _load_labels(path: Path) -> dict[str, str]:
    """Load JSONL labels. Returns {doi: 'include'|'exclude'|'maybe'}."""
    labels: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as e:
            log.warning("Skipping malformed label line: %r (%s)", line, e)
            continue
        doi = entry.get("doi", "").lower().strip()
        label = entry.get("label", "").lower().strip()
        if doi and label in {"include", "exclude", "maybe"}:
            labels[doi] = label
    return labels


def evaluate(
    papers: list[dict],
    labels: dict[str, str],
    query: str,
    ensemble: bool = False,
    use_llm: bool = False,
) -> dict:
    """Run screen_paper over labeled papers, return metrics dict.

    use_llm=True: single-judge LLM screening; ensemble=True: majority vote
    across distinct available models (implies use_llm). Both need Ollama.
    Default (both False) = content-type filter only — same as historical
    behavior.
    """
    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    from screen_llm import screen_paper

    tp = fp = fn = tn = 0
    unlabeled = 0
    maybe = 0
    per_stage: dict[str, list[str]] = {"fp_dois": [], "fn_dois": []}

    for p in papers:
        doi = (p.get("doi") or "").lower().strip()
        if not doi or doi not in labels:
            unlabeled += 1
            continue
        label = labels[doi]
        if label == "maybe":
            maybe += 1
            continue

        adapter = _PaperAdapter.from_dict(p)
        decision, stage, _reason = screen_paper(
            adapter, query, use_llm=(use_llm or ensemble), ensemble=ensemble
        )

        screen_include = decision
        true_include = label == "include"

        if screen_include and true_include:
            tp += 1
        elif screen_include and not true_include:
            fp += 1
            per_stage["fp_dois"].append(doi)
        elif not screen_include and true_include:
            fn += 1
            per_stage["fn_dois"].append(f"{doi} (stage={stage})")
        else:
            tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    # Cohen's κ (machine vs human rater) — MECIR dual-screen agreement evidence
    kappa = cohen_kappa(tp, fp, fn, tn)

    return {
        "query": query,
        "total_papers": len(papers),
        "labeled_include": tp + fn,
        "labeled_exclude": fp + tn,
        "labeled_maybe": maybe,
        "unlabeled": unlabeled,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cohen_kappa": kappa,
        "fp_dois": per_stage["fp_dois"],
        "fn_dois": per_stage["fn_dois"],
    }


def cohen_kappa(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Cohen's κ from a 2×2 confusion matrix.

    κ = (po − pe) / (1 − pe); po = observed agreement,
    pe = expected-by-chance agreement from marginal totals.
    Returns None when degenerate (n=0 or pe=1).
    """
    n = tp + fp + fn + tn
    if n == 0:
        return None
    po = (tp + tn) / n
    pe = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n)
    if pe == 1.0:
        return None
    return (po - pe) / (1.0 - pe)


def _print_report(m: dict) -> None:
    print()
    print("=== Screening evaluation ===")
    print(f"Query: {m['query']}")
    print(
        f"Corpus: {m['total_papers']} papers "
        f"({m['labeled_include']} include, {m['labeled_exclude']} exclude, "
        f"{m['labeled_maybe']} maybe, {m['unlabeled']} unlabeled)"
    )
    print()
    print("Confusion matrix:")
    print("                screen=include    screen=exclude")
    print(f"label=include      {m['tp']:3d} (TP)            {m['fn']:3d} (FN)")
    print(f"label=exclude      {m['fp']:3d} (FP)            {m['tn']:3d} (TN)")
    print()
    print(f"Precision: {m['tp']}/{m['tp'] + m['fp']} = {m['precision']:.3f}")
    print(f"Recall:    {m['tp']}/{m['tp'] + m['fn']} = {m['recall']:.3f}")
    print(f"F1:                                      {m['f1']:.3f}")
    if m.get("cohen_kappa") is not None:
        print(f"Cohen's κ (machine vs human):            {m['cohen_kappa']:.3f}")
    if m["fp_dois"]:
        print(f"\nFalse positives ({len(m['fp_dois'])}):")
        for doi in m["fp_dois"][:10]:
            print(f"  - {doi}")
    if m["fn_dois"]:
        print(f"\nFalse negatives ({len(m['fn_dois'])}):")
        for doi in m["fn_dois"][:10]:
            print(f"  - {doi}")


def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    p = argparse.ArgumentParser(
        prog="eval_screening",
        description="Evaluate screening precision/recall against labeled corpus.",
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument("corpus", type=Path, help="corpus.json or extracted.json")
    p.add_argument("--query", required=True, help="the research query")
    p.add_argument(
        "--labels",
        required=True,
        type=Path,
        help="JSONL file with {doi, label: include|exclude|maybe} per line",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="single-judge LLM screening (needs Ollama)",
    )
    p.add_argument(
        "--ensemble",
        action="store_true",
        help="majority-vote ensemble screening across distinct Ollama models "
        "(Sanghera 2025); implies --use-llm",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="write JSON metrics to this path (default: stdout only)",
    )
    p.add_argument("-v", "--verbose", action="count", default=0)
    args = p.parse_args()
    logging.basicConfig(
        level=max(logging.WARNING - 10 * args.verbose, logging.DEBUG),
        format="%(levelname)-5s %(name)s: %(message)s",
    )

    papers = _load_papers(args.corpus)
    labels = _load_labels(args.labels)
    metrics = evaluate(papers, labels, args.query, ensemble=args.ensemble, use_llm=args.use_llm)
    _print_report(metrics)

    if args.output:
        args.output.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False, default=str),
            encoding="utf-8",
        )
        print(f"\nMetrics written to {args.output}")

    # Exit code: 0 if F1 >= 0.85 (target), 1 otherwise
    return 0 if metrics["f1"] >= 0.85 else 2


if __name__ == "__main__":
    sys.exit(main())
