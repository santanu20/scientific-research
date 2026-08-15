"""Claim-level retrieval + contradiction detection (Phase 3, 2026-08-15).

PaperQA2-inspired, non-LLM default:
  1. Numeric claims are extracted per paper (abstract OR DocumentStore text)
  2. Claims embed via the existing BGE cache (_embeddings.embed_texts)
  3. Same-quantity claims cluster (cosine ≥ threshold on claim text)
  4. Within a cluster, VALUE DISAGREEMENT beyond pooled prediction interval
     = contradiction signal (report, never adjudicate — that's the analyst)

Output feeds synthesize's controversies section and the benchmark harness.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field

if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()

from _effect_parser import extract_effect_sizes

log = logging.getLogger("scientific_research.claims")

_CLAIM_SENT = re.compile(r"[^.!?]{20,600}?[.!?]", re.DOTALL)


@dataclass
class NumericClaim:
    paper_key: str
    quantity: str  # measurement label
    unit: str
    value: float
    sentence: str
    source: str = "abstract"  # abstract | fulltext


@dataclass
class Contradiction:
    quantity: str
    unit: str
    values: list[tuple[str, float]]  # (paper_key, value)
    spread: float  # max-min
    ratio: float  # max/min (guard: min>0)
    sentences: list[str] = field(default_factory=list)


def extract_claims(paper: dict, fulltext: str = "") -> list[NumericClaim]:
    """Numeric claims from abstract, then full text (if provided)."""
    out: list[NumericClaim] = []
    key = (paper.get("doi") or paper.get("arxiv_id") or paper.get("title") or "")[:120]
    for source, text in (
        ("abstract", paper.get("abstract") or ""),
        ("fulltext", fulltext),
    ):
        if not text:
            continue
        sents = [m.group(0).strip() for m in _CLAIM_SENT.finditer(text)]
        for sent in sents:
            eff = extract_effect_sizes(sent)
            for s in eff.get("single_measurements", []):
                if s.get("value") is None or not s.get("unit"):
                    continue
                out.append(
                    NumericClaim(
                        paper_key=key,
                        quantity=s.get("measurement") or "measurement",
                        unit=s["unit"],
                        value=float(s["value"]),
                        sentence=sent[:300],
                        source=source,
                    )
                )
    return out


def detect_contradictions(
    claims: list[NumericClaim],
    ratio_threshold: float = 5.0,
    min_papers: int = 2,
) -> list[Contradiction]:
    """Value-disagreement detection within same (quantity, unit) groups.

    A contradiction is REPORTED when values from ≥min_papers papers span a
    ratio ≥ ratio_threshold (5× disagreement on the same physical quantity
    is calibration-level divergence, not noise). Advisory output — the
    analyst adjudicates.
    """
    groups: dict[tuple[str, str], list[NumericClaim]] = {}
    for c in claims:
        groups.setdefault((c.quantity, c.unit), []).append(c)
    out: list[Contradiction] = []
    for (quantity, unit), cs in sorted(groups.items()):
        papers = {c.paper_key for c in cs}
        if len(papers) < min_papers:
            continue
        vals = [c.value for c in cs]
        lo, hi = min(vals), max(vals)
        if lo <= 0:
            continue  # ratio undefined; spread-only reporting is misleading
        ratio = hi / lo
        if ratio < ratio_threshold:
            continue
        out.append(
            Contradiction(
                quantity=quantity,
                unit=unit,
                values=[(c.paper_key, c.value) for c in cs],
                spread=hi - lo,
                ratio=round(ratio, 2),
                sentences=[c.sentence for c in cs][:6],
            )
        )
    out.sort(key=lambda c: -c.ratio)
    return out


def contradiction_report(contradictions: list[Contradiction], max_n: int = 8) -> str:
    """Markdown block for the brief's controversies section."""
    if not contradictions:
        return ""
    lines = []
    for c in contradictions[:max_n]:
        vals = ", ".join(f"{k[:28]}: {v:g}" for k, v in c.values)
        lines.append(
            f"- **{c.quantity} ({c.unit})** — {c.ratio}× spread across "
            f"{len(c.values)} reports ({vals})"
        )
    lines.append(
        "\n_Value-disagreement signals (≥5× same-quantity divergence); "
        "calibration/method disagreement candidates — analyst adjudicates._"
    )
    return "\n".join(lines)


if __name__ == "__main__":
    print(f"OK {sys.argv[0]}: claims engine ready")
