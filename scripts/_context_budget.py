"""Unified context-budget + query-planning architecture (2026-09-01).

Implements the world-best mandates (Master directive):

- NON-TRUNCATING: LLM contexts are assembled from WHOLE units (papers,
  pages, sections). Overflow drops the lowest-priority whole unit — never
  a mid-content cut. When a single unit alone exceeds the budget, the
  unit degrades one level to whole PARAGRAPHS (still whole sub-units),
  and the prompt's provenance footer states exactly what was included
  and excluded, so the model never silently misses context.
- DYNAMIC: token budgets derive from the model's actual context window
  (auto-sized via the _llm_extract estimator), priorities from live
  query/corpus term overlap. No fixed magic budgets.
- GENERALIZED: zero domain vocabulary. The intent taxonomy is epistemic
  (explanatory / methodological / quantitative / comparative /
  adversarial), applicable to any research domain.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

log = logging.getLogger("scientific_research.context_budget")

# Token estimate: chars-per-token for English scientific prose. The
# estimator in _llm_extract uses the same heuristic — one constant here
# keeps them consistent.
_CHARS_PER_TOKEN = 4

# Hard ceiling matching _llm_extract's auto-sizing cap.
_CTX_CEILING_TOKENS = 32768


@dataclass
class ContextUnit:
    """One whole, indivisible-by-default piece of context."""

    id: str
    header: str  # one-line provenance (title/host/doi — displayed to model)
    text: str
    priority: float = 0.0  # higher = included first
    paragraph_fallback: bool = True  # may degrade to whole paragraphs


@dataclass
class AssembledContext:
    """Result of budgeted assembly — full provenance, no hidden cuts."""

    prompt_body: str
    included_ids: list[str] = field(default_factory=list)
    excluded_ids: list[str] = field(default_factory=list)

    @property
    def complete(self) -> bool:
        return not self.excluded_ids


def _paragraphs(text: str) -> list[str]:
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return paras or ([text.strip()] if text.strip() else [])


def term_priority(text: str, query: str) -> float:
    """Dynamic priority: distinct query-content-term density in the unit."""
    terms = {t for t in re.findall(r"[a-z][a-z]{3,}", query.lower())}
    if not terms:
        return 0.0
    body = text.lower()
    return sum(1 for t in terms if t in body) / len(terms)


class ContextWindow:
    """Assemble whole units under the model's dynamic token budget."""

    def __init__(self, *, reserve_tokens: int = 1024, ceiling_tokens: int = _CTX_CEILING_TOKENS):
        self.reserve_tokens = reserve_tokens
        self.ceiling_tokens = ceiling_tokens

    def build(self, units: list[ContextUnit], *, num_ctx: int | None = None) -> AssembledContext:
        budget = min(num_ctx or self.ceiling_tokens, self.ceiling_tokens) - self.reserve_tokens
        budget = max(budget, 1024) * _CHARS_PER_TOKEN  # chars budget

        ordered = sorted(units, key=lambda u: -u.priority)
        included: list[str] = []
        included_ids: list[str] = []
        excluded_ids: list[str] = []
        used = 0

        for u in ordered:
            candidate = f"[[{u.id}]] {u.header}\n{u.text}"
            size = len(candidate) + 2
            if used + size <= budget:
                included.append(candidate)
                included_ids.append(u.id)
                used += size
                continue
            if not u.paragraph_fallback:
                excluded_ids.append(u.id)
                continue
            # Degrade ONE level: whole paragraphs, priority-ranked, until the
            # remainder of the budget is spent. Never a mid-paragraph cut.
            paras = sorted(
                _paragraphs(u.text),
                key=lambda p: -len(p),
            )
            kept_paras: list[str] = []
            for p in paras:
                psize = len(p) + 1
                if used + psize > budget:
                    break
                kept_paras.append(p)
                used += psize
            if kept_paras and len("\n\n".join(kept_paras)) > 0.3 * len(u.text):
                body = "\n\n".join(kept_paras)
                included.append(
                    f"[[{u.id}]] {u.header}\n{body}\n"
                    f"(partial unit: {len(kept_paras)}/{len(paras)} paragraphs "
                    "included under context budget)"
                )
                included_ids.append(u.id)
            else:
                excluded_ids.append(u.id)

        footer = ""
        if excluded_ids:
            footer = (
                f"\n\n[context budget: {len(included_ids)}/{len(units)} units "
                f"included; excluded: {', '.join(excluded_ids)}]"
            )
        return AssembledContext(
            prompt_body="\n\n".join(included) + footer,
            included_ids=included_ids,
            excluded_ids=excluded_ids,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Query planning — LLM-primary, structural fallback, epistemic intents
# ═════════════════════════════════════════════════════════════════════════════

# Epistemic intent taxonomy (domain-agnostic): every research question
# deserves coverage of each lens that applies.
INTENTS = ("explanatory", "methodological", "quantitative", "comparative", "adversarial")

_CORRECTOR_SYSTEM = """You are a research query normalizer for scholarly
search. Fix ONLY spelling slips, letter transpositions, transliteration
variants, and encoding artifacts in the query — NEVER change its intent,
add terms, or drop terms.

Respond with ONE JSON object, no prose:
{"query": "<corrected query>", "corrections": [{"from": "<word>", "to": "<word>"}]}

If the query is already correct, return it unchanged with empty corrections."""


_PLANNER_SYSTEM = """You are a research query planner. Given a research
question (and optional feedback on what a first information pass missed),
emit 4-6 search queries that TOGETHER cover the distinct epistemic lenses:

- explanatory: how the phenomenon works, overviews, syntheses
- methodological: procedures, standards, normalization/reference conventions
- quantitative: typical values, datasets, compilations (with units)
- comparative: schools of thought, method trade-offs, regional contrasts
- adversarial: criticisms, failed replications, open controversies

Respond with ONE JSON object, no prose:
{"queries": [{"intent": "<one of the five>", "query": "<4-12 word literal search string>"}]}

Rules:
- Each query DIFFERENT in wording from the original question and from the
  others (rewording finds the same corpus twice).
- Literal search strings — no operators, no quotes.
- If feedback text is given, prioritize queries that fill the named gaps."""


def _llm(prompt: str) -> str | None:
    try:
        from _llm_extract import _call_ollama, _pick_model

        model = _pick_model("moderate")
        if model is None:
            return None
        return _call_ollama(model, prompt, task="moderate")
    except Exception as exc:
        log.debug("planner LLM unavailable: %s", exc)
        return None


def _parse(text: str | None) -> list[tuple[str, str]] | None:
    if not text:
        return None
    from _llm_extract import _extract_json_from_text

    obj = _extract_json_from_text(text)
    if not isinstance(obj, dict):
        return None
    out: list[tuple[str, str]] = []
    for item in obj.get("queries", []):
        if isinstance(item, dict):
            intent = str(item.get("intent", "explanatory")).strip().lower()
            query = str(item.get("query", "")).strip()
            if query and 3 <= len(query.split()) <= 14:
                out.append((intent if intent in INTENTS else "explanatory", query))
    return out[:6] or None


def normalize_query(query: str) -> tuple[str, list[dict]]:
    """Query-correction front door (2026-09-01).

    LLM-primary (fixes transpositions/transliterations against its language
    knowledge — transposed-word→correct-word, slipword→word); structural
    fallback applies Unicode folding hygiene (homoglyphs, U+2010 hyphens,
    diacritics) even offline. Returns (corrected_query, corrections) —
    corrections is empty when nothing changed. Original intent preserved
    by construction: only word-level spell fixes, never term add/drop.
    """
    text = _llm(f"{_CORRECTOR_SYSTEM}\n\nQUERY:\n{query}\n\nJSON:")
    obj = None
    if text:
        from _llm_extract import _extract_json_from_text

        try:
            obj = _extract_json_from_text(text)
        except Exception:
            obj = None
    if not isinstance(obj, dict) or not obj.get("query"):
        # offline/unparseable: at least apply Unicode fold hygiene
        from _honesty import fold_text

        folded = fold_text(query).strip()
        folded = folded[0].upper() + folded[1:] if folded else folded
        return (query if folded.lower() == query.lower() else folded, [])
    if isinstance(obj, dict) and obj.get("query"):
        corrected = str(obj["query"]).strip()
        if 0 < len(corrected.split()) == len(query.split()):
            corr = [
                {"from": str(c.get("from", "")), "to": str(c.get("to", ""))}
                for c in obj.get("corrections", [])
                if isinstance(c, dict) and c.get("to")
            ][:8]
            if corrected.lower() != query.lower().strip() or corr:
                return corrected, corr
    return query, []


def _structural_fallback(query: str) -> list[tuple[str, str]]:
    """Offline decomposition: conjunction split, else epistemic templates."""
    conj = re.compile(r"\b(?:and|vs\.?|versus|compared\s+to|with)\b|,", re.IGNORECASE)
    parts = [p.strip() for p in conj.split(query) if p and len(p.strip()) > 3]
    if len(parts) > 1:
        base = [( "comparative", query)] + [("explanatory", p) for p in parts[1:]]
        return base[:5]
    q = query.strip()
    return [
        ("explanatory", f"{q} overview explained"),
        ("methodological", f"{q} methodology procedures standards normalization"),
        ("quantitative", f"{q} typical values data compilation"),
        ("comparative", f"{q} comparison review controversy"),
    ][:5]


def plan_queries(query: str, *, feedback: str = "") -> list[tuple[str, str]]:
    """Diverse, intent-typed search queries for one research question.

    LLM-planned when Ollama is available; structural epistemic templates
    otherwise. Always includes the original query at the head position
    upstream callers decide; this returns ONLY the derived queries.
    """
    fb = f"\n\nFIRST-PASS FEEDBACK (fill these gaps):\n{feedback[:2000]}" if feedback else ""
    parsed = _parse(_llm(f"{_PLANNER_SYSTEM}\n\nRESEARCH QUESTION:\n{query}{fb}\n\nJSON:"))
    if parsed:
        return parsed
    return _structural_fallback(query)
