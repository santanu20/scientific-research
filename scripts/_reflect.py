"""Agentic reflection layer — the research loop's critic.

Implements the reflect half of search→read→critique→re-search (the 2026
deep-research loop): after a brief is synthesized, an LLM critic examines
coverage (brief + contradictions + corpus stats), and either certifies it
sufficient or emits targeted follow-up queries for one more bounded
discovery round. Also generates adaptive PaperQA2 questions in place of
the fixed template lenses.

Rides the battle-tested Ollama stack from ``_llm_extract`` (circuit
breaker, model picker, JSON-from-text extractor) — no new deps, no new
failure modes. All LLM I/O goes through ``_call_ollama`` so the
SCIENTIFIC_RESEARCH_LLM_CALLBACK agent hook applies here too.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger("scientific_research.reflect")

# Context assembly goes through _context_budget.ContextWindow (2026-09-01):
# whole-unit, model-sized budgets, provenance footer — no char truncation.

_CRITIC_SYSTEM = """You are a rigorous research-coverage critic for the geosciences.
You are given: a synthesized research brief, an advisory contradiction report,
and corpus statistics. Judge whether the corpus + brief SUFFICIENTLY answer
the original research question.

Respond with ONE JSON object, no prose:
{
  "verdict": "sufficient" | "insufficient",
  "gaps": ["<specific missing aspect or unanswered sub-question>", ...],
  "follow_up_queries": ["<web/scholar search query, 4-10 words>", ...],
  "reasoning": "<one paragraph justification citing which brief sections are thin>"
}

Rules:
- "insufficient" ONLY for gaps that NEW PAPERS could plausibly fill — not for
  gaps requiring new lab work, proprietary data, or your own uncertainty.
- follow_up_queries: at most 4, each a literal search string (no operators),
  each DIFFERENT from the original query (rewording the same query finds the
  same corpus).
- contradictions listed in the report are known — do not count them as gaps
  unless follow-up papers could resolve them.
- If the brief already flags a question the corpus cannot answer AND the
  corpus is broad, prefer "sufficient"."""

_QUESTIONS_SYSTEM = """You are a research-synthesis question designer for the geosciences.
Given a research topic and paper abstract excerpts, design 4-6 focused
questions a citation-grounded QA system will answer from the corpus.

Respond with ONE JSON object, no prose:
{"questions": ["<question>", ...]}

Rules:
- Each question must be answerable from paper TEXT (methods, values,
  interpretations, disagreements), not from world knowledge.
- Include at least one quantitative question (values + units + context) and
- at least one cross-paper disagreement/consensus question.
- Include at least one METHODOLOGY-CONVENTIONS question: how data in this
  literature are processed, standardized, and presented (reference
  materials, normalization schemes, reporting conventions — e.g., for REE
  work: chondrite normalization; for geochronology: decay constants).
- Name the topic explicitly in each question so retrieval cannot drift to
  neighboring subjects.
- Do NOT ask anything the abstracts clearly cannot support."""


def _llm(prompt: str, *, task: str = "moderate") -> str | None:
    """One LLM call through the canonical client; None on any failure."""
    from _llm_extract import _call_ollama, _pick_model

    model = _pick_model(task)
    if model is None:
        return None
    try:
        return _call_ollama(model, prompt, task=task)
    except Exception as exc:  # circuit open / network — caller decides degrade
        log.warning("reflect LLM call failed (%s): %s", type(exc).__name__, exc)
        return None


def _parse_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    from _llm_extract import _extract_json_from_text

    obj = _extract_json_from_text(text)
    return obj if isinstance(obj, dict) else None


def critique_coverage(
    query: str,
    brief_text: str,
    contradiction_text: str,
    corpus_stats: dict[str, Any],
) -> dict[str, Any] | None:
    """Critique brief coverage. Returns parsed critique dict or None.

    Contract (None = critic unavailable — caller skips the round LOUDLY):
      {verdict: sufficient|insufficient, gaps: [str],
       follow_up_queries: [str], reasoning: str}
    """
    from _context_budget import AssembledContext, ContextUnit, ContextWindow

    brief_ctx: AssembledContext = ContextWindow(reserve_tokens=4096).build(
        [ContextUnit(id="brief", header="RESEARCH BRIEF", text=brief_text, priority=1.0)]
    )
    prompt = (
        f"{_CRITIC_SYSTEM}\n\n"
        f"ORIGINAL RESEARCH QUESTION:\n{query}\n\n"
        f"CORPUS STATISTICS:\n{corpus_stats}\n\n"
        f"CONTRADICTION REPORT (advisory, known):\n"
        f"{contradiction_text or '(no numeric contradictions detected)'}\n\n"
        f"{brief_ctx.prompt_body}\n\n"
        "JSON verdict:"
    )
    critique = _parse_json(_llm(prompt, task="long"))
    if critique is None:
        log.warning("reflect: critic produced no parseable JSON")
        return None
    verdict = critique.get("verdict") or critique.get("status")
    # Live qwen3.5 sometimes emits {"status": "incomplete"} instead of verdict
    verdict = {
        "sufficient": "sufficient",
        "complete": "sufficient",
        "insufficient": "insufficient",
        "incomplete": "insufficient",
    }.get(str(verdict).strip().lower())
    if verdict is None:
        log.warning("reflect: invalid verdict %r — no re-search (fail-safe)", critique.get("verdict"))
        verdict = "sufficient"  # unparseable verdict = no re-search
    critique["verdict"] = verdict
    queries = critique.get("follow_up_queries") or []
    critique["follow_up_queries"] = [
        str(q).strip() for q in queries if str(q).strip() and str(q).strip().lower() != query.lower()
    ][:4]
    critique.setdefault("gaps", [])
    critique.setdefault("reasoning", "")
    return critique


def generate_adaptive_questions(topic: str, abstracts_text: str) -> list[str] | None:
    """Adaptive PaperQA2 questions; None/[] = use template lenses instead."""
    from _context_budget import ContextUnit, ContextWindow

    units = [
        ContextUnit(id=f"abs{i + 1}", header=f"ABSTRACT {i + 1}", text=chunk, priority=1.0)
        for i, chunk in enumerate(c for c in abstracts_text.split("\n\n") if c.strip())
    ]
    body = ContextWindow(reserve_tokens=2048).build(units).prompt_body if units else ""
    prompt = (
        f"{_QUESTIONS_SYSTEM}\n\n"
        f"RESEARCH TOPIC: {topic}\n\n"
        f"PAPER ABSTRACTS:\n{body}\n\n"
        "JSON questions:"
    )
    parsed = _parse_json(_llm(prompt, task="moderate"))
    if not parsed:
        return None
    questions = [str(q).strip() for q in parsed.get("questions", []) if str(q).strip()]
    return questions[:6] or None
