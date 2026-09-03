#!/usr/bin/env python3
"""Phase 0: research context builder — decompose a query BEFORE searching.

Architecture (2026-08-21): a human researcher understands the question
before typing into a database. The pipeline historically searched FIRST
(expanded query blob) and screened after — which selected garbage for
niche queries (search-engine keyword relevance has no notion of topic,
and sibling-expansion diluted primary subjects, e.g. "pyrite thermometry"
→ 4 generic siblings matching every thermometry paper).

build_research_context() produces a structured context from general
resources (Wikidata/Wikipedia aliases, morphological term analysis,
existing intent/domain classifiers). Every downstream phase consumes it:

    discovery  → per-entity search strategies (no dilution)
    screening  → entity-aware primary rules with sanitized aliases
    synthesis  → dynamic section structure from intent + entities
    audit      → step-by-step quality telemetry

Fail-open contract: network-down degrades to morphology-only context
(same behavior as the pre-context pipeline); it never blocks a run.
"""

from __future__ import annotations

import logging
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

log = logging.getLogger("scientific_research.context")

# Locality description markers from Wikidata/Wikipedia: a query anchored on
# one of these is NICHE — top academic-API keyword search misses its
# literature (regional journals), so hybrid web search is warranted early.
_LOCALITY_DESC = re.compile(
    r"\b(?:town|village|district|city|taluka|tehsil|county|municipality|"
    r"settlement|commune|prefecture|block|census town)\b",
    re.IGNORECASE,
)

_COORD_RE = re.compile(
    r"\b([a-z][a-z\-]{2,})\s+(?:and|vs\.?|versus|compared\s+(?:to|with))\s+"
    r"([a-z][a-z\-]{2,})\b",
    re.IGNORECASE,
)


@dataclass
class Entity:
    """One query concept: term, kind, and sanitized aliases."""

    term: str
    kind: str = "concept"  # locality | method | material | organism | concept
    aliases: list[str] = field(default_factory=list)
    is_primary: bool = False


@dataclass
class ResearchContext:
    """Structured decomposition of the user's query (Phase 0 output)."""

    query: str
    intent: str = "subtopic"  # comparative | survey | verification | ...
    domain: str = ""
    entities: list[Entity] = field(default_factory=list)
    niche: bool = False  # locality-anchored → hybrid web search early
    search_strategies: list[str] = field(default_factory=list)
    screening_terms: list[str] = field(default_factory=list)
    alias_map: dict[str, list[str]] = field(default_factory=dict)
    source_notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "intent": self.intent,
            "domain": self.domain,
            "entities": [
                {"term": e.term, "kind": e.kind, "aliases": e.aliases,
                 "primary": e.is_primary}
                for e in self.entities
            ],
            "niche": self.niche,
            "search_strategies": self.search_strategies,
            "screening_terms": self.screening_terms,
            "alias_map": self.alias_map,
        }


def _classify_entity(term: str, description: str) -> str:
    """Entity kind from its wiki description (general markers, not vocab)."""
    d = (description or "").lower()
    if _LOCALITY_DESC.search(d):
        return "locality"
    if re.search(r"\b(?:mineral|rock|ore|chemical compound|element)\b", d):
        return "material"
    if re.search(
        r"\b(?:method|technique|algorithm|procedure|process|therapy|"
        r"medication|drug|software)\b",
        d,
    ):
        return "method"
    if re.search(r"\b(?:species|genus|family of|plant|bacteria|virus)\b", d):
        return "organism"
    return "concept"


def _entity_description(term: str) -> str:
    """Best-effort wiki description for kind classification. '' offline."""
    try:
        import json
        import urllib.parse
        import urllib.request

        from _semantic_context import _WIKI_API, _WIKI_TIMEOUT

        req = urllib.request.Request(
            _WIKI_API + urllib.parse.quote(term.replace(" ", "_"), safe=""),
            headers={
                "User-Agent": "scientific-research/2.0 (research pipeline)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_WIKI_TIMEOUT) as resp:
            if resp.status != 200:
                return ""
            return (json.loads(resp.read().decode("utf-8", "replace")) or {}).get(
                "description", ""
            )
    except Exception as e:
        log.debug("entity description unavailable for %r: %s", term, e)
        return ""


def build_research_context(query: str, *, use_network: bool = True) -> ResearchContext:
    """Decompose a raw query into the structured research context.

    Layers (fail-open at each):
      1. content terms + coordinated pairs (morphology, offline-safe)
      2. intent (detect_research_type) + domain (existing classifiers)
      3. wiki aliases + entity kinds (network; skipped offline)
      4. search strategies: ORIGINAL query first (expansion dilutes),
         then per-entity + pairwise variants for coordinated queries
    """
    from _honesty import query_content_terms

    ctx = ResearchContext(query=query)

    # ── Layer 1: terms + coordination ──────────────────────────────────
    terms = query_content_terms(query)
    coord_pairs: list[tuple[str, str]] = []
    content_set = set(terms)
    for m in _COORD_RE.finditer(query.lower()):
        w1 = m.group(1)
        w2 = m.group(2)
        def stem(w):
            return w[:-1] if w.endswith("s") and len(w) > 4 else w
        if stem(w1) in content_set and stem(w2) in content_set:
            coord_pairs.append((stem(w1), stem(w2)))

    # ── Layer 2: intent + domain ───────────────────────────────────────
    try:
        from _classifiers import detect_research_type

        ctx.intent = detect_research_type(query)
    except Exception:
        ctx.intent = "subtopic"
    try:
        from _semantic_context import classify_context_semantic

        ctx.domain = classify_context_semantic(query) or ""
    except Exception:
        ctx.domain = ""

    # ── Layer 3: entities with aliases + kinds ─────────────────────────
    coord_flat = {t for pair in coord_pairs for t in pair}
    if use_network:
        # Per-layer fail-open: a missing/broken alias provider must not
        # abort the whole context (morphology entities still stand).
        wiki_aliases = None
        try:
            from _semantic_context import wiki_aliases as _wiki_aliases

            wiki_aliases = _wiki_aliases
        except Exception as e:
            log.debug("wiki alias provider unavailable: %s", e)

        for t in terms:
            e = Entity(term=t, is_primary=(t == terms[0] or t in coord_flat))
            if wiki_aliases is not None and len(t) >= 4:
                desc = _entity_description(t)
                if desc:
                    e.kind = _classify_entity(t, desc)
                    if e.kind == "locality":
                        ctx.niche = True
                e.aliases = wiki_aliases(t, max_aliases=4)
                if e.aliases:
                    ctx.alias_map[t] = e.aliases
            ctx.entities.append(e)
    else:
        for t in terms:
            ctx.entities.append(
                Entity(term=t, is_primary=(t == terms[0] or t in coord_flat))
            )

    # ── Layer 4: search strategies ─────────────────────────────────────
    # Domain context disambiguates web queries: bare "gondpipiri dyke"
    # returns Dick Van Dyke articles on ddgs; adding the domain term
    # ("gondpipiri dyke geochemistry") returns the actual intrusions
    # literature. Works generically — the domain comes from the BGE
    # classifier, not a hardcoded list.
    _DOMAIN_TERMS = {
        "igneous": "geochemistry",
        "metamorphic": "petrology",
        "sedimentary": "sedimentology",
        "economic": "mineralization",
        "structural": "structural geology",
    }
    domain_hint = ""
    shared = [t for t in terms if t not in coord_flat]
    shared_ctx = list(shared)
    if ctx.domain:
        domain_hint = _DOMAIN_TERMS.get(ctx.domain, "")
    if domain_hint and domain_hint not in shared_ctx:
        shared_ctx.append(domain_hint)

    strategies = [query]
    for a, b in coord_pairs[:2]:
        for ent in (a, b):
            strategies.append(f"{ent} {' '.join(shared_ctx)}" if shared_ctx else ent)
            # Entity+domain alone: ddgs relevance is volatile across calls
            # and paper-spelling variants ("Gondpipri" vs "Gondpipiri") miss
            # when multi-term queries fragment — the minimal variant is
            # the most robust probe.
            if domain_hint:
                strategies.append(f"{ent} {domain_hint}")
        strategies.append(f"{a} {b} {' '.join(shared_ctx)}".strip())
    ctx.search_strategies = list(dict.fromkeys(strategies))[:8]

    # ── Layer 5: screening terms (entities + aliases) ──────────────────
    ctx.screening_terms = [e.term for e in ctx.entities if e.is_primary] + [
        e.term for e in ctx.entities if not e.is_primary
    ]

    if ctx.niche:
        ctx.source_notes.append(
            "locality-anchored query — hybrid web search enabled from discovery"
        )
    log.info(
        "Context: intent=%s domain=%s entities=%s niche=%s strategies=%d",
        ctx.intent, ctx.domain or "?",
        [f"{e.term}:{e.kind}" for e in ctx.entities],
        ctx.niche, len(ctx.search_strategies),
    )
    return ctx



def _derive_sections(papers: list, query: str, cited: list) -> list[str]:
    """Section structure derived from the ACTUAL corpus + query structure.

    Not a template: sections reflect what the corpus can support.
    - coordinated entity pairs (X vs Y) → per-entity evidence sections
      + a dedicated comparison section
    - measurement-rich corpora → quantitative synthesis section
    - methodological corpora → method comparison section
    Always: findings, limitations, gaps (only when evidence exists).
    """
    sections: list[str] = []
    q = query.lower()

    # Per-entity sections for coordinated comparisons
    pairs = [(m.group(1), m.group(2)) for m in _COORD_RE.finditer(q)]
    hay = " || ".join(
        f"{(p.get('title') or '')} {(p.get('abstract') or '')}".lower() for p in papers
    )
    for a, b in pairs[:2]:
        if a in hay and b in hay:
            sections += [f"Evidence: {a}", f"Evidence: {b}", f"Comparison: {a} vs {b}"]
        elif a in hay:
            sections += [f"Evidence: {a}"]
    if pairs:
        sections.append("Comparative synthesis")

    # Quantitative synthesis when measurements exist
    n_meas = sum(
        1 for p in papers for m in (p.get("measurements") or []) if isinstance(m, dict)
    )
    if n_meas >= 3:
        sections.append("Quantitative synthesis")

    # Method comparison when >1 distinct method themes
    themes = {p.get("study_type") or "" for p in papers} - {""}
    if len(themes) >= 2:
        sections.append("Methodological landscape")

    sections += ["Findings", "Limitations", "Research gaps and future directions"]
    return list(dict.fromkeys(sections))


if __name__ == "__main__":
    print(f"OK {sys.argv[0]}: research context builder ready")
