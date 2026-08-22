"""Deterministic honesty gates for the research pipeline.

Pure stdlib functions — no LLM, no network. These gates exist because a
pipeline that synthesizes confident prose from an off-topic corpus is worse
than one that refuses: hallucinated regional framing ("dykes of Gadchiroli")
built from papers about other regions is scientific fabrication (§5 H7).

Gates:
    extract_query_entities   — distinctive tokens in the user query
    corpus_coverage          — per-entity paper support in title+abstract
    sufficiency_verdict      — refuse synthesis when corpus cannot answer query
    grounding_audit          — entities asserted in brief but absent from corpus
    render_insufficient_brief — honest "no on-target literature" report
"""

from __future__ import annotations

import re

# Generic research/geology vocabulary — never treated as a distinctive entity.
# A query "dykes age exposed in gadchiroli" must yield ["gadchiroli"], not
# ["dykes", "age", "exposed", "gadchiroli"].
_GENERIC_TERMS = frozenset(
    {
        "age",
        "ages",
        "aged",
        "dating",
        "dated",
        "exposed",
        "exposure",
        "comparison",
        "comparative",
        "compare",
        "compared",
        "study",
        "studies",
        "analysis",
        "review",
        "survey",
        "overview",
        "update",
        "progress",
        "region",
        "regions",
        "regional",
        "area",
        "areas",
        "evidence",
        "new",
        "insights",
        "implications",
        "evolution",
        "history",
        "case",
        "part",
        "geology",
        "geological",
        "question",
        "questions",
        "problem",
        "problems",
        "note",
        "notes",
        "data",
        "results",
        "system",
        "systems",
        "district",
        "districts",
        "province",
        "approach",
        "based",
        "using",
        "from",
        "with",
        "and",
        "the",
        "for",
        "into",
        "their",
    }
)

_TOKEN_RE = re.compile(r"[a-z][a-z\-]{3,}")
# Minimum prefix length for fuzzy locality matching (handles spelling
# variants like gondpipiri/gondpipri). ponytail: prefix match, real
# gazetteer if false positives ever appear.
_FUZZY_PREFIX_LEN = 6


def extract_query_entities(query: str) -> list[str]:
    """Distinctive tokens from a query: proper-noun-ish, non-generic.

    Returns lowercase entities, longest first. Tokens in _GENERIC_TERMS and
    pure numbers are dropped; everything else len>=4 is kept.
    """
    tokens = _TOKEN_RE.findall(query.lower())
    seen: set[str] = set()
    entities: list[str] = []
    for tok in sorted(set(tokens), key=len, reverse=True):
        if tok in _GENERIC_TERMS or tok in seen:
            continue
        seen.add(tok)
        entities.append(tok)
    return entities


def _text_matches(text: str, entity: str) -> bool:
    """Entity present in text: exact substring, or stable-prefix for long
    locality names (spelling variants)."""
    if entity in text:
        return True
    if len(entity) >= 8:
        return entity[:_FUZZY_PREFIX_LEN] in text
    return False


def _record_text(p) -> str:
    """title+abstract lowercased for dicts OR record objects."""
    if isinstance(p, dict):
        title = str(p.get("title") or "")
        abstract = str(p.get("abstract") or "")
    else:
        title = str(getattr(p, "title", "") or "")
        abstract = str(getattr(p, "abstract", "") or "")
    return f"{title} {abstract}".lower()


def corpus_coverage(papers: list, entities: list[str]) -> dict[str, int]:
    """Count papers whose title+abstract mention each entity."""
    if not entities:
        return {}
    texts = [_record_text(p) for p in papers]
    coverage: dict[str, int] = {}
    for ent in entities:
        coverage[ent] = sum(1 for t in texts if t and _text_matches(t, ent))
    return coverage


def sufficiency_verdict(
    entities: list[str],
    coverage: dict[str, int],
    n_discovered: int,
    n_excluded: int,
    n_final: int | None = None,
) -> tuple[bool, str]:
    """True (insufficient) when the screened corpus cannot address the query.

    Fires ONLY when ALL hold — broad surveys without distinctive entities
    always pass:
      - query yielded at least one entity
      - ZERO final-corpus papers mention ANY entity
      - screening rejected the overwhelming majority (>80%) of what was
        discovered, i.e. discovery itself returned mostly off-topic material
    """
    if not entities or n_discovered <= 0:
        return False, ""
    if any(coverage.get(e, 0) > 0 for e in entities):
        return False, ""
    # Zero final corpus while candidates existed: nothing supports the
    # query. The >80% path below misses this (10 rejected of 15+10 = 40%)
    # because its ratio mixes discovered and excluded counts.
    if n_final is not None and n_final == 0 and n_discovered > 0:
        return (
            True,
            f"0/{len(entities)} query entities supported by corpus; "
            f"empty corpus after screening ({n_discovered} discovered, "
            f"{n_excluded} rejected)",
        )
    total = n_discovered + n_excluded
    if total > 0 and n_excluded / total > 0.8:
        return (
            True,
            f"0/{len(entities)} query entities supported by corpus; "
            f"{n_excluded}/{total} discovered papers rejected by screening",
        )
    return False, ""


def grounding_audit(
    brief_text: str,
    entities: list[str],
    coverage: dict[str, int],
) -> list[str]:
    """Entities the brief asserts in prose but the corpus never supports.

    The H1 title line is excluded (it always contains the raw query).
    Returns the unsupported entities actually found in the body — these are
    exactly the claims an LLM fabricated from the query string alone.
    """
    lines = brief_text.splitlines()
    body = "\n".join(lines[1:]) if lines and lines[0].startswith("# ") else brief_text
    body = body.lower()
    return [e for e in entities if coverage.get(e, 0) == 0 and _text_matches(body, e)]


def render_grounding_warning(unsupported: list[str], coverage: dict[str, int]) -> str:
    """Markdown warning section appended when the brief asserts unsupported entities."""
    rows = "\n".join(f"- `{e}`: 0 supporting papers in corpus" for e in unsupported)
    return (
        "\n\n---\n\n"
        "## Corpus Coverage Warning\n\n"
        "The synthesis above mentions the following query terms that **no paper "
        "in the verified corpus actually discusses**. These statements are NOT "
        "supported by the retrieved literature and were likely inferred from the "
        "query wording alone — treat them as unverified:\n\n"
        f"{rows}\n"
    )


def render_insufficient_brief(
    query: str,
    entities: list[str],
    coverage: dict[str, int],
    screening_log: list[dict],
    n_discovered: int,
) -> str:
    """Honest 'no on-target literature found' brief.

    Renders instead of synthesis when sufficiency_verdict fires. Lists what
    was discovered and why it was rejected, so the run is auditable rather
    than silently empty — and NEVER fabricates comparative claims.
    """
    lines = [
        f"# Research Brief: {query}",
        "",
        "**Status: INSUFFICIENT EVIDENCE — synthesis refused.**",
        "",
        f"The pipeline discovered {n_discovered} candidate papers, but after "
        "relevance screening **none of the verified corpus addresses the query's "
        "key terms**:",
        "",
    ]
    if entities:
        lines.append("| Query term | Papers in final corpus |")
        lines.append("|---|---|")
        for e in entities:
            lines.append(f"| `{e}` | {coverage.get(e, 0)} |")
        lines.append("")
    lines.append("No honest comparison or synthesis is possible from this corpus.")
    lines.append("Generating one anyway would fabricate findings (hallucination guard).")
    lines.append("")

    if screening_log:
        lines.append(f"## Why {len(screening_log)} candidates were rejected (sample)")
        lines.append("")
        lines.append("| Title | Screening reason |")
        lines.append("|---|---|")
        for entry in screening_log[:15]:
            title = (entry.get("title") or "")[:70]
            reason = (entry.get("reason") or "")[:80]
            lines.append(f"| {title} | {reason} |")
        lines.append("")

    lines.append("## Suggested next steps")
    lines.append("")
    lines.append("- Enable web-search sources for niche/regional topics")
    lines.append(
        "- Broaden the query (e.g. use the geological province name, "
        "not only district/locality names)"
    )
    lines.append(
        "- Check that the target literature exists in indexed sources "
        "(OpenAlex/Crossref/EarthArXiv)"
    )
    lines.append("")
    return "\n".join(lines)


def _stem(w: str) -> str:
    return w[:-1] if w.endswith("s") and len(w) > 4 else w


# Domain-AGNOSTIC term matching. No hand-coded synonym lists (a geo jargon
# bag matches "pyrite thermometry" but silently fails every medical/CS/
# physics query, and grows monotonically per bug report). Three general
# mechanisms cover cross-domain morphology:
#   1. suffix-class stemming: thermometer/thermometry/thermometric share
#      the stem "thermometr"; classification/classifier share "classif";
#      diagnosis/diagnostic share "diagnos". Suffix list is general
#      English science morphology, not vocabulary.
#   2. compound-root containment: "thermobarometry" and "thermometry"
#      share the >=5-char alpha root "thermo"; "immunohistochemistry"
#      and "chemistry" share "chemi". Works for any compound coinage.
#   3. exact substring (after 1+2 fail).
_SUFFIX_CLASSES: list[tuple[str, ...]] = [
    ("metry", "meter", "metric", "metrics", "metries"),
    ("logy", "logical", "logies", "logic"),
    ("graphy", "graphic", "graphs"),
    ("tion", "tional", "tions"),
    ("sion", "sional", "sions"),
    ("ment", "mental", "ments"),
    ("ance", "ant", "ancy"),
    ("ence", "ent", "ency"),
    ("ity", "ities"),
    ("ic", "ical", "ics"),
    ("al", "ally"),
    ("er", "ers", "ery"),
    ("or", "ors", "ory"),
    ("ive", "ively", "ivity"),
    ("ous", "ously", "osity"),
]


def _root(word: str, min_len: int = 5) -> str:
    """Prefix of `word` after stripping the LONGEST matching suffix once.
    Unstrippable/short words pass through unchanged."""
    w = word.lower()
    best_suffix = ""
    for sufs in _SUFFIX_CLASSES:
        for s in sufs:
            if w.endswith(s) and len(w) - len(s) >= min_len:
                if len(s) > len(best_suffix):
                    best_suffix = s
    return w[: len(w) - len(best_suffix)] if best_suffix else w


def _alpha_roots(word: str, min_len: int = 5) -> set[str]:
    """All >=min_len contiguous alpha runs of a token ('thermobarometry' →
    {'thermobarom', prefixes...}; caller does containment either way)."""
    w = re.sub(r"[^a-z]", "", word.lower())
    return {w} if len(w) >= min_len else set()


def _term_matches(term: str, hay: str, aliases: list[str] | None = None) -> bool:
    """General morphological match of one query term in paper text.

    `aliases` (from the Wikipedia resource layer, `_semantic_context.
    wiki_aliases`) extends matching to true synonyms that share no
    morphology ("pyrite" ↔ "fool's gold"). No hardcoded vocabularies.
    """
    if not term or not hay:
        return False
    if term in hay:
        return True
    for alias in aliases or []:
        if alias and alias in hay:
            return True
    tl = re.sub(r"[^a-z]", "", term.lower())
    if len(tl) < 5:
        return False
    # 1. Suffix-class stem comparison ("thermometry"/"thermometers" → thermo,
    # "classification"/"classifiers" → classif). Plural 's' is folded by the
    # suffix classes ("er"/"ers"), NOT by naive stripping (which mangles
    # "diagnosis" → "diagnosi").
    t_root = _root(tl)
    for w in re.findall(r"[a-z][a-z\-]+", hay):
        w_root = _root(w)
        if w.startswith(t_root) or (
            len(w_root) >= 5 and t_root.startswith(w_root[: len(t_root)])
        ) or (
            len(w_root) >= 5 and w_root.startswith(t_root[: len(w_root)])
        ):
            return True
    # 2. Compound-root containment: term inside a longer compound or vice
    # versa ("thermometry" in "geothermobarometry").
    for w in re.findall(r"[a-z][a-z\-]+", hay):
        wl = w.replace("-", "")
        if len(wl) >= 5 and (tl in wl or wl in tl) and abs(len(wl) - len(tl)) <= 14:
            return True
        # Term ROOT inside a compound ("thermo" root of "thermometry"
        # inside "geothermobarometry"). ≥6-char root keeps specificity.
        if len(t_root) >= 6 and t_root in wl:
            return True
    # 3. Shared prefix ≥5 ("pyrite" vs "pyritization" → pyrit; "diagnosis"
    # vs "diagnostic" → diagnos). Both words ≥5 chars keeps this specific.
    for w in re.findall(r"[a-z][a-z\-]+", hay):
        wl = w.replace("-", "")
        if len(wl) >= 5:
        # longest common prefix
            n = 0
            for a, b in zip(tl, wl):
                if a != b:
                    break
                n += 1
            if n >= 5:
                return True
    return False



def query_content_terms(query: str) -> list[str]:
    """Content terms of a query: entities + geological/scientific vocabulary.

    Generalizes term-coverage filtering: 'dykes age exposed in gadchiroli'
    yields ['gadchiroli', 'dyke'] (entity + content word), so a paper must
    substantively match the SUBJECT, not merely mention a place.
    """
    tokens = re.findall(r"[a-z][a-z\-]{2,}", query.lower())
    seen: list[str] = []
    for tok in tokens:
        stem = _stem(tok)
        if tok in _GENERIC_TERMS or stem in _GENERIC_TERMS:
            continue
        if stem not in {_stem(s) for s in seen}:
            seen.append(stem)
    return seen


def filter_by_term_coverage(
    papers: list,
    query: str,
    min_terms: int = 2,
    alias_map: dict[str, list[str]] | None = None,
) -> tuple[list, list[dict]]:
    """Keep papers matching >= min_terms DISTINCT query content terms.

    Deterministic recall gate against place-name-only matches: a malaria
    survey titled 'Asymptomatic malaria transmission, Gadchiroli' matches
    one term ('gadchiroli') and is excluded; 'Dykes of the Gadchiroli
    district' matches two and is kept. Single-term queries degrade to 1.
    `alias_map` (Wikipedia-derived, `_semantic_context.wiki_aliases`)
    extends each term with true synonyms. Returns (kept, exclusion_log).
    """
    terms = query_content_terms(query)
    alias_map = alias_map or {}
    threshold = min(min_terms, len(terms)) if terms else 0
    if threshold <= 0:
        return papers, []
    # The FIRST content term is the query's PRIMARY subject ('pyrite' in
    # 'pyrite thermometry') — a paper that misses it but matches only
    # generic siblings is off-target. EXCEPT coordinated entity queries
    # ("X and Y dyke system", "A vs B"): both X and Y are primary subjects;
    # a paper about EITHER locality alone is on-target. Structure-driven
    # (and/vs/versus coordination), general across domains.
    primary_set = {terms[0]} if terms else set()
    _coord = re.compile(
        r"\b([a-z][a-z\-]+)\s+(?:and|vs\.?|versus|compared\s+(?:to|with))\s+"
        r"([a-z][a-z\-]+)\b",
        re.IGNORECASE,
    )
    for m in _coord.finditer(query):
        for w in (m.group(1), m.group(2)):
            stem = _stem(w.lower())
            if stem in terms:
                primary_set.add(stem)

    kept: list = []
    excluded: list[dict] = []
    for p in papers:
        hay = f"{getattr(p, 'title', '') or ''} {getattr(p, 'abstract', '') or ''}".lower()
        matched = [t for t in terms if _term_matches(t, hay, alias_map.get(t))]
        hits = len(matched)
        primary_hit = bool(primary_set & set(matched))
        if hits >= threshold and (len(terms) < 2 or primary_hit):
            kept.append(p)
        else:
            excluded.append(
                {
                    "paper_id": getattr(p, "primary_id", ""),
                    "doi": getattr(p, "doi", "") or "",
                    "title": getattr(p, "title", "") or "",
                    "decision": "exclude",
                    "stage": "term_coverage",
                    "reason": (
                        f"matched {hits}/{threshold}+ query terms"
                        + (
                            ""
                            if primary_hit
                            else f" (primary term '{'/'.join(sorted(primary_set))}' missing)"
                        )
                        + f" ({', '.join(matched) if matched else 'none'} of "
                        f"{', '.join(terms[:6])})"
                    ),
                }
            )
    return kept, excluded
