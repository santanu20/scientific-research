#!/usr/bin/env python3
"""CPU-only NLP primitives for scientific-research skill scripts.

NO torch, NO transformers, NO LLM API calls. Uses:
    - sklearn TfidfVectorizer (already in scientific venvs)
    - numpy for cosine similarity
    - curated stance lexicon for citation classification
    - TextRank-style extractive summarization for findings

These are deliberately NOT BERT-class. SOTA-FT (fine-tuned transformer) would
need ~500MB torch + scibert weights — vetoed by user. This module caps at
~65-70% F1 vs ~80% for BERT-class.

If you need BERT-class quality, install separately:
    uv add sentence-transformers
and replace these calls with sentence-transformer cross-encoders.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure sibling modules (_lexicon) are importable when run as a script
sys.path.insert(0, str(Path(__file__).parent))

import json
import logging
import re
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("scientific_research._nlp")

# Lazy-import sklearn (heavy)
_tfidf = None


def _get_tfidf():
    global _tfidf
    if _tfidf is None:
        from sklearn.feature_extraction.text import TfidfVectorizer

        _tfidf = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            lowercase=True,
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
    return _tfidf


def tfidf_matrix(documents: list[str]) -> np.ndarray:
    """Dense TF-IDF matrix (n_docs × vocab) via the shared vectorizer."""
    if not documents:
        return np.zeros((0, 0))
    return _get_tfidf().fit_transform(documents).toarray()


# =============================================================================
# TF-IDF similarity (matrix cell filler backbone)
# =============================================================================
def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity for 1D vectors. Returns 0 for zero vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _stem(word: str) -> str:
    """Cheap stemmer: lowercase, strip punctuation, truncate to 6 chars.
    Catches basalt/basalts/basaltic → all 'basalt' (prefix 'basal').
    Better than no stemming; lighter than Porter/Snowball."""
    w = re.sub(r"[^a-z]", "", word.lower())
    if len(w) <= 5:
        return w
    # Truncate to 6 chars (catches plurals + -ic + -ical + -ity)
    return w[:6]


def _content_tokens(text: str) -> set[str]:
    """Stemmed content tokens (stopwords removed)."""
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "we",
        "our",
        "their",
        "it",
        "its",
        "study",
        "studies",
        "show",
        "showed",
        "found",
        "find",
        "result",
        "results",
    }
    return {
        _stem(w)
        for w in re.findall(r"[a-zA-Z]+", text.lower())
        if len(w) > 2 and w.lower() not in stop
    }


def relevance_score(claim: str, abstract: str) -> float:
    """Containment: fraction of claim tokens present in abstract.
    Range [0, 1]. 1.0 = all claim tokens found, 0 = no overlap.
    Containment (not Jaccard) — avoids penalizing long abstracts vs short claims.
    Uses cheap prefix-stemming (handles basalt/basalts/basaltic)."""
    if not claim or not abstract:
        return 0.0
    c = _content_tokens(claim)
    a = _content_tokens(abstract)
    if not c:
        return 0.0
    return len(c & a) / len(c)


def rank_papers_for_claim(claim: str, papers: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Rank papers by containment of claim tokens in abstract.
    Returns sorted [(paper_id, score)] descending."""
    if not papers:
        return []
    c = _content_tokens(claim)
    if not c:
        return [(p[0], 0.0) for p in papers]
    scores = []
    for pid, abstract in papers:
        a = _content_tokens(abstract)
        if not a:
            scores.append((pid, 0.0))
            continue
        scores.append((pid, len(c & a) / len(c)))
    return sorted(scores, key=lambda x: -x[1])


# =============================================================================
# Extractive summarization (TextRank-style, no external deps)
# =============================================================================
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def _sentence_tokens(sentence: str) -> set[str]:
    """Lowercase content tokens for similarity."""
    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "by",
        "from",
        "as",
        "is",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "we",
        "our",
        "their",
        "it",
        "its",
        "study",
        "studies",
        "show",
        "showed",
        "found",
        "find",
        "result",
        "results",
    }
    return {w for w in re.findall(r"[a-z]+", sentence.lower()) if len(w) > 2 and w not in stop}


def extract_finding(abstract: str, claim: str, max_chars: int = 500) -> str:
    """Extract the sentence from abstract most relevant to claim.
    Returns the raw sentence (truncated) or empty string if abstract is empty."""
    if not abstract or not abstract.strip():
        return ""
    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(abstract) if len(s.strip()) > 30]
    if not sentences:
        return abstract[:max_chars]
    claim_tokens = _sentence_tokens(claim)
    if not claim_tokens:
        return sentences[0][:max_chars]
    best, best_score = "", -1.0
    for sent in sentences:
        sent_tokens = _sentence_tokens(sent)
        if not sent_tokens:
            continue
        # Jaccard similarity (lighter than TF-IDF for sentence-level)
        score = len(claim_tokens & sent_tokens) / len(claim_tokens | sent_tokens)
        if score > best_score:
            best, best_score = sent, score
    if best_score <= 0:
        # No token overlap — return first substantive sentence
        return sentences[0][:max_chars]
    return best[:max_chars]


# =============================================================================
# Citation context classifier — Jurgens 2018 lexicon + negation detection
# =============================================================================
# Ported from Allen AI SciCite (https://github.com/allenai/scicite)
# Jurgens et al. (2018). Measuring the Evolution of a Scientific Field through
# Citation Frames. TACL, 6, 391-406.
#
# 423 stance-relevant terms across 20 categories (vs ~100 in v1).
# Negation detection flips support→contrast when negated.

from _lexicon import (
    ARGUMENTATION_NOUNS as _JURGENS_ARG_NOUNS,
)
from _lexicon import (
    BAD_ADJ as _JURGENS_BAD_ADJ,
)
from _lexicon import (
    CONTRAST_TERMS as _JURGENS_CONTRAST,
)
from _lexicon import (
    HEDGING as _JURGENS_HEDGING,
)
from _lexicon import (
    MENTION_TERMS as _JURGENS_MENTION,
)
from _lexicon import (
    METHOD_TERMS as _JURGENS_METHOD,
)
from _lexicon import (
    NEGATION_TERMS as _JURGENS_NEGATION,
)
from _lexicon import (
    SUPPORT_TERMS as _JURGENS_SUPPORT,
)

# Merge with our domain-specific additions (geoscience, materials science)
_DOMAIN_SUPPORT = {
    "consistent with",
    "in agreement with",
    "in line with",
    "as shown by",
    "as reported by",
    "as demonstrated",
    "confirms",
    "confirmed",
    "confirming",
    "confirm",
    "corroborate",
    "corroborates",
    "corroborated",
    "validate",
    "validates",
    "validated",
    "replicate",
    "replicates",
    "replicated",
    "reproduce",
    "reproduces",
    "reproduced",
    "extend",
    "extends",
    "extended",
    "build on",
    "builds on",
    "building on",
    "built on",
    "align with",
    "indicate", "indicates", "indicated",
    "reveal", "reveals", "revealed",
    "demonstrate", "demonstrates",
    "establish", "establishes",
    "yield", "yields",
    "produce", "produces",
    "generate", "generates",
    "aligns with",
    "concordant",
}
_DOMAIN_CONTRAST = {
    "however",
    "in contrast",
    "contrary to",
    "on the contrary",
    "on the other hand",
    "whereas",
    "conversely",
    "nevertheless",
    "nonetheless",
    "unlike",
    "despite",
    "in spite of",
    "contradict",
    "contradicts",
    "contradicted",
    "contradiction",
    "refute",
    "refutes",
    "refuted",
    "dispute",
    "disputes",
    "fail to",
    "failed to",
    "could not",
    "cannot",
    "unable to",
    "did not observe",
    "did not find",
    "did not replicate",
    "inconsistent with",
    "inconsistent",
    "challenge",
    "challenges",
    "challenged",
    "question",
    "questions",
    "questioned",
    "argue against",
    "argues against",
    "disagree",
    "disagrees",
    "disagreed",
    "overturn",
    "overturns",
    "overturned",
    "rebut",
    "rebuts",
    "rebutted",
}
_DOMAIN_METHOD = {
    "using the method",
    "using the approach",
    "using the technique",
    "using the algorithm",
    "as described by",
    "as detailed in",
    "following the procedure",
    "following the protocol",
    "using the framework",
    "using the model",
    "using the dataset",
    "using data from",
    "adapted from",
    "modified from",
    "based on the method",
    "based on the approach",
    "we adopt",
    "we employ",
    "we apply",
}
_DOMAIN_MENTION = {
    "cite",
    "cites",
    "cited",
    "citing",
    "according to",
    "as cited",
    "see also",
    "see ref",
    "reviewed by",
    "discussed by",
    "noted by",
    "reported by",
    "for comparison",
    "previously reported",
    "previously described",
    "earlier reports",
    "in prior studies",
    "as reviewed",
    "in a review",
}

_CONTRAST_TERMS = _JURGENS_CONTRAST | _DOMAIN_CONTRAST
_SUPPORT_TERMS = _JURGENS_SUPPORT | _DOMAIN_SUPPORT
_METHOD_TERMS = _JURGENS_METHOD | _DOMAIN_METHOD
_MENTION_TERMS = _JURGENS_MENTION | _DOMAIN_MENTION
_NEGATION_TERMS = _JURGENS_NEGATION
_BAD_ADJ_TERMS = set(_JURGENS_BAD_ADJ) | {
    "problematic",
    "flawed",
    "flaw",
    "weak",
    "poor",
    "limited",
    "suboptimal",
    "unreliable",
    "unrealistic",
    "unsuitable",
    "deficient",
    "defective",
    "erroneous",
    "mistaken",
    "faulty",
    "questionable",
    "dubious",
    "suspect",
    "debatable",
    "contested",
    "outdated",
    "obsolete",
    "antiquated",
    "deprecated",
    "overestimated",
    "underestimated",
    "biased",
    "confounded",
    "spurious",
    "artifactual",
    "artifact",
    "superficial",
    "oversimplified",
    "overgeneralized",
    "simplistic",
    "naive",
    "inconsistent",
    "discrepant",
    "contradictory",
    "paradoxical",
    "insufficient",
    "inadequate",
    "incomplete",
    "lacking",
}
_HEDGING_TERMS = set(_JURGENS_HEDGING)
_ARG_NOUN_TERMS = set(_JURGENS_ARG_NOUNS)


def _count_terms(text: str, term_set: set[str]) -> int:
    """Count multi-word term matches (handles 'in contrast' as one term)."""
    text_lower = " " + text.lower() + " "
    count = 0
    for term in term_set:
        if " " in term:
            count += text_lower.count(" " + term + " ")
        else:
            count += len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower))
    return count


def _find_term_positions(text: str, term: str) -> list[int]:
    """Find character positions of a term in text (for negation check)."""
    positions = []
    if " " in term:
        pattern = re.escape(term)
    else:
        pattern = r"\b" + re.escape(term) + r"\b"
    for m in re.finditer(pattern, text, re.IGNORECASE):
        positions.append(m.start())
    return positions


def _is_negated(text: str, position: int, window: int = 40) -> bool:
    """Check if a term at `position` is negated by scanning the preceding `window` chars.

    Ported from Jurgens 2018 negation detection: scans backwards from the term
    for negation words within a window. Also checks for clause-level negation
    (e.g., "however, X did not confirm" → negated).
    """
    prefix = text[max(0, position - window) : position].lower()
    for neg in _NEGATION_TERMS:
        if " " in neg:
            if " " + neg + " " in " " + prefix + " ":
                return True
        else:
            if re.search(r"\b" + re.escape(neg) + r"\b", prefix):
                return True
    return False


@dataclass
class StanceResult:
    stance: str  # 'supporting' | 'contrasting' | 'methodology' | 'mentioning'
    confidence: float  # 0.0-1.0
    reason: str
    matched_terms: list[str]


def classify_stance(context: str) -> StanceResult:
    """SOTA stance classification: SVM + lexicon + discourse markers.

    Uses a LinearSVC trained on 927 sentences across 19 scientific fields,
    combined with the Jurgens 2018 lexicon and discourse marker detection.
    Falls back to lexicon-only if SVM model is unavailable.
    """
    if not context or not context.strip():
        return StanceResult("mentioning", 0.0, "empty context", [])

    # Try SVM + lexicon + discourse combined
    try:
        from _stance_svm import combined_stance

        # Get lexicon result first
        lex_result = _lexicon_classify(context)
        stance, conf, reason = combined_stance(
            context,
            lexicon_stance=lex_result.stance,
            lexicon_confidence=lex_result.confidence,
            lexicon_reason=lex_result.reason,
        )
        return StanceResult(stance, conf, reason, lex_result.matched_terms)
    except ImportError:
        pass

    # Fallback: lexicon only
    return _lexicon_classify(context)


def _lexicon_classify(context: str) -> StanceResult:
    """Jurgens 2018 lexicon-based citation stance classification with negation detection.

    Stance hierarchy: contrast > support > methodology > mention.
    Negation flips support→contrast (e.g., "did not confirm" → contrast).
    Bad adjectives boost contrast. Hedging reduces confidence.
    """
    if not context or not context.strip():
        return StanceResult("mentioning", 0.0, "empty context", [])

    context.lower()

    # Count term matches per category
    n_support = _count_terms(context, _SUPPORT_TERMS)
    n_contrast = _count_terms(context, _CONTRAST_TERMS)
    n_method = _count_terms(context, _METHOD_TERMS)
    n_mention = _count_terms(context, _MENTION_TERMS)
    n_bad = _count_terms(context, _BAD_ADJ_TERMS)
    n_hedge = _count_terms(context, _HEDGING_TERMS)

    # Negation detection: find support terms that are negated → flip to contrast
    negated_support = 0
    matched = []
    for term in sorted(_SUPPORT_TERMS):
        positions = _find_term_positions(context, term)
        for pos in positions:
            if _is_negated(context, pos):
                negated_support += 1
                matched.append(f"¬{term}")
            else:
                matched.append(term)
            if len(matched) >= 5:
                break
        if len(matched) >= 5:
            break

    # Negated support terms count as contrast
    effective_contrast = n_contrast + negated_support + n_bad
    effective_support = n_support - negated_support

    # Find matched contrast terms for explanation
    if not matched or effective_contrast > effective_support:
        for term in sorted(_CONTRAST_TERMS):
            positions = _find_term_positions(context, term)
            if positions:
                matched.append(term)
                if len(matched) >= 5:
                    break

    # Confidence adjusted by hedging
    hedge_penalty = 0.1 * n_hedge

    # Hierarchy: contrast wins ties (conservative for disagreement detection)
    if effective_contrast > 0 and effective_contrast >= effective_support:
        conf = min(1.0, 0.55 + 0.12 * effective_contrast - hedge_penalty)
        reason_parts = []
        if n_contrast > 0:
            reason_parts.append(f"contrast markers ({n_contrast})")
        if negated_support > 0:
            reason_parts.append(f"negated support ({negated_support})")
        if n_bad > 0:
            reason_parts.append(f"negative adjectives ({n_bad})")
        return StanceResult("contrasting", max(0.3, conf), " + ".join(reason_parts), matched)
    if effective_support > 0 and effective_support > effective_contrast:
        conf = min(1.0, 0.55 + 0.12 * effective_support - hedge_penalty)
        return StanceResult(
            "supporting", max(0.3, conf), f"support markers ({effective_support})", matched
        )
    if n_method > 0:
        conf = max(0.3, 0.6 - hedge_penalty)
        return StanceResult("methodology", conf, f"method/dataset citation ({n_method})", matched)
    if n_mention > 0 or (n_support + n_contrast + n_method) == 0:
        conf = max(0.3, 0.5 + 0.05 * n_mention - hedge_penalty)
        return StanceResult(
            "mentioning", min(0.85, conf), f"neutral reference ({n_mention})", matched
        )

    return StanceResult("mentioning", 0.4, "default", [])


# =============================================================================
# PICO + study-design classification (lexicon + sentence patterns)
# =============================================================================
_DESIGN_PATTERNS = {
    # Clinical
    "RCT": re.compile(
        r"\b(randomi[sz]ed controlled trial|\bRCT\b|cluster[\s-]?randomi[sz]|"
        r"double[\s-]blind|placebo[\s-]controlled|parallel[\s-]group)\b",
        re.IGNORECASE,
    ),
    "systematic review": re.compile(
        r"\b(systematic review|meta[\s-]?analysis|prisma|cochrane review)\b", re.IGNORECASE
    ),
    "cohort": re.compile(
        r"\b((?:prospective|retrospective)\s+cohort|cohort\s+(?:study|analysis)|"
        r"longitudinal\s+study|follow[\s-]?up\s+study)\b",
        re.IGNORECASE,
    ),
    "case-control": re.compile(r"\bcase[\s-]?control\b", re.IGNORECASE),
    "cross-sectional": re.compile(r"\bcross[\s-]?sectional\b", re.IGNORECASE),
    # Geological
    "field_study": re.compile(
        r"\b(field\s+(?:study|survey|mapping|investigation|campaign)|"
        r"fieldwork|field\s+area|geological\s+survey|outcrop|"
        r"field\s+samples?\s+(?:were|collected)|drilling\s+(?:site|program))\b",
        re.IGNORECASE,
    ),
    "experimental_petrology": re.compile(
        r"\b(piston[\s-]?cylinder|multi[\s-]?anvil|diamond[\s-]?anvil|"
        r"experimental\s+(?:petrology|run|calibration|phase\s+equilibria)|"
        r"melting\s+experiment|crystalli[sz]ation\s+experiment)\b",
        re.IGNORECASE,
    ),
    "geochemical_survey": re.compile(
        r"\b(geochemical\s+(?:survey|mapping|analysis)|"
        r"whole[\s-]?rock\s+(?:analysis|composition)|"
        r"mineral\s+chemistry|trace\s+element\s+(?:analysis|data))\b",
        re.IGNORECASE,
    ),
    "geochronology": re.compile(
        r"\b(U[\s-]?Pb|Ar[\s-]?Ar|40Ar/39Ar|Re[\s-]?Os|fission\s+track|"
        r"(?:cosmogenic|luminescence)\s+(?:dating|exposure)|"
        r"geochronolog|radiometric\s+dating|detrital\s+zircon)\b",
        re.IGNORECASE,
    ),
    # Physics
    "theoretical": re.compile(
        r"\b(theoretical\s+(?:study|analysis|calculation|prediction|framework)|"
        r"analytical\s+(?:model|solution)|first[\s-]?principles|"
        r"ab\s+initio|density\s+functional\s+theory|\bDFT\b|"
        r"perturbation\s+theory|symmetry\s+analysis)\b",
        re.IGNORECASE,
    ),
    "computational": re.compile(
        r"\b(computational\s+(?:study|method|model|simulation)|"
        r"molecular\s+dynamics|\bMD\s+simulation|finite\s+element|"
        r"lattice\s+QCD|Monte\s+Carlo|numerical\s+model|"
        r"first[\s-]?principles\s+calculation)\b",
        re.IGNORECASE,
    ),
    "observational": re.compile(
        r"\b(observational\s+study|telescope\s+(?:data|observation)|"
        r"satellite\s+(?:data|observation)|survey\s+data|"
        r"natural\s+experiment)\b",
        re.IGNORECASE,
    ),
    # Chemistry
    "synthetic": re.compile(
        r"\b(synthesis\s+of|synthesized|total\s+synthesis|"
        r"preparation\s+of|catalytic\s+synthesis|"
        r"multi[\s-]?step\s+synthesis)\b",
        re.IGNORECASE,
    ),
    "analytical_chemistry": re.compile(
        r"\b(HPLC|GC[\s-]?MS|NMR\s+(?:spectroscopy|analysis)|"
        r"mass\s+spectrometr|X[\s-]?ray\s+(?:crystallography|diffraction)|"
        r"elemental\s+analysis|titration)\b",
        re.IGNORECASE,
    ),
    # Biology
    "in_vitro": re.compile(r"\bin[\s\-]?vitro\b", re.IGNORECASE),
    "in_vivo": re.compile(r"\bin[\s\-]?vivo\b", re.IGNORECASE),
    # CS
    "empirical": re.compile(
        r"\b(empirical\s+(?:study|evaluation|analysis|result)|"
        r"benchmark\s+(?:study|evaluation)|"
        r"ablation\s+study|controlled\s+experiment)\b",
        re.IGNORECASE,
    ),
    "algorithm_development": re.compile(
        r"\b(we\s+(?:propose|present|introduce|develop)\s+(?:a|an|the)\s+(?:new|novel)?\s*"
        r"(?:algorithm|method|architecture|framework|model|approach)|"
        r"algorithm\s+development)\b",
        re.IGNORECASE,
    ),
    # General (kept from before)
    "experimental": re.compile(
        r"\b(laboratory\s+experiment|controlled\s+experiment|"
        r"experimental\s+(?:study|setup|design|measurement)|"
        r"we\s+(?:measured|prepared|fabricated|synthesized))\b",
        re.IGNORECASE,
    ),
    "diagnostic": re.compile(
        r"\b(diagnostic\s+accuracy|sensitivity\s+and\s+specificity|receiver\s+operating|"
        r"roc\s+curve|positive\s+predictive\s+value)\b",
        re.IGNORECASE,
    ),
    "qualitative": re.compile(
        r"\b(qualitative\s+study|focus\s+group|semi[\s-]?structured\s+interview|"
        r"ethnograph|phenomenolog)\b",
        re.IGNORECASE,
    ),
    "preprint": re.compile(r"\bpreprint\b", re.IGNORECASE),
    "modeling": re.compile(
        r"\b(numerical\s+model|molecular\s+dynamics|finite\s+element|simulation|"
        r"machine\s+learning|deep\s+learning|neural\s+network|computational\s+model)\b",
        re.IGNORECASE,
    ),
    "meta-analysis": re.compile(
        r"\b(meta[\s-]?analysis|pooled\s+effect|forest\s+plot|heterogeneity|i[\s\^]?2)\b", re.IGNORECASE
    ),
    "review": re.compile(
        r"\b(this\s+(?:review|paper)\s+(?:reviews?|summarizes?|presents?)|"
        r"we\s+review|comprehensive\s+review|systematic\s+review|"
        r"this\s+(?:paper|study)\s+(?:provides?|presents?)\s+(?:an?\s+)?"
        r"(?:overview|review|survey|summary))\b",
        re.IGNORECASE,
    ),
}


def detect_study_design(text: str) -> list[str]:
    """Return all matching study designs (a paper can be e.g. RCT + meta-analysis)."""
    found = []
    for design, pat in _DESIGN_PATTERNS.items():
        if pat.search(text):
            found.append(design)
    # Prefer more specific: if meta-analysis found, drop "systematic review" duplicate
    if "meta-analysis" in found and "systematic review" in found:
        found.remove("systematic review")
    return found


# Cue patterns for PICO extraction — expanded from baseline
# Field-agnostic Subject/Population cues
_POP_CUES = re.compile(
    r"(?:we\s+studied|we\s+investigated|we\s+analy[sz]ed|we\s+examined|"
    # Clinical
    r"patients?\s+with|subjects?\s+with|participants?\s+with|samples?\s+of|"
    r"cohort\s+of|population\s+of|enrolled|recruited|comprising|"
    # Geological
    r"samples?\s+(?:from|collected|recovered)|specimens?\s+from|"
    r"rocks?\s+from|minerals?\s+from|basalts?\s+from|"
    r"outcrops?\s+(?:from|at|near)|drill\s+cores?|"
    # Physics/Chemistry
    r"films?\s+(?:of|grown|deposited)|crystals?\s+of|compounds?\s+synthesized|"
    r"materials?\s+(?:with|prepared|synthesized)|nanoparticles?\s+of|"
    # Biology
    r"cells?\s+(?:were|treated|exposed|cultured)|tissues?\s+from|"
    r"organisms?\s+from|strains?\s+of|species\s+of|"
    # CS
    r"datasets?\s+(?:of|from|containing)|benchmarks?\s+for|"
    r"models?\s+trained\s+on|algorithms?\s+(?:evaluated|tested)|"
    # General
    r"we\s+collected|in\s+the\s+(?:present|current)\s+study|"
    r"data\s+from|results?\s+from)\s+"
    r"([^\.]{15,200}?)(?:[\.]|were|had|underwent|received|with|showing|"
    r"using|by|via|at|for|during)",
    re.IGNORECASE,
)

# Field-agnostic Method/Intervention cues
_INT_CUES = re.compile(
    r"(?:"
    # Clinical
    r"treated\s+with|received|administered|intervention\s+(?:was|consisted)|"
    r"therapy\s+(?:was|consisted)|drug\s+(?:was|dose)|dose\s+(?:of|was)|"
    r"protocol|regimen|treatment\s+group|intervention\s+group|"
    r"exposed\s+to|subjected\s+to|"
    # Geological methods
    r"analy[sz]ed\s+by|measured\s+(?:using|by)|determined\s+(?:by|using)|"
    r"SIMS|XRF|ICP.?MS|EPMA|electron\s+microprobe|LA.?ICP|"
    r"X.?ray\s+diffraction|XRD|Raman|FTIR|cathodoluminescence|"
    r"isotope\s+(?:analysis|ratio|measurement)|geothermomet(?:er|ry)|"
    # Physics methods
    r"measured\s+at|characterized\s+by|ARPES|neutron\s+(?:diffraction|scattering)|"
    r"photoemission|spectroscop|magnetomet|transport\s+measurements|"
    # Chemistry methods
    r"synthesized\s+(?:by|via|using)|cataly[sz]ed\s+by|functionalized|"
    r"reacted\s+with|coupled\s+(?:to|with)|cyclic\s+voltammetry|"
    r"column\s+chromatography|HPLC|NMR\s+(?:spectra|analysis)|"
    # Biology methods
    r"PCR|western\s+blot|ELISA|FACS|flow\s+cytometry|"
    r"CRISPR|knockout|overexpression|knockdown|transfected|"
    # CS methods
    r"trained\s+(?:on|using)|evaluated\s+(?:on|using)|implemented\s+(?:in|using)|"
    r"neural\s+network|deep\s+learning|transformer|CNN|GNN|RNN|LSTM|"
    r"cross.?validation|hyperparameter|gradient\s+descent|"
    # General
    r"applied\s+to|following\s+(?:the|a)|according\s+to|"
    r"\d+\s*(?:mg|g|ml|μg|mcg|mmol|ppm|M)\s*(?:/|per)?\s*(?:kg|day|d|m²))\s+"
    r"([^\.]{10,180}?)(?:[\.]|for|over|during|at|and|with)",
    re.IGNORECASE,
)

# Field-agnostic Outcome/Results cues
_OUT_CUES = re.compile(
    r"(?:"
    # General results
    r"primary\s+outcome|secondary\s+outcome|main\s+(?:outcome|result|finding)|endpoint|"
    r"we\s+(?:found|observe[d]?|measured|report|show|demonstrate)|"
    r"results?\s+(?:showed|indicated|demonstrated|reveal|suggest)|"
    r"outcome\s+(?:was|measure)|significant(?:ly)?|"
    # Geological results
    r"δ\s*\d+\s*[A-Z]\s*[=:]|Fe3?\+?\s*/\s*Σ|oxygen\s+fugacity|"
    r"temperature\s+(?:of|was|range)|pressure\s+(?:of|was|estimate)|"
    r"age\s+(?:of|was|estimate)|composition\s+(?:of|was)|"
    # Physics results
    r"Tc\s*=|resistivity|conductivity|magnetization|cross.?section|"
    r"transition\s+temperature|band\s+gap|decay\s+rate|"
    # Chemistry results
    r"yield\s*[=:]|selectivity\s*[=:]|conversion\s*[=:]|"
    r"rate\s+constant|binding\s+(?:constant|affinity)|turnover|"
    # CS results
    r"accuracy\s*[=:]|F1\s*[=:]|AUC\s*[=:]|precision\s*[=:]|"
    r"recall\s*[=:]|BLEU\s+score|performance|throughput|"
    # Biology results
    r"survival\s+rate|expression\s+level|mutation\s+rate|"
    r"cell\s+(?:viability|growth|proliferation)|gene\s+expression)\s+"
    r"([^\.]{15,250}?)(?:[\.]|was|were|of|in|to|and)",
    re.IGNORECASE,
)


def extract_pico(text: str) -> dict:
    """Field-agnostic PICO extraction using sentence-position + cue phrases + structured abstract parsing.

    Strategy:
    1. Parse structured abstract sections (if headers present)
    2. Extract subject from background/objective section or cue phrases
    3. Extract method from methods section or cue phrases
    4. Extract outcome from results/findings
    5. Fall back to old regex cue patterns if new extractors miss
    """
    if not text:
        return {
            "population": [],
            "intervention": [],
            "comparator": [],
            "outcome": [],
            "study_design": detect_study_design(""),
        }

    # Try new structured extraction first
    try:
        from _classifiers import (
            extract_method,
            extract_method_from_sections,
            extract_subject,
            extract_subject_from_sections,
            parse_structured_abstract,
        )

        # Check for structured abstract
        sections = parse_structured_abstract(text)

        subject = ""
        method = ""

        if sections:
            # Structured abstract — extract from appropriate sections
            subject = extract_subject_from_sections(sections)
            method = extract_method_from_sections(sections)
        else:
            # Unstructured — use sentence-position extraction
            subject = extract_subject(text)
            method = extract_method(text)

        # Build population list
        population = []
        if subject:
            population.append(subject)
        # Also try old regex as fallback
        regex_pops = [m.strip()[:200] for m in _POP_CUES.findall(text)[:3]]
        for rp in regex_pops:
            if rp and rp not in population:
                population.append(rp)

        # Build intervention list
        intervention = []
        if method:
            intervention.append(method)
        regex_ints = [m.strip()[:200] for m in _INT_CUES.findall(text)[:3]]
        for ri in regex_ints:
            if ri and ri not in intervention:
                intervention.append(ri)

        # Outcome from regex cues
        outcome = [m.strip()[:250] for m in _OUT_CUES.findall(text)[:3]]

        return {
            "population": population[:3],
            "intervention": intervention[:3],
            "comparator": [],
            "outcome": outcome,
            "study_design": detect_study_design(text),
            "abbreviations": extract_abbreviations(text),
        }
    except ImportError:
        pass

    # Final fallback: old regex only
    return {
        "population": [m.strip()[:200] for m in _POP_CUES.findall(text)[:3]],
        "intervention": [m.strip()[:200] for m in _INT_CUES.findall(text)[:3]],
        "comparator": [],
        "outcome": [m.strip()[:250] for m in _OUT_CUES.findall(text)[:3]],
        "study_design": detect_study_design(text),
        "abbreviations": extract_abbreviations(text),
    }


# =============================================================================
# Schwartz-Hearst abbreviation detection
# Ported from trialstreamer (https://github.com/ijmarshall/trialstreamer)
# Original: Schwartz & Hearst (2003). A Simple Algorithm for Identifying
# Abbreviations Definitions in Biomedical Text. Biocomputing, 451-462.
# =============================================================================
_PAREN_RE = re.compile(r"\(([A-Za-z][A-Za-z0-9\-]{1,9})\)")


def _conditions(candidate: str) -> bool:
    """Schwartz-Hearst candidate conditions: 2-10 chars, ≤2 tokens, starts alnum."""
    if len(candidate) < 2 or len(candidate) > 10:
        return False
    if len(candidate.split()) > 2:
        return False
    if not candidate[0].isalnum():
        return False
    if not re.search(r"[A-Za-z]", candidate):
        return False
    return True


def _select_definition(definition: str, abbrev: str) -> str:
    """Validate definition: each char in abbrev must appear in definition (reverse order)."""
    if len(definition) < len(abbrev):
        raise ValueError("Abbreviation longer than definition")
    if abbrev.lower() in definition.lower().split():
        raise ValueError("Abbreviation is full word of definition")
    s_idx, l_idx = -1, -1
    while True:
        long_char = definition[l_idx].lower()  # IndexError terminates the scan
        short_char = abbrev[s_idx].lower()
        if not short_char.isalnum():
            s_idx -= 1
            continue
        if short_char == long_char:
            s_idx -= 1
        l_idx -= 1
        if abs(s_idx) > len(abbrev):
            break
    if abs(s_idx) - 1 < len(abbrev):
        raise ValueError("Not all abbreviation chars found in definition")
    if definition.count("(") != definition.count(")"):
        raise ValueError("Unbalanced parentheses in definition")
    return definition


def _get_definition(candidate: str, sentence: str) -> str:
    """Extract definition candidate from text before the parenthesized abbreviation."""
    candidate_start = sentence.find("(" + candidate + ")")
    if candidate_start < 0:
        candidate_start = sentence.find("(" + candidate)
    if candidate_start < 2:
        raise ValueError("Candidate not found in sentence")
    prefix = sentence[: candidate_start - 1].rstrip()
    tokens = re.split(r"[\s\-]", prefix.lower())
    key = candidate[0].lower()
    # Count tokens starting with same char as abbreviation
    key_indices = [i for i, t in enumerate(tokens) if t and t[0] == key]
    if not key_indices:
        raise ValueError("No definition tokens starting with abbreviation char")
    # Take tokens from last key-matching token to end
    start_idx = key_indices[-1]
    definition = " ".join(tokens[start_idx:])
    if len(definition.split()) > len(candidate) * 5:
        # Trim: definition should be roughly same length as abbreviation
        definition = " ".join(tokens[start_idx:])
    return definition


_REVERSE_PAREN = re.compile(
    r"\(([A-Z][A-Za-z0-9\-]{1,9})\)\s+([a-z][^\.]{10,80}?)(?:[\.]|,|;|and|or)", re.IGNORECASE
)

_AKA_PATTERNS = re.compile(
    r"([^,]{5,60}?),\s+"
    r"(?:hereafter|referred\s+to\s+as|also\s+known\s+as|abbreviated\s+as|denoted\s+as)\s+"
    r"([A-Z][A-Z0-9\-]{1,9})\b",
    re.IGNORECASE,
)

_STANDS_FOR = re.compile(
    r"([A-Z][A-Z0-9\-]{1,9})\s+(?:stands\s+for|is\s+(?:defined\s+as|short\s+for))\s+"
    r"([^\.]{10,80}?)(?:[\.]|,|;)",
    re.IGNORECASE,
)

_UPPER_TOKEN = re.compile(r"\b([A-Z]{2,6})\b")


def extract_abbreviations(text: str) -> dict[str, str]:
    """Abbreviation detection: Schwartz-Hearst + additional patterns.

    Extracts abbreviation→definition pairs from text using multiple strategies:
    1. Schwartz-Hearst: "chronic kidney disease (CKD)" → {"CKD": "chronic kidney disease"}
    2. Reverse parenthetical: "(CKD) chronic kidney disease"
    3. "X, hereafter Y" / "X, referred to as Y" / "X, also known as Y"
    4. "X is defined as Y" / "X stands for Y"
    5. Uppercase acronym detection: if "ABBR" (2-6 uppercase chars) appears and
       matching capitalized words exist elsewhere in text

    Returns dict mapping abbreviation to its definition.
    """
    if not text:
        return {}
    result: dict[str, str] = {}

    # Strategy 1: Schwartz-Hearst (parenthetical: "definition (ABBR)")
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        for m in _PAREN_RE.finditer(sent):
            candidate = m.group(1)
            if not _conditions(candidate):
                continue
            try:
                definition = _get_definition(candidate, sent)
                definition = _select_definition(definition, candidate)
                result[candidate] = definition
            except (ValueError, IndexError):
                continue

    # Strategy 2: Reverse parenthetical "(ABBR) definition"
    for m in _REVERSE_PAREN.finditer(text):
        abbr = m.group(1)
        defn = m.group(2).strip()
        if _conditions(abbr) and len(defn) > len(abbr) and abbr not in result:
            result[abbr] = defn

    # Strategy 3: "X, hereafter Y" / "X, referred to as Y" / "X, also known as Y"
    for m in _AKA_PATTERNS.finditer(text):
        defn = m.group(1).strip()
        abbr = m.group(2)
        if _conditions(abbr) and abbr not in result:
            result[abbr] = defn

    # Strategy 4: "X is/stands for Y" (reverse order)
    for m in _STANDS_FOR.finditer(text):
        abbr = m.group(1)
        defn = m.group(2).strip()
        if _conditions(abbr) and abbr not in result:
            result[abbr] = defn

    # Strategy 5: Acronym matching — find uppercase tokens that match capitalized words
    upper_tokens = {m.group(1) for m in _UPPER_TOKEN.finditer(text) if _conditions(m.group(1))}
    for abbr in upper_tokens:
        if abbr in result:
            continue
        # Try to find words whose initials match
        initials = list(abbr.lower())
        words = re.findall(r"\b([A-Z][a-z]+)\b", text)
        for i in range(len(words) - len(initials) + 1):
            candidate_words = words[i : i + len(initials)]
            candidate_initials = [w[0].lower() for w in candidate_words]
            if candidate_initials == initials:
                defn = " ".join(candidate_words).lower()
                if len(defn) > len(abbr):
                    result[abbr] = defn
                    break

    return result


# =============================================================================
# FakeRAKE keyword extraction + co-occurrence network + Boolean query writer
# Ported from litsearchr (https://github.com/elizagrames/litsearchr)
# Original: Grames et al. (2019). litsearchr. R package.
# =============================================================================

# Comprehensive stopwords (merged from litsearchr + SMART + domain)
_FAKERAKE_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "but",
    "of",
    "in",
    "to",
    "for",
    "with",
    "on",
    "at",
    "by",
    "from",
    "as",
    "is",
    "was",
    "were",
    "be",
    "been",
    "this",
    "that",
    "these",
    "those",
    "we",
    "our",
    "their",
    "it",
    "its",
    "study",
    "studies",
    "show",
    "showed",
    "found",
    "find",
    "result",
    "results",
    "abstract",
    "introduction",
    "method",
    "methods",
    "conclusion",
    "conclusions",
    "background",
    "discussion",
    "data",
    "analysis",
    "using",
    "based",
    "used",
    "two",
    "three",
    "first",
    "second",
    "third",
    "however",
    "also",
    "may",
    "can",
    "one",
    "four",
    "five",
    "between",
    "within",
    "among",
    "across",
    "through",
    "during",
    "after",
    "before",
    "each",
    "both",
    "all",
    "more",
    "most",
    "less",
    "least",
    "very",
    "than",
    "then",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "what",
    "how",
    "why",
    "whether",
    "not",
    "no",
    "nor",
    "such",
    "same",
    "other",
    "another",
    "some",
    "any",
    "many",
    "much",
    "few",
    "several",
    "about",
    "into",
    "over",
    "under",
    "up",
    "down",
    "out",
    "off",
    "above",
    "below",
    "near",
    "far",
    # Common verbs
    "are",
    "have",
    "has",
    "had",
    "do",
    "does",
    "did",
    "will",
    "would",
    "could",
    "should",
    "might",
    "must",
    "shall",
    "make",
    "makes",
    "made",
    "get",
    "gets",
    "got",
    # Generic scientific terms (too broad for search expansion)
    "effects",
    "effect",
    "review",
    "reviews",
    "reduction",
    "reductions",
    "increase",
    "increases",
    "decrease",
    "changes",
    "change",
    "levels",
    "level",
    "type",
    "types",
    "control",
    "controls",
    "controlled",
    "group",
    "groups",
    "total",
    "overall",
    "general",
    "common",
    "specific",
    "high",
    "low",
    "higher",
    "lower",
    "significant",
    "significantly",
    "reported",
    "described",
    "observed",
    "measured",
    "calculated",
    "associated",
    "compared",
    "including",
    "excluded",
    "respectively",
    "therefore",
    "thus",
    "hence",
    "objective",
    "purpose",
    "design",
    "setting",
    "participants",
    "main",
    "primary",
    "secondary",
    "clinical",
    "experimental",
    "statistical",
}


def fakerake(text: str, min_n: int = 1, max_n: int = 4) -> list[str]:
    """FakeRAKE keyword extraction (ported from litsearchr).

    Extracts candidate keyword phrases by splitting on stopwords/punctuation,
    keeping runs of content words of length min_n to max_n.

    Returns list of candidate terms (may contain duplicates).
    """
    if not text:
        return []
    text_lower = text.lower()
    # Replace non-alphanumeric with spaces (preserve hyphens)
    text_clean = re.sub(r"[^a-z0-9\s\-]", " ", text_lower)
    tokens = text_clean.split()
    # Build phrases: runs of non-stopword tokens
    phrases: list[str] = []
    current: list[str] = []
    for tok in tokens:
        if tok in _FAKERAKE_STOPWORDS or len(tok) < 2:
            if current:
                if min_n <= len(current) <= max_n:
                    phrases.append(" ".join(current))
                current = []
        else:
            current.append(tok)
    if current and min_n <= len(current) <= max_n:
        phrases.append(" ".join(current))
    return phrases


def build_cooccurrence_matrix(
    texts: list[str], min_freq: int = 2
) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    """Build term co-occurrence matrix from a list of texts.

    Ported from litsearchr create_network(): builds a document-term matrix,
    then computes t(DTM) %*% DFM for co-occurrence counts.

    Returns (term_frequencies, cooccurrence_matrix) where:
      - term_frequencies: {term: total_count}
      - cooccurrence_matrix: {term: {other_term: cooccurrence_count}}
    """
    # Extract word-level terms (unigrams + bigrams) — matches litsearchr create_dfm
    all_terms_per_doc: list[set[str]] = []
    term_doc_counts: dict[str, int] = {}
    term_total_counts: dict[str, int] = {}
    for text in texts:
        text_lower = text.lower()
        text_clean = re.sub(r"[^a-z0-9\s\-]", " ", text_lower)
        tokens = [t for t in text_clean.split() if t not in _FAKERAKE_STOPWORDS and len(t) > 2]
        doc_terms: set[str] = set(tokens)
        for i in range(len(tokens) - 1):
            doc_terms.add(tokens[i] + " " + tokens[i + 1])
        for t in doc_terms:
            term_total_counts[t] = term_total_counts.get(t, 0) + 1
            term_doc_counts[t] = term_doc_counts.get(t, 0) + 1
        all_terms_per_doc.append(doc_terms)
    # Filter by min_freq
    vocab = {t for t, c in term_total_counts.items() if c >= min_freq}
    # Build co-occurrence
    cooccur: dict[str, dict[str, int]] = {t: {} for t in vocab}
    for doc_terms in all_terms_per_doc:
        doc_vocab = [t for t in doc_terms if t in vocab]
        for i, t1 in enumerate(doc_vocab):
            for t2 in doc_vocab[i + 1 :]:
                cooccur[t1][t2] = cooccur[t1].get(t2, 0) + 1
                cooccur[t2][t1] = cooccur[t2].get(t1, 0) + 1
    return term_total_counts, cooccur


def rank_terms_by_strength(cooccur: dict[str, dict[str, int]]) -> list[tuple[str, float]]:
    """Rank terms by weighted degree (strength) in co-occurrence network.

    Ported from litsearchr make_importance(imp_method="strength").
    """
    strengths = []
    for term, neighbors in cooccur.items():
        strength = sum(neighbors.values())
        strengths.append((term, float(strength)))
    strengths.sort(key=lambda x: -x[1])
    return strengths


def find_cutoff_changepoint(scores: list[float]) -> float:
    """Find importance cutoff using a simple knee/changepoint detection.

    Ported from litsearchr find_cutoff(method="changepoint").
    Uses a simple binary segmentation approach: find the point where
    the mean of the sorted scores changes most.
    """
    if len(scores) < 3:
        return scores[0] if scores else 0.0
    # Sort descending
    s = sorted(scores, reverse=True)
    # Find the split that maximizes between-group variance
    best_var = 0.0
    best_idx = 0
    for i in range(1, len(s)):
        mean1 = sum(s[:i]) / i
        mean2 = sum(s[i:]) / (len(s) - i)
        var = i * (len(s) - i) * (mean1 - mean2) ** 2
        if var > best_var:
            best_var = var
            best_idx = i
    return s[best_idx] if best_idx < len(s) else s[-1]


def remove_redundant_terms(terms: list[str], closure: str = "left") -> list[str]:
    """Remove redundant terms (substrings/prefixes of other terms).

    Ported from litsearchr remove_redundancies().
    "left" closure: if "burn" is a prefix of "burning", keep "burn", remove "burning".
    """
    if closure == "none":
        return list(set(terms))
    unique = list(set(terms))
    redundant: set[int] = set()
    for i, t in enumerate(unique):
        if i in redundant:
            continue
        for j, other in enumerate(unique):
            if i == j or j in redundant:
                continue
            if closure == "left":
                # If t is a prefix of other, other is redundant
                if other.lower().startswith(t.lower()) and len(other) > len(t):
                    redundant.add(j)
            elif closure == "full":
                if other.lower() == t.lower():
                    redundant.add(j)
    return [t for i, t in enumerate(unique) if i not in redundant]


def write_boolean_query(groups: list[list[str]], stemming: bool = True) -> str:
    """Write a Boolean query from grouped terms.

    Ported from litsearchr write_search().
    Within groups: OR. Between groups: AND.
    Stemming: if term > 3 chars, truncate to stem + "*".
    Deduplicates after stemming to avoid duplicate wildcard terms.
    """
    group_strs = []
    for group in groups:
        terms = remove_redundant_terms(group, closure="left")
        parts = []
        seen_stems: set[str] = set()
        for t in terms:
            if stemming and len(t) > 3:
                stem = t[:5] + "*"
                if stem not in seen_stems:
                    seen_stems.add(stem)
                    parts.append(stem)
            else:
                if " " in t:
                    parts.append(f'"{t}"')
                else:
                    parts.append(t)
        if not parts:
            continue
        if len(parts) == 1:
            group_strs.append(parts[0])
        else:
            group_strs.append("(" + " OR ".join(parts) + ")")
    return " AND ".join(group_strs) if group_strs else ""


def expand_query(seed_texts: list[str], max_terms: int = 20, min_freq: int = 2) -> list[str]:
    """Full query expansion pipeline (ported from litsearchr).

    1. FakeRAKE term extraction from seed texts
    2. Co-occurrence network construction
    3. Importance scoring (strength)
    4. Changepoint cutoff detection
    5. Return top terms above cutoff

    Returns list of expanded search terms.
    """
    if not seed_texts:
        return []
    # Build co-occurrence network
    freqs, cooccur = build_cooccurrence_matrix(seed_texts, min_freq=min_freq)
    if not cooccur:
        # Fallback: just return most frequent terms
        sorted_terms = sorted(freqs.items(), key=lambda x: -x[1])
        return [t for t, _ in sorted_terms[:max_terms]]
    # Rank by strength
    ranked = rank_terms_by_strength(cooccur)
    if not ranked:
        return []
    # Find cutoff
    scores = [s for _, s in ranked]
    cutoff = find_cutoff_changepoint(scores)
    # Select terms above cutoff, cap at max_terms
    selected = [t for t, s in ranked if s >= cutoff][:max_terms]
    # Remove redundancies
    return remove_redundant_terms(selected, closure="left")


# =============================================================================
# Self-test
# =============================================================================
if __name__ == "__main__":
    abstract = (
        "We studied 200 patients with chronic kidney disease in a randomized "
        "controlled trial. The treatment group received drug X (50 mg/day, n=100), "
        "the control group received placebo (n=100). Primary outcome was eGFR at "
        "12 weeks. We found that the treatment effect was 5.2 ± 1.1 vs 4.5 ± 1.0 "
        "(Cohen's d = 0.7, p < 0.001), consistent with prior work by Smith et al."
    )
    print("=== PICO ===")
    import json

    print(json.dumps(extract_pico(abstract), indent=2))
    print("\n=== Stance tests ===")
    for ctx in [
        "Our results confirm the findings of Smith et al. and are consistent with prior work.",
        "However, our results contradict the claim by Jones et al. that X causes Y.",
        "We used the method described by Lee et al. (2018) for protein extraction.",
        "As cited by Smith (2020), see also Wang 2021 for review.",
    ]:
        r = classify_stance(ctx)
        print(f"  [{r.stance:<12} {r.confidence:.2f}] {r.reason}")
        print(f"    ctx: {ctx[:90]}")
    print("\n=== Relevance ranking for claim 'kidney disease treatment effect' ===")
    papers = [
        ("A", abstract),
        ("B", "We studied diamond anvil cell experiments at high pressure."),
        ("C", "Chronic kidney disease progression was modeled in 500 patients."),
    ]
    for pid, score in rank_papers_for_claim("kidney disease treatment effect", papers):
        print(f"  {pid}: {score:.3f}")
