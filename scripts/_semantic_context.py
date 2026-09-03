"""Semantic geological context classifier.

Replaces ALL hardcoded term lists with BGE embedding discipline centroids.
Computes one centroid per discipline from seed terms, then classifies any
query by nearest centroid (cosine similarity). Generalizes to ANY geological
term — no manual maintenance, no word-by-word additions.

Also provides Wikipedia REST API lookup for proper nouns (place names like
"Dongargarh", "Singhbhum", "Bushveld") that geodict/BGE can't classify alone.

Architecture:
    Layer 0: Geodict term matching (4768 Wikidata terms, instant)
    Layer 1: BGE discipline centroids (semantic, handles synonyms/variants)
    Layer 2: Wikipedia REST API (proper nouns, place names)
    Layer 3: Caller falls back to LLM

Caching:
    Centroids computed once from seed terms, cached to ~/.cache/scientific-research/
    Query embeddings cached by content hash (in _embeddings.py)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
from _timeouts import TIMEOUTS

log = logging.getLogger("scientific_research.semantic_context")

_CACHE_DIR = Path.home() / ".cache" / "scientific_research" / "semantic_context"
_CENTROID_CACHE = _CACHE_DIR / "discipline_centroids.npz"
_CENTROID_META = _CACHE_DIR / "discipline_centroids_meta.json"

# Discipline seed terms — NOT a lookup table. These are REPRESENTATIVE terms
# for computing BGE centroid embeddings. The centroid then matches ANY
# semantically related term automatically. Adding a new term here is like
# adding a training example, not a lookup entry.
_DISCIPLINE_SEEDS: dict[str, list[str]] = {
    # ── All-fields extension (2026-08-22): the classifier must serve any
    # discipline. Same mechanism, broader domain coverage — each entry is
    # bootstrap vocabulary for an embedding centroid, never a gate.
    "clinical_medicine": [
        "randomized controlled trial",
        "placebo",
        "clinical outcome",
        "patient cohort",
        "adverse event",
        "diagnostic accuracy",
        "treatment efficacy",
        "double-blind",
        "mortality rate",
        "clinical guideline",
        "biomarker",
        "dose response",
    ],
    "molecular_biology": [
        "gene expression",
        "CRISPR",
        "genome sequencing",
        "transcription factor",
        "protein folding",
        "PCR",
        "cell signaling pathway",
        "mutagenesis",
        "RNA interference",
        "western blot",
        "phenotype",
        "knockout mouse",
    ],
    "ecology": [
        "biodiversity",
        "species richness",
        "ecosystem services",
        "habitat fragmentation",
        "population dynamics",
        "food web",
        "conservation biology",
        "invasive species",
        "succession",
        "trophic level",
        "meta population",
        "biome",
    ],
    "environmental": [
        "water quality",
        "air pollution",
        "heavy metal contamination",
        "climate change impact",
        "carbon emissions",
        "wastewater treatment",
        "soil contamination",
        "environmental monitoring",
        "remediation",
        "particulate matter",
        "greenhouse gas",
        "risk assessment",
    ],
    "chemistry": [
        "catalysis",
        "organic synthesis",
        "reaction mechanism",
        "spectroscopy",
        "chromatography",
        "coordination complex",
        "polymerization",
        "stoichiometry",
        "chemical kinetics",
        "supramolecular",
        "electrochemistry",
        "functional group",
    ],
    "physics": [
        "quantum mechanics",
        "superconductivity",
        "phase transition",
        "laser optics",
        "particle accelerator",
        "condensed matter",
        "thermodynamics",
        "entanglement",
        "semiconductor",
        "plasma",
        "relativity",
        "band structure",
    ],
    "computer_science": [
        "deep learning",
        "neural network",
        "transformer architecture",
        "natural language processing",
        "computer vision",
        "reinforcement learning",
        "benchmark dataset",
        "convolutional",
        "algorithm complexity",
        "distributed system",
        "compiler optimization",
        "ablation study",
    ],
    "engineering": [
        "finite element analysis",
        "structural health monitoring",
        "fatigue life",
        "compressive strength",
        "heat exchanger",
        "control system",
        "factor of safety",
        "nondestructive testing",
        "manufacturing process",
        "vibration analysis",
        "reinforced concrete",
        "stress strain",
    ],
    "materials_science": [
        "microstructure",
        "tensile strength",
        "crystal structure",
        "thin film",
        "composite material",
        "sintering",
        "grain boundary",
        "X-ray diffraction",
        "hardness",
        "corrosion resistance",
        "ceramic",
        "alloy design",
    ],
    "social_science": [
        "survey methodology",
        "panel data",
        "semi-structured interview",
        "regression discontinuity",
        "socioeconomic status",
        "thematic analysis",
        "questionnaire validity",
        "demographic transition",
        "policy evaluation",
        "likert scale",
        "case study methodology",
        "statistical significance",
    ],
    # Legacy geology entries follow (original 12).

    "igneous": [
        "granite",
        "granodiorite",
        "tonalite",
        "syenite",
        "gabbro",
        "diorite",
        "peridotite",
        "anorthosite",
        "magma",
        "pluton",
        "batholith",
        "intrusion",
        "fractional crystallization",
        "partial melting",
        "layered intrusion",
        "cumulate",
        "ophiolite",
        "craton",
        "Archean",
        "greenstone belt",
    ],
    "volcanic": [
        "basalt",
        "rhyolite",
        "andesite",
        "dacite",
        "volcano",
        "caldera",
        "eruption",
        "pyroclastic",
        "lava flow",
        "ignimbrite",
        "tephra",
        "volcanic hazard",
        "effusive eruption",
        "flood basalt",
        "volcanic ash",
    ],
    "metamorphic": [
        "metamorphism",
        "metamorphic grade",
        "metamorphic facies",
        "schist",
        "gneiss",
        "eclogite",
        "amphibolite",
        "granulite",
        "marble",
        "quartzite",
        "migmatite",
        "skarn",
        "blueschist",
        "P-T path",
        "metamorphic reaction",
        "ultrahigh pressure",
        "exhumation",
        "subduction metamorphism",
    ],
    "mantle": [
        "mantle",
        "asthenosphere",
        "lithospheric mantle",
        "mantle wedge",
        "mantle plume",
        "mantle source",
        "mantle melt",
        "peridotite mantle",
        "mantle convection",
        "mantle tomography",
    ],
    "sedimentary": [
        "sedimentary rock",
        "sandstone",
        "shale",
        "limestone",
        "turbidite",
        "carbonate platform",
        "sequence stratigraphy",
        "sedimentary basin",
        "diagenesis",
        "detrital",
        "flysch",
        "molasse",
        "delta",
        "clastic",
    ],
    "ore": [
        "ore deposit",
        "mineralization",
        "hydrothermal alteration",
        "porphyry copper",
        "gold vein",
        "epithermal",
        "VHMS",
        "SEDEX",
        "BIF iron formation",
        "metallogeny",
        "ore genesis",
        "prospectivity",
        "sulfide mineralization",
        "chromite deposit",
        "nickel sulfide",
    ],
    "structural": [
        "structural geology",
        "tectonics",
        "fault",
        "fold",
        "thrust",
        "shear zone",
        "deformation",
        "strain",
        "stress",
        "cleavage",
        "foliation",
        "lineation",
        "kinematics",
        "orogen",
        "collision",
        "fold-thrust belt",
        "nappe",
        "rifting",
    ],
    "planetary": [
        "planetary geology",
        "lunar geology",
        "martian geology",
        "impact crater",
        "meteorite impact",
        "shock metamorphism",
        "suevite",
        "breccia impact",
        "exoplanet geology",
    ],
    "hydrogeological": [
        "hydrogeology",
        "groundwater",
        "aquifer",
        "contaminant transport",
        "permeability",
        "hydraulic conductivity",
        "water table",
        "well",
    ],
    "geophysical": [
        "seismic tomography",
        "gravity anomaly",
        "magnetic anomaly",
        "geophysical survey",
        "crustal thickness",
        "seismic reflection",
        "magnetotelluric",
        "heat flow",
        "geophysical imaging",
    ],
    "fugacity": [
        "oxygen fugacity",
        "sulfur fugacity",
        "redox state",
        "fO2",
        "fS2",
        "oxygen barometry",
        "mantle redox",
        "oxidation state",
        "FMQ buffer",
        "NNO buffer",
        "oxygen fugacity buffer",
    ],
    "seismic_anisotropy": [
        "seismic anisotropy",
        "mantle anisotropy",
        "SKS splitting",
        "shear wave splitting",
        "lattice preferred orientation",
        "LPO",
        "olivine LPO",
        "azimuthal anisotropy",
        "radial anisotropy",
    ],
}

_centroids: np.ndarray | None = None
_centroid_labels: list[str] | None = None


def _compute_centroids() -> tuple[np.ndarray, list[str]] | None:
    """Compute BGE discipline centroids from seed terms.

    Returns (centroids_array, labels) or None if BGE unavailable.
    Centroids are cached to disk after first computation.
    """
    global _centroids, _centroid_labels
    if _centroids is not None and _centroid_labels is not None:
        return _centroids, _centroid_labels

    # Try disk cache
    if _CENTROID_CACHE.exists() and _CENTROID_META.exists():
        try:
            _centroids = np.load(_CENTROID_CACHE)["centroids"]
            _centroid_labels = json.loads(_CENTROID_META.read_text())["labels"]
            log.debug(
                "Discipline centroids loaded from cache (%d disciplines)",
                len(_centroid_labels),
            )
            return _centroids, _centroid_labels
        except Exception as e:
            log.debug("Centroid cache read failed: %s — re-computing", e)

    # Compute fresh
    try:
        from _embeddings import embed_texts, is_available

        if not is_available():
            log.debug("BGE embeddings unavailable — semantic context disabled")
            return None
    except ImportError:
        log.debug("_embeddings module unavailable — semantic context disabled")
        return None

    labels: list[str] = []
    centroid_list: list[np.ndarray] = []

    for discipline, seeds in _DISCIPLINE_SEEDS.items():
        log.debug("Embedding discipline '%s' (%d seeds)...", discipline, len(seeds))
        embs = embed_texts(seeds, use_cache=True)
        if embs is None or embs.shape[0] == 0:
            log.warning("Embedding failed for discipline '%s' — skipping", discipline)
            continue
        # Normalize each embedding, then average
        norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10
        embs_normalized = embs / norms
        centroid = embs_normalized.mean(axis=0)
        # Re-normalize centroid
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        centroid_list.append(centroid.astype(np.float32))
        labels.append(discipline)

    if not centroid_list:
        return None

    _centroids = np.array(centroid_list)
    _centroid_labels = labels

    # Cache to disk
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(_CENTROID_CACHE, centroids=_centroids)
        _CENTROID_META.write_text(json.dumps({"labels": labels}, indent=2), encoding="utf-8")
        log.info(
            "Discipline centroids cached: %d disciplines to %s",
            len(labels),
            _CENTROID_CACHE,
        )
    except Exception as e:
        log.debug("Centroid cache write failed: %s", e)

    return _centroids, _centroid_labels


def classify_context_semantic(query: str, min_similarity: float = 0.15) -> str | None:
    """Classify query into geological discipline using BGE centroids.

    Embeds the query once, finds nearest discipline centroid by cosine
    similarity. Handles ANY geological term — no manual lists needed.

    Returns discipline string ("igneous", "volcanic", "metamorphic", etc.)
    or None if BGE unavailable or similarity below threshold.
    """
    result = _compute_centroids()
    if result is None:
        return None

    centroids, labels = result

    try:
        from _embeddings import embed_texts

        emb = embed_texts([query], use_cache=True)
        if emb is None or emb.shape[0] == 0:
            return None
        emb = emb[0]
        emb_norm = emb / (np.linalg.norm(emb) + 1e-10)
    except Exception as e:
        log.debug("Query embedding failed: %s", e)
        return None

    # Cosine similarity to each centroid
    sims = centroids @ emb_norm
    best_idx = int(np.argmax(sims))
    best_sim = float(sims[best_idx])

    if best_sim < min_similarity:
        log.debug(
            "Semantic context: best match '%s' sim=%.3f < %.3f threshold for '%s'",
            labels[best_idx],
            best_sim,
            min_similarity,
            query[:60],
        )
        return None

    log.debug(
        "Semantic context: '%s' → '%s' (sim=%.3f)",
        query[:60],
        labels[best_idx],
        best_sim,
    )
    return labels[best_idx]


def match_material_semantic(query: str) -> str | None:
    """Find the closest geological material in the query using BGE.

    Instead of iterating a hardcoded MATERIALS dict, uses the Wikidata
    geodict (4768 terms) + BGE to find the semantically closest material.

    Two-step: (1) word-boundary match against geodict (instant), (2) BGE
    nearest-neighbor if no exact match.
    """
    q_lower = query.lower()

    # Step 1: word-boundary match against Wikidata geodict
    try:
        from _geodict import get_all_terms

        all_terms = get_all_terms()
        for term in sorted(all_terms, key=len, reverse=True):
            if len(term) < 3:
                continue
            if re.search(r"\b" + re.escape(term.lower()) + r"\b", q_lower):
                return term.lower()
    except Exception as e:
        log.debug("Geodict matching failed: %s", e)

    # Step 2: BGE nearest-neighbor against geodict (slower, handles typos/synonyms)
    result = _compute_centroids()
    if result is None:
        return None

    try:
        from _embeddings import embed_texts

        emb = embed_texts([query], use_cache=True)
        if emb is None:
            return None

        # Use a subset of geodict terms (top 500 most common) for speed
        from _geodict import get_minerals, get_rocks

        candidate_terms = (get_rocks() + get_minerals())[:500]
        if not candidate_terms:
            return None

        cand_embs = embed_texts(candidate_terms[:200], use_cache=True)
        if cand_embs is None:
            return None

        query_norm = emb[0] / (np.linalg.norm(emb[0]) + 1e-10)
        cand_norms = cand_embs / (np.linalg.norm(cand_embs, axis=1, keepdims=True) + 1e-10)
        sims = cand_norms @ query_norm

        best_idx = int(np.argmax(sims))
        best_sim = float(sims[best_idx])

        if best_sim > 0.55:
            return candidate_terms[best_idx].lower()
    except Exception as e:
        log.debug("BGE material matching failed: %s", e)

    return None


# =============================================================================
# Wikipedia REST API — proper noun / place name classification
# =============================================================================

_WIKI_API = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_WIKI_TIMEOUT = TIMEOUTS.wikipedia


def classify_context_wikipedia(query: str) -> str | None:
    """Classify query context using Wikipedia REST API.

    For proper nouns (place names like 'Dongargarh', 'Singhbhum') that
    geodict/BGE can't classify. Queries Wikipedia, extracts page categories,
    maps geological categories to disciplines.

    Returns discipline string or None if not geological / not found.
    """
    # Extract potential proper nouns (capitalized words that aren't common terms)
    words = query.split()
    proper_nouns = [w for w in words if w[0].isupper() and len(w) > 3]

    for noun in proper_nouns[:2]:  # try first 2 proper nouns
        try:
            url = _WIKI_API + urllib.parse.quote(noun)
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": "GeoKit/research (semantic classifier)",
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=_WIKI_TIMEOUT) as resp:
                data = json.loads(resp.read())

            extract = (data.get("extract") or "").lower()
            description = (data.get("description") or "").lower()

            # Check if this is a geological entity
            geo_indicators = [
                "geolog",
                "rock",
                "mineral",
                "formation",
                "granite",
                "basalt",
                "intrusion",
                "volcanic",
                "metamorphic",
                "sedimentary",
                "ore",
                "deposit",
                "fault",
                "tecton",
                "craton",
                "shield",
                "batholith",
                "pluton",
                "magma",
                "metamorph",
                "stratigraph",
                "province",
            ]

            is_geological = any(ind in extract or ind in description for ind in geo_indicators)
            if not is_geological:
                continue

            # Classify by extracting discipline keywords from the extract
            text = f"{extract} {description}"

            # Check against discipline seeds
            discipline_scores: dict[str, int] = {}
            for discipline, seeds in _DISCIPLINE_SEEDS.items():
                score = sum(1 for seed in seeds if seed.lower() in text)
                if score > 0:
                    discipline_scores[discipline] = score

            if discipline_scores:
                best = max(discipline_scores, key=discipline_scores.get)
                log.debug(
                    "Wikipedia classified '%s' → '%s' (scores: %s)",
                    noun,
                    best,
                    discipline_scores,
                )
                return best

        except Exception as e:
            log.debug("Wikipedia lookup failed for '%s': %s", noun, e)
            continue

    return None

def _wikidata_aliases(term: str) -> list[str]:
    """Structured aliases from Wikidata: wbsearchentities → QID →
    wbgetentities.aliases (en). Two GETs, no sentence parsing — the
    authoritative multilingual alias set for the concept.
    Fails open ([]) on any error."""
    wd_api = "https://www.wikidata.org/w/api.php"
    ua = {
        # polite-pool identifier per Wikimedia API etiquette guidelines
        "User-Agent": "scientific-research/2.0 (research pipeline; polite-pool)",
        "Accept": "application/json",
    }

    def _get(params: dict) -> dict:
        url = wd_api + "?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(
            urllib.request.Request(url, headers=ua), timeout=_WIKI_TIMEOUT
        ) as resp:
            return json.loads(resp.read().decode("utf-8", errors="replace"))

    try:
        # wbsearchentities returns mixed-type hits for ambiguous terms —
        # "zircon" hit #1 is a composer (person), hit #2 the mineral. Pick
        # the first NON-person hit by description screening, not blind #1.
        hits: list[dict] = []
        for variant in dict.fromkeys([term, term.capitalize(), term.upper()]):
            search = _get(
                {
                    "action": "wbsearchentities",
                    "search": variant,
                    "language": "en",
                    "format": "json",
                    "limit": 5,
                }
            )
            hits = search.get("search") or []
            if hits:
                break
        _person_desc = re.compile(
            r"\b(?:given name|surname|family name|composer|musician|painter|"
            r"politician|actor|actress|footballer|athlete|writer|novelist|"
            r"film director|songwriter|singer|artist|architect)\b",
            re.IGNORECASE,
        )
        _label_person = re.compile(r"^(?:human|person)\b", re.IGNORECASE)
        chosen = None
        for h in hits:
            label = (h.get("label") or "").strip().lower()
            desc = (h.get("description") or "").strip()
            # Label must match the query case-insensitively — entity labels
            # are stored in canonical case ("CRISPR"), queries arrive any case
            if label != term.lower():
                continue
            if _person_desc.search(desc) or _label_person.match(desc):
                continue
            chosen = h
            break
        if not chosen:
            return []
        qid = chosen.get("id", "")
        if not re.match(r"^Q\d+$", qid or ""):
            return []
        ent = _get(
            {
                "action": "wbgetentities",
                "ids": qid,
                "props": "aliases",
                "languages": "en",
                "format": "json",
            }
        )
        entity = (ent.get("entities") or {}).get(qid) or {}
        out = []
        for al in (entity.get("aliases") or {}).get("en") or []:
            raw = (al.get("value") or "").strip()
            # Person-name check on the ORIGINAL casing — the value is
            # lowercased below, which would defeat the capitalization
            # heuristic ("Andrew Aversa" → "andrew aversa" passes naive
            # filters and poisons paper matching).
            if _looks_person_name(raw):
                continue
            v = raw.lower()
            # ≤7 words: full acronym expansions are the highest-value
            # aliases ("clustered regularly interspaced short palindromic
            # repeats"). Sanitized by the shared _clean_alias contract.
            if 2 <= len(v.split()) <= 7 and _clean_alias(v, term.lower(), out):
                out.append(v)
        return out
    except Exception as e:
        log.debug("wikidata alias lookup unavailable for %r: %s", term, e)
        return []


def _looks_person_name(v: str) -> bool:
    """Heuristic person-name filter: exactly 2-3 capitalized tokens, no
    hyphens/digits/parentheses (chemical and technical aliases keep those)."""
    tokens = v.split()
    if not (2 <= len(tokens) <= 3):
        return False
    if re.search(r"[-\d()]", v):
        return False
    return all(t[:1].isupper() and t[1:].islower() for t in tokens if t)


# ── Alias sanitizer (single source of truth for both wiki layers) ────────
_ALIAS_BOILERPLATE = {
    "family", "class", "order", "genus", "species", "group", "type",
    "term", "word", "name", "list", "article", "page", "category",
    "specimen", "toponym", "surname",
}
# Single-word geographic-unit aliases (country/continent/state) must NEVER
# stand in for a specific locality: "gadchiroli" → "india" matched every
# Indian-geology paper and defeated the primary-term gate.
_ALIAS_GEO_UNIT = re.compile(
    r"^(?:india|china|japan|russia|brazil|canada|australia|germany|"
    r"france|italy|spain|uk|usa|united states|united kingdom|england|"
    r"africa|asia|europe|north america|south america|antarctica|"
    r"oceania|maharashtra)$"
)
# IPA-transcription detector. NOTE (bug 2026-09-01): the affricates t͡ʃ/d͡ʒ
# MUST be alternations, NOT character-class members — inside [...] they
# decompose to plain 't' and 'd', rejecting most English aliases
# ("fool's gold" has a 'd').
_ALIAS_TRANSCRIPTION = re.compile(r"[\[\]/ˈˌɡʈɖɳɽ͡ʃʒ]|t͡ʃ|d͡ʒ")


def _clean_alias(v: str, low_term: str, existing: list[str]) -> bool:
    """True when alias `v` is acceptable: not the term itself, not a dup,
    not boilerplate, not a broad geo-unit, not a pronunciation guide."""
    if not v or v == low_term or v in existing:
        return False
    if v in _ALIAS_BOILERPLATE:
        return False
    if _ALIAS_GEO_UNIT.match(v):
        return False
    if _ALIAS_TRANSCRIPTION.search(v):
        return False
    return True



def wiki_aliases(term: str, max_aliases: int = 6) -> list[str]:
    """Aliases for ANY concept: Wikidata structured aliases FIRST (proper
    `aliases` field — no sentence parsing), Wikipedia summary apposition
    phrases as fallback enrichment. Generalizes query vocabulary to every
    domain (geology, medicine, CS). Cached on disk; fails open ([]) on
    network/parse error. No packages — plain REST GETs."""
    term = (term or "").strip()
    if not term:
        return []

    import hashlib

    cache = (
        Path.home()
        / ".cache"
        / "scientific_research"
        / "wiki_aliases"
        / (hashlib.sha256(term.lower().encode()).hexdigest()[:24] + ".json")
    )
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    aliases: list[str] = []
    low_term = term.lower()

    # Layer 1 — Wikidata structured aliases (authoritative); _wikidata_aliases
    # already sanitizes via _clean_alias, this is a second guard.
    for v in _wikidata_aliases(term):
        if _clean_alias(v, low_term, aliases):
            aliases.append(v)

    # Layer 2 — Wikipedia summary fallback/enrichment (apposition phrases)
    try:
        req = urllib.request.Request(
            _WIKI_API + urllib.parse.quote(term.replace(" ", "_"), safe=""),
            headers={
                "User-Agent": "scientific-research/2.0 (research pipeline)",
                "Accept": "application/json",
            },
        )
        with urllib.request.urlopen(req, timeout=_WIKI_TIMEOUT) as resp:
            data = (
                json.loads(resp.read().decode("utf-8", errors="replace"))
                if resp.status == 200
                else {}
            )
    except Exception as e:
        log.debug("wiki summary unavailable for %r: %s", term, e)
        data = {}

    extract = data.get("extract", "")
    desc = data.get("description", "")
    first_sent = re.split(r"(?<=[.!?])\s", extract)[0] if extract else ""

    def _keep(phrase: str) -> None:
        p = re.sub(r"\s+", " ", phrase.strip().strip(",;:.()\u201c\u201d\"'"))
        words = p.split()
        if not (1 <= len(words) <= 5):
            return
        if re.search(r"\b(?:is|are|was|were|which|that|who|also|known)\b", p, re.I):
            return
        pl = p.lower()
        if _clean_alias(pl, low_term, aliases):
            aliases.append(pl)

    head = re.split(r"\b(?:is|are|refers to|denotes)\b", first_sent, maxsplit=1)[0]
    for chunk in re.split(r",|;|\(|\)|\bor\b", head, flags=re.I):
        _keep(chunk)
    for chunk in re.split(r"\bof\b|\bwith\b|,|\;", desc, flags=re.I):
        _keep(chunk)

    out = aliases[:max_aliases]
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    except OSError:
        pass
    return out
