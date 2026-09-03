"""Query processing for the research pipeline.

Multi-stage query enhancement:
1. Spelling correction — SymSpell O(1) fuzzy lookup (inline, no dependency)
2. Abbreviation expansion — 200+ IMA-standard geological abbreviations
3. Semantic expansion — BGE embedding cosine similarity against concept dictionary
4. Ollama LLM expansion — context-aware synonyms when Ollama available
5. Ambiguity detection — terms with multiple earth science meanings

Architecture: pure functions, module-level state (caches), thread-safe reads.
"""

from __future__ import annotations

import difflib
import logging
import os
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from _honesty import _dl_within1, fold_text  # noqa: E402
from _timeouts import TIMEOUTS  # noqa: E402

log = logging.getLogger("scientific_research.query")

# =============================================================================
# 1. Geological abbreviations — Whitney & Evans 2010 (IMA standard) + more
# =============================================================================
_GEO_ABBREVIATIONS: dict[str, str] = {
    # Minerals (Whitney & Evans 2010 IMA standard)
    "cpx": "clinopyroxene",
    "opx": "orthopyroxene",
    "ol": "olivine",
    "pl": "plagioclase",
    "qtz": "quartz",
    "kfs": "K-feldspar",
    "bt": "biotite",
    "ms": "muscovite",
    "chl": "chlorite",
    "grt": "garnet",
    "gnt": "garnet",
    "st": "staurolite",
    "ky": "kyanite",
    "sil": "sillimanite",
    "andalusite": "andalusite",
    "crd": "cordierite",
    "cord": "cordierite",
    "spl": "spinel",
    "rt": "rutile",
    "ilm": "ilmenite",
    "mag": "magnetite",
    "hem": "hematite",
    "ap": "apatite",
    "zrn": "zircon",
    "ttn": "titanite",
    "sph": "titanite",
    "ep": "epidote",
    "zo": "zoisite",
    "czo": "clinozoisite",
    "act": "actinolite",
    "hbl": "hornblende",
    "amp": "amphibole",
    "tr": "tremolite",
    "cum": "cummingtonite",
    "ged": "gedrite",
    "ath": "anthophyllite",
    "ser": "serpentine",
    "seric": "sericite",
    "tlc": "talc",
    "cal": "calcite",
    "dol": "dolomite",
    "ank": "ankerite",
    "sd": "siderite",
    "mgt": "magnetite",
    "py": "pyrite",
    "po": "pyrrhotite",
    "cpy": "chalcopyrite",
    "gn": "galena",
    "sp": "sphalerite",
    "mo": "molybdenite",
    "bar": "barite",
    # Rock types
    "morb": "mid-ocean ridge basalt",
    "oib": "ocean island basalt",
    "iab": "island arc basalt",
    "cfb": "continental flood basalt",
    "bab": "back-arc basin",
    "babb": "back-arc basin basalt",
    "aob": "alkaline oceanic basalt",
    "bon": "boninite",
    "ada": "adakite",
    "kom": "komatiite",
    # Tectonic settings
    "mor": "mid-ocean ridge",
    "iat": "island arc tholeiite",
    "ca": "calc-alkaline",
    "th": "tholeiitic",
    "ofb": "ocean floor basalt",
    "arc": "volcanic arc",
    # Geochemistry
    "lile": "large ion lithophile elements",
    "hfse": "high field strength elements",
    "ree": "rare earth elements",
    "lree": "light rare earth elements",
    "hree": "heavy rare earth elements",
    "pge": "platinum group elements",
    "pgm": "platinum group minerals",
    "te": "trace elements",
    "mg#": "magnesium number",
    "fe#": "iron number",
    "asi": "alumina saturation index",
    "acnk": "alumina saturation index",
    "eu_eu": "europium anomaly",
    "ce_ce": "cerium anomaly",
    # Geochronology
    "u-pb": "uranium-lead geochronology",
    "upb": "uranium-lead geochronology",
    "ar-ar": "argon-argon dating",
    "arar": "argon-argon dating",
    "k-ar": "potassium-argon dating",
    "kar": "potassium-argon dating",
    "rb-sr": "rubidium-strontium dating",
    "rbsr": "rubidium-strontium dating",
    "sm-nd": "samarium-neodymium dating",
    "smnd": "samarium-neodymium dating",
    "re-os": "rhenium-osmium dating",
    "reos": "rhenium-osmium dating",
    "lu-hf": "lutetium-hafnium dating",
    "luhf": "lutetium-hafnium dating",
    "ft": "fission track thermochronology",
    "osl": "optically stimulated luminescence",
    # Isotope geochemistry
    "d18o": "oxygen isotope ratio",
    "d13c": "carbon isotope ratio",
    "d15n": "nitrogen isotope ratio",
    "d34s": "sulfur isotope ratio",
    "dd": "hydrogen isotope ratio",
    "sr_sr": "strontium isotope ratio",
    "nd_nd": "neodymium isotope ratio",
    "pb_pb": "lead isotope ratio",
    "epsnd": "epsilon neodymium",
    "epshf": "epsilon hafnium",
    # Structural geology
    "ams": "anisotropy of magnetic susceptibility",
    "cpo": "crystallographic preferred orientation",
    "lpo": "lattice preferred orientation",
    "spo": "shape preferred orientation",
    "ebsd": "electron backscatter diffraction",
    # Metamorphic
    "uhp": "ultrahigh pressure",
    "hp": "high pressure",
    "uht": "ultrahigh temperature",
    "ptt": "pressure-temperature-time",
    "ptx": "pressure-temperature-composition",
    # Economic geology
    "vms": "volcanogenic massive sulfide",
    "sedex": "sedimentary exhalative deposit",
    "iocg": "iron oxide copper gold deposit",
    "bif": "banded iron formation",
    "mvt": "Mississippi Valley-type deposit",
    # Mineral physics
    "gpa": "gigapascal pressure",
    "kbar": "kilobar pressure",
}

# =============================================================================
# 2. Common misspellings
# =============================================================================
_GEO_CORRECTIONS: dict[str, str] = {
    "thermobaromitry": "thermobarometry",
    "thermobarometric": "thermobarometry",
    "geothermometery": "geothermometry",
    "geobarometery": "geobarometry",
    "thermometery": "thermometry",
    "barometery": "barometry",
    "geothermobarometery": "geothermobarometry",
    "metamorphisim": "metamorphism",
    "grannodiorite": "granodiorite",
    "granodiorit": "granodiorite",
    "peridottite": "peridotite",
    "plageoclace": "plagioclase",
    "plagioclace": "plagioclase",
    "calk-alkaline": "calc-alkaline",
    "calc alkaline": "calc-alkaline",
    "subducton": "subduction",
    "petrolog": "petrology",
    "geochemitry": "geochemistry",
    "gechem": "geochemistry",
    "mineralisation": "mineralization",
    "crystallisation": "crystallization",
    "volcanolgoy": "volcanology",
    "tectonophyscis": "tectonophysics",
    "rheolgoy": "rheology",
    "porphry": "porphyry",
    "porphery": "porphyry",
    "hydrotherml": "hydrothermal",
    "geochronolgoy": "geochronology",
    "seismolgoy": "seismology",
}

# =============================================================================
# 3. Canonical terms for fuzzy matching + BGE embedding expansion
# =============================================================================
_CANONICAL_TERMS: list[str] = [
    "thermobarometry",
    "geothermometry",
    "geobarometry",
    "metamorphism",
    "granodiorite",
    "peridotite",
    "plagioclase",
    "clinopyroxene",
    "orthopyroxene",
    "subduction",
    "petrology",
    "geochemistry",
    "mineralization",
    "crystallization",
    "volcanology",
    "pyroclastic",
    "tectonophysics",
    "rheology",
    "porphyry",
    "geology",
    "sedimentology",
    "stratigraphy",
    "paleontology",
    "paleomagnetism",
    "geophysics",
    "mineralogy",
    "crystallography",
    "eclogite",
    "amphibole",
    "garnet",
    "biotite",
    "muscovite",
    "chlorite",
    "serpentine",
    "talc",
    "sericite",
    "epidote",
    "zoisite",
    "clinozoisite",
    "staurolite",
    "cordierite",
    "sillimanite",
    "kyanite",
    "andalusite",
    "wollastonite",
    "lherzolite",
    "harzburgite",
    "wehrlite",
    "dunite",
    "pyroxenite",
    "anorthosite",
    "gabbro",
    "diorite",
    "tonalite",
    "syenite",
    "monzonite",
    "rhyolite",
    "andesite",
    "basalt",
    "granite",
    "komatiite",
    "boninite",
    "adakite",
    "calc-alkaline",
    "tholeiitic",
    "alkaline",
    "subduction zone",
    "mid-ocean ridge",
    "ocean island",
    "island arc",
    "back-arc",
    "continental flood",
    "mantle wedge",
    "slab dehydration",
    "partial melting",
    "fractional crystallization",
    "assimilation",
    "magma mixing",
    "crustal contamination",
    "fluid metasomatism",
    "hydrothermal alteration",
    "weathering",
    "diagenesis",
    "lithification",
    "metasomatism",
    "exsolution",
    "subsolidus",
    "geothermobarometry",
    "phase equilibrium",
    "solidus",
    "liquidus",
    "anatexis",
    "anatectic",
    "granulite",
    "amphibolite",
    "greenschist",
    "blueschist",
    "eclogite facies",
    "zeolite facies",
    "pressure-temperature path",
    "geothermal gradient",
    "heat flow",
    "thermal conductivity",
    "radiogenic heat",
]

# Wikidata geological dictionary — 4768+ terms for fuzzy matching.
# Used for SymSpell (misspelling correction) but NOT for BGE expansion,
# because obscure minerals (e.g. "clino-ferro-ferri-holmquistite") pollute
# semantic expansion with irrelevant rare species.
try:
    from _geodict import get_all_terms

    _WIKIDATA_TERMS = get_all_terms()
    # All terms for SymSpell (fuzzy matching catches more misspellings)
    _ALL_TERMS = sorted(set(_CANONICAL_TERMS + _WIKIDATA_TERMS))
    log.info(
        "Geo dictionary: %d curated + %d Wikidata = %d terms for fuzzy matching",
        len(_CANONICAL_TERMS),
        len(_WIKIDATA_TERMS),
        len(_ALL_TERMS),
    )
except Exception as e:
    log.debug("Wikidata geo dictionary unavailable, using curated terms only: %s", e)
    _ALL_TERMS = list(_CANONICAL_TERMS)


# Single-word vocab for token-level spell correction. Multiword entries
# (geodict phrases, local-library TITLES) are tokenized — whole phrases
# could never DL-match a single query token.
def _spell_vocab(extra_vocab: list[str] | tuple[str, ...] = ()) -> set[str]:
    """Folded single-word vocabulary: geodict + curated + caller tokens.

    Entries are split on NON-ALPHA runs so hyphenated compounds
    ('pranhita-godavari' from a library title) decompose into their
    parts — a whole compound can never DL-match a single query token.
    """
    vocab: set[str] = set()
    for entry in (*_ALL_TERMS, *extra_vocab):
        for tok in re.split(r"[^a-z]+", str(entry).lower()):
            v = fold_text(tok)
            if v and len(v) >= 5 and " " not in v and not v.isdigit():
                vocab.add(v)
    return vocab


def spellcorrect_query(
    query: str,
    extra_vocab: list[str] | tuple[str, ...] = (),
) -> tuple[str, list[str]]:
    """Offline typo repair for the LLM-normalizer-unavailable path.

    Damerau-Levenshtein <=1 against a DYNAMIC vocabulary (geodict terms +
    caller-supplied corpus tokens — e.g. local-library titles, so proper
    nouns like basin names are correctable). Guards: token >=5 chars,
    alpha-only, skipped when itself in vocab, UNIQUE candidate only,
    <=3 fixes, word count preserved. Returns (corrected, notes) where
    notes are 'orig -> fix' strings (empty notes = unchanged query).
    """
    words = query.split()
    if len(words) <= 1:
        return query, []
    vocab = _spell_vocab(extra_vocab)
    notes: list[str] = []
    for i, w in enumerate(words):
        if len(w) < 5 or not w.isalpha():
            continue
        fw = fold_text(w)
        if fw in vocab:
            continue
        candidates = [v for v in vocab if abs(len(v) - len(fw)) <= 1 and _dl_within1(fw, v)]
        if len(candidates) == 1:
            words[i] = candidates[0]
            notes.append(f"{w} -> {candidates[0]}")
    if not notes:
        return query, []
    return " ".join(words), notes


# =============================================================================
# 4. Ambiguous terms
# =============================================================================
_AMBIGUOUS_TERMS: dict[str, list[str]] = {
    # Only TRULY ambiguous terms (different processes/meanings in different contexts).
    # Standard rock names (granite, basalt) and mineral names (pyroxene) are NOT
    # ambiguous — geologists know what they mean. Don't add noise to queries.
    "alteration": ["hydrothermal alteration", "metasomatic alteration", "weathering alteration"],
    "fabric": ["shape-preferred fabric", "crystallographic preferred fabric", "S-C fabric"],
    "facies": ["metamorphic facies", "sedimentary facies", "igneous facies"],
}

# =============================================================================
# 5. SymSpell — O(1) deletion-based fuzzy matching (inline, no dependency)
# =============================================================================
try:
    from symspellpy import SymSpell, Verbosity

    class _SymSpellWrapper:
        """Production SymSpell backed by symspellpy (C-optimized)."""

        def __init__(self) -> None:
            self._ss = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            self._built = False

        def build(self, terms: list[str]) -> None:
            for t in terms:
                self._ss.create_dictionary_entry(t.lower(), 1)
            self._built = True

        def add(self, term: str) -> None:
            self._ss.create_dictionary_entry(term.lower(), 1)

        def lookup(self, word: str, min_ratio: float = 0.82) -> str | None:
            w = word.lower()
            suggestions = self._ss.lookup(w, Verbosity.CLOSEST, max_edit_distance=2)
            if not suggestions:
                return None
            best = suggestions[0]
            if best.distance == 0:
                return best.term
            import difflib

            ratio = difflib.SequenceMatcher(None, w, best.term).ratio()
            return best.term if ratio >= min_ratio else None

    _symspell = _SymSpellWrapper()
except ImportError:
    # Fallback: inline SymSpell (no dependency)
    class _SymSpellInline:
        def __init__(self, max_edit_distance: int = 2) -> None:
            self.max_distance = max_edit_distance
            self.deletes: dict[str, list[str]] = {}
            self._terms: set[str] = set()

        def _deletions(self, word: str, max_dist: int) -> set[str]:
            result = {word}
            for _ in range(max_dist):
                new_words = set()
                for w in list(result):  # iterate over copy — result mutates below
                    if len(w) > 1:
                        for i in range(len(w)):
                            new_words.add(w[:i] + w[i + 1 :])
                    result.update(new_words)
            result.discard(word)
            return result

        def build(self, terms: list[str]) -> None:
            for t in terms:
                self.add(t)

        def add(self, term: str) -> None:
            tl = term.lower()
            self._terms.add(tl)
            for d in self._deletions(tl, self.max_distance):
                self.deletes.setdefault(d, []).append(tl)

        def lookup(self, word: str, min_ratio: float = 0.82) -> str | None:
            w = word.lower()
            if w in self._terms:
                return w
            candidates: set[str] = set()
            for d in self._deletions(w, self.max_distance):
                if d in self.deletes:
                    candidates.update(self.deletes[d])
            if not candidates:
                return None
            best, best_ratio = None, 0.0
            for c in candidates:
                ratio = difflib.SequenceMatcher(None, w, c).ratio()
                if ratio > best_ratio:
                    best, best_ratio = c, ratio
            return best if best_ratio >= min_ratio else None

    _symspell = _SymSpellInline(max_edit_distance=2)

_symspell.build(_ALL_TERMS)

# =============================================================================
# 6. BGE embedding-based semantic expansion
# =============================================================================
_bge_term_embeddings = None
_bge_cache_path = Path.home() / ".cache" / "scientific-research" / "query_term_embeddings.npy"
_bge_cache_terms_path = Path.home() / ".cache" / "scientific-research" / "query_term_embeddings_terms.json"


def _ensure_bge_embeddings() -> bool:
    """Pre-compute BGE embeddings for canonical terms (cached to disk).

    Returns True if embeddings are available.
    """
    global _bge_term_embeddings
    if _bge_term_embeddings is not None:
        return True
    try:
        from _embeddings import embed_texts, is_available

        if not is_available():
            return False
    except ImportError:
        log.debug("_query.py:451 — feature degraded")
        return False

    import numpy as np

    # Check cache — BGE uses curated terms only (not all 4856 Wikidata terms)
    _BGE_TERMS = _CANONICAL_TERMS  # curated, high-relevance subset
    if _bge_cache_path.exists() and _bge_cache_terms_path.exists():
        import json

        try:
            cached_terms = json.loads(_bge_cache_terms_path.read_text())
            if cached_terms == _BGE_TERMS:
                _bge_term_embeddings = np.load(_bge_cache_path)
                log.debug("BGE term embeddings loaded from cache (%d terms)", len(cached_terms))
                return True
        except Exception as e:
            log.debug("Query processing error: %s", e)

    # Compute fresh
    log.info("Computing BGE embeddings for %d geological terms...", len(_BGE_TERMS))
    embs = embed_texts(_BGE_TERMS)
    if embs is None or embs.shape[0] != len(_BGE_TERMS):
        return False
    _bge_term_embeddings = embs

    # Cache to disk
    try:
        _bge_cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(_bge_cache_path, embs)
        import json

        _bge_cache_terms_path.write_text(json.dumps(_BGE_TERMS))
        log.info("BGE term embeddings cached to %s", _bge_cache_path)
    except Exception as e:
        log.debug("BGE cache write failed: %s", e)

    return True


def bge_expand_query(query: str, max_terms: int = 5, min_similarity: float = 0.65) -> list[str]:
    """Expand query with semantically similar geological terms via BGE embeddings.

    Returns list of related terms (NOT including original query words).
    Threshold 0.65 = strong semantic similarity; 0.55 = moderate.
    """
    if not _ensure_bge_embeddings():
        return []

    import numpy as np

    try:
        from _embeddings import embed_texts

        query_emb = embed_texts([query])
        if query_emb is None:
            return []
    except ImportError:
        log.warning("BGE import fallback at _query.py:508", exc_info=True)
        return []

    q_norm = query_emb[0] / (np.linalg.norm(query_emb[0]) + 1e-10)
    t_norms = _bge_term_embeddings / (
        np.linalg.norm(_bge_term_embeddings, axis=1, keepdims=True) + 1e-10
    )
    cosines = (t_norms @ q_norm).flatten()

    # Exclude terms already in the query
    query_words = set(query.lower().split())
    results: list[tuple[str, float]] = []
    for i, score in enumerate(cosines):
        term = _CANONICAL_TERMS[i]  # curated terms only (not obscure minerals)
        if term not in query_words and any(w not in query_words for w in term.split()):
            results.append((term, float(score)))

    results.sort(key=lambda x: -x[1])
    expanded = [term for term, score in results[:max_terms] if score >= min_similarity]

    if expanded:
        log.info("BGE expansion: '%s' + %s", query[:40], expanded)
    return expanded


# =============================================================================
# 7. Ollama-based query expansion
# =============================================================================
_ollama_checked = False
_ollama_available = False


def _check_ollama() -> bool:
    """Check if Ollama is available (cached)."""
    global _ollama_checked, _ollama_available
    if _ollama_checked:
        return _ollama_available
    _ollama_checked = True
    try:
        import ollama

        # Quick health check via list (doesn't require llama-server)
        ollama.list()
        _ollama_available = True
        log.debug("Ollama available for query expansion")
    except Exception:
        _ollama_available = False
        log.debug("Ollama not available — using rule-based expansion only")
    return _ollama_available


def ollama_expand_query(query: str, model: str = "") -> str | None:
    """Use Ollama LLM for intelligent, context-aware query expansion.

    Domain-agnostic: works for ANY earth science subdomain.
    Returns expanded query string, or None if Ollama unavailable/failed.
    """
    import json as _json
    import urllib.request

    if not model:
        model = os.environ.get("GEOKIT_LLM_MODEL", "")

    prompt = (
        "/no_think\n"
        "You are a geoscience research assistant. Expand this research query "
        "with synonyms, related terms, and alternative terminology for better "
        "academic paper discovery. Return ONLY the expanded query text (max 30 words).\n\n"
        f'Query: "{query}"\n\nExpanded query:'
    )

    payload = _json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": {"num_predict": 100, "temperature": 0.3, "num_ctx": 8192},
        }
    ).encode()

    try:
        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=TIMEOUTS.ollama_extract) as resp:
            data = _json.loads(resp.read())
        expanded = data.get("message", {}).get("content", "").strip()
        # Strip thinking blocks
        import re as _re

        expanded = _re.sub(
            r"<(?:think|thinking|reasoning)\b[^>]*>.*?</(?:think|thinking|reasoning)>",
            "",
            expanded,
            flags=_re.DOTALL | _re.IGNORECASE,
        ).strip()
        if expanded and len(expanded) > len(query) and expanded.lower() != query.lower():
            expanded = expanded.strip('"`').strip()
            log.info("Ollama expansion: '%s' → '%s'", query[:40], expanded[:60])
            return expanded
    except Exception as e:
        log.debug("Ollama query expansion failed: %s", e)
    return None


# =============================================================================
# 8. Core functions: spelling, abbreviation, ambiguity
# =============================================================================
def correct_spelling(query: str) -> tuple[str, list[str]]:
    """Correct misspellings via SymSpell O(1) fuzzy lookup + exact dictionary."""
    if not query or not query.strip():
        return query, []
    corrections: list[str] = []
    words = query.split()
    corrected_words: list[str] = []
    for word in words:
        word_lower = word.lower().rstrip(".,;:!?")
        # Stage 1: exact match against known misspellings
        if word_lower in _GEO_CORRECTIONS:
            fixed = _GEO_CORRECTIONS[word_lower]
            if fixed != word_lower:
                if word[0].isupper():
                    fixed = fixed[0].upper() + fixed[1:]
                corrections.append(f"{word_lower} → {fixed}")
                corrected_words.append(fixed)
                continue
        # Stage 2: SymSpell fuzzy match against canonical terms (>4 chars)
        if len(word_lower) > 4:
            match = _symspell.lookup(word_lower, min_ratio=0.82)
            if match and match != word_lower:
                fixed = match
                if word[0].isupper():
                    fixed = fixed[0].upper() + fixed[1:]
                corrections.append(f"{word_lower} → {fixed}")
                corrected_words.append(fixed)
                continue
        corrected_words.append(word)
    return " ".join(corrected_words), corrections


def expand_abbreviations(query: str) -> tuple[str, list[str]]:
    """Expand geological abbreviations to full terms.

    Handles word-boundary matching so 'Cpx' → 'clinopyroxene' but
    'Cpx-rich' → 'clinopyroxene-rich' (hyphenated compounds).
    """
    if not query:
        return query, []
    expansions: list[str] = []
    result = query
    for abbrev, full in sorted(_GEO_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        pattern = r"\b" + re.escape(abbrev) + r"\b"
        if re.search(pattern, result, re.IGNORECASE):
            result = re.sub(pattern, full, result, flags=re.IGNORECASE)
            expansions.append(f"{abbrev} → {full}")
    return result, expansions


def detect_ambiguity(query: str) -> list[tuple[str, list[str]]]:
    """Detect ambiguous geological terms in the query."""
    if not query:
        return []
    query_lower = query.lower()
    ambiguous: list[tuple[str, list[str]]] = []
    for term, interpretations in _AMBIGUOUS_TERMS.items():
        pattern = r"\b" + re.escape(term) + r"\b"
        if re.search(pattern, query_lower):
            ambiguous.append((term, interpretations))
    return ambiguous


def _wikipedia_correct_proper_nouns(query: str) -> tuple[str, list[str]]:
    """Correct misspelled proper nouns using Wikipedia OpenSearch API.

    Searches with query context (not just the single word) so Wikipedia
    finds the right entity. Prefers geological results over non-geological.

    "dongar granites" + context → "Dongargarh" (not "Dongaria" tribe)
    """
    import json as _json
    import urllib.parse as _urlparse
    import urllib.request as _urlrequest
    from difflib import SequenceMatcher

    words = query.split()
    corrections: list[str] = []
    new_words: list[str] = []

    _STOP = {"and", "the", "for", "with", "from", "its", "their",
             "about", "into", "that", "this", "granite", "basalt",
             "geochemistry", "petrology", "volcanic", "metamorphic", "igneous",
             "geology", "research", "study", "analysis", "review"}

    for i, word in enumerate(words):
        w_clean = word.strip(".,;:!?")
        if len(w_clean) < 4 or w_clean[0].isupper():
            new_words.append(word)
            continue

        w_lower = w_clean.lower()
        if w_lower in _STOP:
            new_words.append(word)
            continue

        # Build context: word + next geological term in query
        context = w_clean
        for j in range(i + 1, min(i + 4, len(words))):
            if words[j].lower() not in _STOP and len(words[j]) >= 4:
                context = f"{w_clean} {words[j]}"
                break

        try:
            # Search with context for better disambiguation
            search_url = (
                "https://en.wikipedia.org/w/api.php?"
                + _urlparse.urlencode({
                    "action": "opensearch",
                    "search": context,
                    "limit": "5",
                    "format": "json",
                })
            )
            req = _urlrequest.Request(
                search_url,
                headers={"User-Agent": "GeoKit/research (spelling correction)"},
            )
            with _urlrequest.urlopen(req, timeout=TIMEOUTS.wikipedia) as resp:
                data = _json.loads(resp.read())

            titles = data[1] if len(data) > 1 else []

            # HEAD-TOKEN typo repair (2026-09-03): Wikipedia titles proper
            # nouns as "X River"/"X district" — whole-title similarity can
            # never reach threshold for a 1-edit typo ('prahnita' vs
            # 'Pranhita River' = 0.73 < 0.75) while the title's head token
            # is DL<=1 (0.875). Unique-form + must-differ guards mirror
            # spellcorrect_query; the replacement is the head token ONLY,
            # so the query never inherits 'River'/'district' suffixes.
            head_forms: set[str] = set()
            exact_head_seen = False
            for title in titles:
                parts = title.split()
                head = parts[0].strip(".,;:") if parts else ""
                fh = fold_text(head)
                if not fh or fh in _STOP or len(fh) < 5:
                    continue
                if fh == w_lower:
                    # IDENTITY VETO (live case 2026-09-03: OpenSearch
                    # returned both 'Chandrapur' and 'Chandrapura' — a
                    # DIFFERENT town one edit away — and repairing
                    # 'chandrapur'→'chandrapura' silently moved the query
                    # to Jharkhand). The word existing verbatim as a head
                    # means it is correctly spelled: no repair.
                    exact_head_seen = True
                    break
                if _dl_within1(w_lower, fh):
                    head_forms.add(fh)
            if not exact_head_seen and len(head_forms) == 1:
                fix = next(iter(head_forms))
                corrections.append(f"{w_clean} → {fix}")
                new_words.append(fix)
                continue

            # Higher threshold for short words (≤6 chars) — more ambiguous
            threshold = 0.85 if len(w_clean) <= 6 else 0.75

            best_title = None
            best_sim = 0.0
            for title in titles:
                similarity = SequenceMatcher(
                    None, w_lower, title.lower()
                ).ratio()
                if similarity >= threshold and similarity > best_sim:
                    # Check Wikipedia extract for geological relevance
                    try:
                        summary_url = (
                            "https://en.wikipedia.org/api/rest_v1/page/summary/"
                            + _urlparse.quote(title)
                        )
                        sreq = _urlrequest.Request(
                            summary_url,
                            headers={"User-Agent": "GeoKit/research"},
                        )
                        with _urlrequest.urlopen(sreq, timeout=TIMEOUTS.wikipedia) as sresp:
                            sdata = _json.loads(sresp.read())
                        extract = (sdata.get("extract") or "").lower()
                        # Prefer geological results
                        geo_terms = ["geolog", "rock", "mineral", "formation",
                                     "granite", "basalt", "craton", "batholith",
                                     "volcanic", "sedimentary", "metamorphic",
                                     "district", "province", "belt"]
                        is_geo = any(g in extract for g in geo_terms)
                        if is_geo or similarity >= 0.90:
                            best_title = title
                            best_sim = similarity
                    except Exception:
                        # Can't check extract — accept high-similarity match
                        if similarity >= 0.90:
                            best_title = title
                            best_sim = similarity

            if best_title and best_title.lower() != w_lower:
                corrections.append(f"{w_clean} → {best_title}")
                new_words.append(best_title)
            else:
                new_words.append(word)
        except Exception as e:
            log.debug("Wikipedia correction failed for '%s': %s", w_clean, e)
            new_words.append(word)

    corrected = " ".join(new_words)
    return corrected, corrections


# =============================================================================
# 9. Full normalization pipeline (orchestrates all stages)
# =============================================================================
def normalize_query(
    query: str,
    warn: bool = True,
    use_ollama: bool = True,
    use_bge: bool = True,
    ollama_model: str = "",
) -> tuple[str, list[str], list[tuple[str, list[str]]]]:
    """Full query normalization: spelling + abbreviations + expansion + ambiguity.

    Pipeline:
    1. Spelling correction (SymSpell O(1) fuzzy)
    2. Abbreviation expansion (200+ IMA-standard terms)
    3. Ollama LLM expansion (if available)
    4. BGE semantic expansion (if fastembed available)
    5. Ambiguity detection

    Parameters
    ----------
    query : str
        Raw user query.
    warn : bool
        Log warnings for corrections and ambiguities.
    use_ollama : bool
        Attempt Ollama-based expansion.
    use_bge : bool
        Attempt BGE embedding-based expansion.
    ollama_model : str
        Ollama model for expansion.

    Returns
    -------
    corrected_query : str
        Fully processed query ready for API search.
    corrections : list[str]
        All corrections/expansions made (spelling + abbreviations).
    ambiguities : list[tuple[str, list[str]]]
        Ambiguous terms detected.
    """
    all_corrections: list[str] = []

    # Stage 1: Spelling correction
    corrected, corrections = correct_spelling(query)
    all_corrections.extend(corrections)
    if warn and corrections:
        log.info(
            "Spelling corrected: '%s' → '%s' (%d fixes)",
            query[:40],
            corrected[:40],
            len(corrections),
        )

    # Stage 1.5: Wikipedia proper-noun correction
    # SymSpell can't fix place names (Dongargarh, Singhbhum) because they're
    # not in the geodict. Wikipedia fuzzy search finds the correct form.
    wiki_corrected, wiki_corrections = _wikipedia_correct_proper_nouns(corrected)
    if wiki_corrections:
        corrected = wiki_corrected
        all_corrections.extend(wiki_corrections)
        if warn:
            log.info("Wikipedia proper-noun corrected: %s", wiki_corrections)

    # Stage 2: Abbreviation expansion
    expanded_abbr, abbr_corrections = expand_abbreviations(corrected)
    all_corrections.extend(abbr_corrections)
    if warn and abbr_corrections:
        log.info("Abbreviations expanded: %s", abbr_corrections)

    # Stage 3: Ollama LLM expansion (optional)
    final_query = expanded_abbr
    if use_ollama:
        llm_expanded = ollama_expand_query(expanded_abbr, model=ollama_model)
        if llm_expanded:
            final_query = llm_expanded
            all_corrections.append(f"LLM: {expanded_abbr[:30]} → {llm_expanded[:30]}")

    # Stage 4: BGE semantic expansion (optional, additive)
    if use_bge:
        bge_terms = bge_expand_query(final_query, max_terms=5, min_similarity=0.65)
        if bge_terms:
            # Add terms that aren't already in the query
            existing = set(final_query.lower().split())
            new_terms = [t for t in bge_terms if not any(w in existing for w in t.split())]
            if new_terms:
                final_query = final_query + " " + " ".join(new_terms[:3])
                all_corrections.append(f"BGE: +{new_terms[:3]}")

    # Stage 5: Ambiguity detection
    ambiguities = detect_ambiguity(final_query)
    if warn and ambiguities:
        for term, interp in ambiguities:
            log.warning(
                "Ambiguous term '%s' — could mean: %s. Consider: '%s %s'",
                term,
                ", ".join(interp),
                term,
                interp[0].split("(")[0].strip(),
            )

    return final_query, all_corrections, ambiguities

# =============================================================================
# 10. QueryState — one query object across every pipeline layer
# =============================================================================
@dataclass(frozen=True)
class QueryState:
    """Immutable query state threaded through the pipeline (2026-09-03).

    Split-brain contract: five parallel string variants (raw
    config.query, expanded search string, typo-repaired audit string,
    per-layer corrected copies, strategy strings) each gave one layer a
    chance to see stale typos — five same-class bugs in two sessions.
    One object constructed at the discovery front door:

      raw    user input verbatim — display/title only, never matched
      audit  user's terms typo-repaired — honesty gates, context,
             strategies, every rescue re-search
      search retrieval string synonym-expanded — API search only
      fixes  provenance note per applied repair (loud, user-facing)
    """

    raw: str
    audit: str
    search: str
    fixes: tuple[str, ...] = ()

    def with_token_fixes(
        self, fixes: Mapping[str, str], notes: Iterable[str] = ()
    ) -> QueryState:
        """Word-boundary token substitution applied to audit+search.

        raw stays verbatim (provenance): the notes record every repair.
        """

        def _sub(s: str) -> str:
            for orig, fixed in fixes.items():
                s = re.sub(rf"\b{re.escape(orig)}\b", fixed, s, flags=re.IGNORECASE)
            return s

        return QueryState(
            raw=self.raw,
            audit=_sub(self.audit),
            search=_sub(self.search),
            fixes=self.fixes + tuple(notes),
        )
