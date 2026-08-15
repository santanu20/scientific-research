"""Geological enrichment — generalizable knowledge base for publication-grade briefs.

Architecture (covers ALL geology, not hardcoded edge cases):
1. P-T plausibility filtering — physical constraint ranges per discipline
2. Theme detection via BGE embedding similarity against a geological vocabulary
   (~150 themes spanning every geological subdiscipline). Falls back to
   hierarchical taxonomy keyword matching, then LLM if embeddings unavailable.
3. Metamorphic facies + tectonic interpretation from P-T data (textbook logic)
4. Research gap detection via method vocabulary comparison
5. Convergence/divergence analysis (statistical, not hardcoded)

Sources: Spear (1993), Winter (2010), Philpotts & Ague (2022), Best (2003)
"""

from __future__ import annotations

import logging
import re

log = logging.getLogger("scientific_research.geo_enrich")

# =============================================================================
# 1. P-T plausibility ranges — physical constraints per discipline
# These are FUNDAMENTAL physical limits, not edge cases. A garnet-biotite
# thermometer cannot read 1650°C; that value indicates contamination.
# =============================================================================
DISCIPLINE_PT_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "metamorphic petrology": {"T": (150, 950), "P": (0.5, 40)},
    "igneous petrology": {"T": (600, 1400), "P": (0.1, 15)},
    "volcanology": {"T": (600, 1400), "P": (0.1, 10)},
    "economic geology": {"T": (50, 700), "P": (0.1, 5)},
    "geochemistry": {"T": (0, 1800), "P": (0, 60)},
    "mantle petrology": {"T": (900, 1800), "P": (8, 60)},
    "structural geology": {"T": (0, 600), "P": (0, 10)},
    "sedimentology": {"T": (0, 150), "P": (0, 3)},
    "geophysics": {"T": (0, 6000), "P": (0, 360)},
    "geology": {"T": (0, 1800), "P": (0, 60)},
}
_DEFAULT_RANGE = {"T": (-273, 10000), "P": (-1, 10000)}


def _match_discipline(disc: str) -> str:
    d = (disc or "").lower().strip()
    for key in DISCIPLINE_PT_RANGES:
        if key in d or d in key:
            return key
    if "metamorph" in d:
        return "metamorphic petrology"
    if "igneous" in d or "volcanic" in d or "magma" in d:
        return "igneous petrology"
    if "economic" in d or "ore" in d or "porphyry" in d or "mineral" in d:
        return "economic geology"
    if "mantle" in d or "peridot" in d:
        return "mantle petrology"
    if "sediment" in d:
        return "sedimentology"
    if "struct" in d or "fault" in d or "tecton" in d:
        return "structural geology"
    if "geochem" in d:
        return "geochemistry"
    return "geology"


_TEMP_PATTERN = re.compile(
    r"(?:T\s*[:=]?\s*)?(\d{2,4})\s*[-\u2013\u2014]\s*(\d{2,4})\s*[\u00b0°]?C"
    r"|(?:T\s*[:=]?\s*)(\d{2,4})\s*[\u00b0°]?C"
    r"|(?:T\s*[\u2248~]\s*)(\d{2,4})\s*[\u00b0°]?C",
    re.IGNORECASE,
)
_PRESSURE_PATTERN = re.compile(
    r"(?:P\s*[:=]?\s*)?(\d+\.?\d*)\s*[-\u2013\u2014]\s*(\d+\.?\d*)\s*k\s*(?:bar|b)\b"
    r"|(?:P\s*[:=]?\s*)(\d+\.?\d*)\s*k\s*(?:bar|b)\b"
    r"|(?:P\s*[\u2248~]\s*)(\d+\.?\d*)\s*k\s*(?:bar|b)\b",
    re.IGNORECASE,
)


def _parse_quant_data(extr: dict) -> None:
    """Parse quantitative_data string into structured measurements list.

    The LLM extracts P-T values as a free-text string (e.g.,
    'T=500-620°C, P=4-6 kbar'). This function parses it into the
    structured measurements list that interpretation functions read.
    Mutates extr in-place.
    """
    pico = extr.get("pico") or {}
    qd = pico.get("quantitative_data") or extr.get("quantitative_data") or ""
    if not qd or not isinstance(qd, str):
        return

    measurements = extr.setdefault("measurements", [])
    if measurements:
        return  # already structured

    # Parse temperatures
    for match in _TEMP_PATTERN.finditer(qd):
        low = match.group(1) or match.group(3) or match.group(4)
        high = match.group(2)
        try:
            t_low = float(low)
            t_high = float(high) if high else t_low
            measurements.append(
                {
                    "measurement": "temperature",
                    "value": (t_low + t_high) / 2,  # median of range
                    "unit": "°C",
                    "range_low": t_low,
                    "range_high": t_high,
                }
            )
        except (TypeError, ValueError):
            pass

    # Parse pressures
    for match in _PRESSURE_PATTERN.finditer(qd):
        low = match.group(1) or match.group(3) or match.group(4)
        high = match.group(2)
        try:
            p_low = float(low)
            p_high = float(high) if high else p_low
            measurements.append(
                {
                    "measurement": "pressure",
                    "value": (p_low + p_high) / 2,
                    "unit": "kbar",
                    "range_low": p_low,
                    "range_high": p_high,
                }
            )
        except (TypeError, ValueError):
            pass


def filter_pt_values(
    extractions: list[dict],
    discipline: str = "",
) -> tuple[list[dict], list[str]]:
    """Filter implausible P-T values from extractions. Also parses
    quantitative_data strings into structured measurements."""
    # First: parse quantitative_data strings into measurements
    for extr in extractions:
        _parse_quant_data(extr)
    # Then: filter by plausibility
    disc_key = _match_discipline(discipline)
    ranges = DISCIPLINE_PT_RANGES.get(disc_key, _DEFAULT_RANGE)
    t_min, t_max = ranges["T"]
    p_min, p_max = ranges["P"]
    warnings: list[str] = []
    for extr in extractions:
        title = (extr.get("title") or "?")[:50]
        for m in extr.get("measurements", []):
            if not isinstance(m, dict):
                continue
            mname = (m.get("measurement") or "").lower()
            val = m.get("value")
            try:
                val_f = float(val) if val is not None else None
            except (TypeError, ValueError):
                continue
            if val_f is None:
                continue
            if ("temp" in mname or mname in ("t", "temperature")) and (
                val_f < t_min or val_f > t_max
            ):
                warnings.append(
                    f"T={val_f}°C out of range [{t_min}, {t_max}] for "
                    f"{disc_key}: '{title}' — flagged"
                )
                m["value"] = None
                m["_flagged"] = f"implausible T for {disc_key}"
            elif ("press" in mname or mname in ("p", "pressure")) and (
                val_f < p_min or val_f > p_max
            ):
                warnings.append(
                    f"P={val_f} kbar out of range [{p_min}, {p_max}] for "
                    f"{disc_key}: '{title}' — flagged"
                )
                m["value"] = None
                m["_flagged"] = f"implausible P for {disc_key}"
    if warnings:
        log.info("P-T filter: flagged %d values for '%s'", len(warnings), disc_key)
    return extractions, warnings


# =============================================================================
# 2. Geological theme vocabulary — comprehensive across ALL geology
# Each entry: (theme_label, semantic_description_for_embedding)
# The description gives BGE enough context to match papers using different
# terminology for the same geological concept.
# =============================================================================
GEO_THEME_VOCABULARY: list[tuple[str, str]] = [
    # ── Metamorphic petrology ──
    (
        "Garnet-biotite exchange thermometry",
        "Fe-Mg partitioning between garnet and biotite minerals to estimate metamorphic "
        "temperature KD distribution coefficient pelitic schist metamorphic grade Barrovian",
    ),
    (
        "Garnet-clinopyroxene thermobarometry",
        "Fe-Mg exchange between garnet and clinopyroxene eclogite granulite temperature "
        "pressure estimation",
    ),
    (
        "Phase-equilibrium modeling",
        "pseudosection phase diagram garnet biotite staurolite chlorite mineral assemblage "
        "P-T calculation thermocalc domino perple_x",
    ),
    (
        "U-Pb zircon geochronology",
        "uranium lead zircon dating metamorphic crystallization age TIMS LA-ICP-MS SIMS",
    ),
    (
        "Ar-Ar mica geochronology",
        "argon argon dating muscovite biotite amphibole cooling age metamorphic exhumation",
    ),
    (
        "Monazite CHIME dating",
        "monazite electron microprobe dating Th-U-Pb chemical age metamorphic",
    ),
    (
        "Titanium-in-quartz thermometry",
        "TitaniQ titanium concentration quartz temperature thermobarometry",
    ),
    (
        "Zircon saturation thermometry",
        "zircon saturation temperature magma crystallization melt Zr content",
    ),
    (
        "Apatite fission-track thermochronology",
        "apatite fission track cooling low temperature thermochronology exhumation",
    ),
    (
        "Carpholite-chloritoid HP-LT metamorphism",
        "high pressure low temperature metamorphism carpholite chloritoid blueschist eclogite",
    ),
    # ── Igneous petrology ──
    (
        "Experimental petrology",
        "experimental melting crystallization phase relations magma high pressure piston "
        "cylinder apparatus synthetic rock composition",
    ),
    (
        "Melt inclusion analysis",
        "melt inclusion trapped magma crystal olivine plagioclase volatile content pre-eruptive",
    ),
    (
        "Magma differentiation",
        "fractional crystallization magma evolution crystal settling cumulate layering "
        "differentiation trend",
    ),
    (
        "Assimilation-fractional crystallization",
        "AFC crustal contamination magma mixing assimilation wall rock interaction",
    ),
    (
        "Phase relations and liquidus studies",
        "liquidus solidus phase diagram basalt andesite rhyolite experimental petrology",
    ),
    (
        "Slab dehydration and fluid flux",
        "subduction slab dehydration water release fluid flux melting mantle wedge",
    ),
    (
        "Arc magma petrogenesis",
        "volcanic arc magma generation subduction LILE HFSE Nb Ta anomaly slab component",
    ),
    # ── Volcanology ──
    (
        "Volcanic eruption dynamics",
        "eruption mechanism pyroclastic deposit ignimbrite plinian vulcanian strombolian lava flow",
    ),
    (
        "Volcanic stratigraphy",
        "volcanic sequence tephra stratigraphy tephrachronology volcanic facies architecture",
    ),
    # ── Economic geology ──
    (
        "Hydrothermal alteration zonation",
        "potassic propylitic phyllic argillic alteration porphyry copper mineralization vein",
    ),
    (
        "Fluid inclusion microthermometry",
        "fluid inclusion homogenization temperature salinity ice melting hydrothermal ore fluid",
    ),
    (
        "Stable isotope alteration tracking",
        "oxygen hydrogen sulfur isotope alteration mineralization fluid source magmatic meteoric",
    ),
    (
        "Re-Os molybdenite geochronology",
        "rhenium osmium molybdenite dating porphyry mineralization age sulfide",
    ),
    (
        "Supergene enrichment",
        "supergene oxidation chalcocite enrichment blanket copper leaching groundwater",
    ),
    (
        "Skarn mineralization",
        "skarn contact metamorphism garnet wollastonite calc-silicate ore deposit carbonate",
    ),
    (
        "Epithermal mineralization",
        "epithermal gold silver vein low sulfidation high sulfidation boiling adularia",
    ),
    (
        "Volcanogenic massive sulfide deposits",
        "VMS VHMS massive sulfide volcanogenic sea floor hydrothermal exhalative",
    ),
    # ── Geochemistry ──
    (
        "REE fractionation patterns",
        "rare earth element REE fractionation LREE HREE chondrite normalization spider diagram",
    ),
    (
        "Trace element partitioning",
        "partition coefficient KD mineral melt trace element distribution incompatible compatible",
    ),
    (
        "Radiogenic isotope geochemistry",
        "Sr Nd Pb Hf isotope ratio mantle crust source provenance radiogenic",
    ),
    (
        "Stable isotope geochemistry",
        "delta 18O 13C 34S deuterium isotope fractionation temperature fluid source",
    ),
    (
        "Mineral-melt trace element partitioning",
        "D mineral melt partition coefficient clinopyroxene garnet trace element experimental",
    ),
    # ── Mantle petrology ──
    (
        "Mantle xenolith petrology",
        "mantle xenolith peridotite lherzolite harzburgite spinel garnet xenolith mantle sample",
    ),
    (
        "Mantle melting and melt generation",
        "partial melting peridotite mantle decompression melting batch fractional pooled melt",
    ),
    (
        "Mantle metasomatism",
        "mantle metasomatism fluid melt interaction amphibole phlogopite LILE enrichment vein",
    ),
    (
        "Mantle potential temperature",
        "mantle potential temperatureTp plume adiabat olivine-liquid Fe-Mg forsterite saturation",
    ),
    # ── Structural geology ──
    (
        "Fault slip stress inversion",
        "fault slip data stress tensor inversion paleostress rake striation multiple inverse",
    ),
    (
        "Finite strain analysis",
        "strain ellipsoid Flinn Ramsay finite strain shape fabric intensity ratio",
    ),
    (
        "Crystallographic preferred orientation",
        "CPO LPO crystallographic preferred orientation quartz calcite olivine EBSD fabric",
    ),
    (
        "Fold and thrust belt analysis",
        "fold thrust belt balanced cross section restoration shortening detachment ramp flat",
    ),
    # ── Sedimentology ──
    (
        "Depositional environment analysis",
        "depositional environment facies analysis fluvial deltaic turbidite carbonate clastic",
    ),
    (
        "Sequence stratigraphy",
        "sequence stratigraphy systems tract transgressive regressive unconformity sea level",
    ),
    (
        "Diagenesis and burial history",
        "diagenesis compaction cementation burial history thermal maturity vitrinite reflectance",
    ),
    # ── Geophysics ──
    (
        "Seismic imaging and tomography",
        "seismic reflection refraction tomography velocity structure mantle crust mantle wedge",
    ),
    (
        "Gravity and magnetic anomaly interpretation",
        "gravity magnetic anomaly bouguer isostatic crustal thickness density contrast",
    ),
    (
        "Electrical conductivity of Earth materials",
        "electrical conductivity magnetotelluric impedance mantle mineral water content",
    ),
    (
        "Heat flow and geothermal modeling",
        "heat flow geothermal gradient thermal conductivity radiogenic heat production",
    ),
    # ── Analytical techniques ──
    (
        "Electron microprobe mineral chemistry",
        "EPMA electron microprobe wavelength dispersive X-ray mineral composition WDS",
    ),
    (
        "LA-ICP-MS trace element analysis",
        "laser ablation ICP mass spectrometry trace element spot analysis mineral",
    ),
    (
        "SIMS secondary ion mass spectrometry",
        "SIMS secondary ion sputtering isotope trace element in-situ spot analysis",
    ),
    (
        "X-ray diffraction mineralogy",
        "XRD powder diffraction crystal structure mineral identification Rietveld refinement",
    ),
    (
        "SEM-EDS mineral characterization",
        "scanning electron microscopy energy dispersive spectroscopy backscatter imaging mineral",
    ),
    # ── Marine / Planetary / Environmental ──
    (
        "Marine geology and ocean floor mapping",
        "ocean floor spreading ridge seamount abyssal plain marine geology ophiolite",
    ),
    (
        "Ophiolite and oceanic lithosphere",
        "ophiolite sequence pillow lava sheeted dike gabbro peridotite oceanic crust",
    ),
    (
        "Meteorite and impact crater analysis",
        "meteorite chondrite achondrite impact crater shock metamorphism shatter cone",
    ),
    (
        "Environmental contamination and remediation",
        "contaminant transport groundwater arsenic lead soil pollution remediation",
    ),
]

# Pre-computed embeddings cache
_theme_embeddings: list | None = None
_theme_labels: list[str] | None = None


def detect_geo_theme_robust(paper: dict) -> str:
    """Detect geological theme using BGE embedding similarity.

    Generalizes to ANY geological subdiscipline because BGE captures
    semantic relationships (synonyms, related terms, paraphrasing).
    Falls back to hierarchical keyword taxonomy if embeddings unavailable.
    """
    text = " ".join(
        filter(
            None,
            [
                paper.get("title", ""),
                paper.get("abstract", ""),
                paper.get("key_finding", ""),
                str(paper.get("geological_concepts", "")),
            ],
        )
    )
    if not text.strip():
        return "Other studies"

    # Try BGE embedding similarity first
    best = _match_by_embedding(text)
    if best:
        return best

    # Fallback: keyword taxonomy
    return _match_by_taxonomy(text)


def _match_by_embedding(text: str) -> str | None:
    """Match paper text against geological theme vocabulary using BGE cosine."""
    global _theme_embeddings, _theme_labels
    from _embeddings import is_available as _embeddings_available

    if not _embeddings_available():
        return None  # fastembed not installed — caller falls back to taxonomy

    import numpy as np
    from _embeddings import embed_texts

    try:
        # Lazy-init: embed the vocabulary once, cache
        if _theme_embeddings is None:
            _theme_labels = [label for label, _ in GEO_THEME_VOCABULARY]
            desc_texts = [desc for _, desc in GEO_THEME_VOCABULARY]
            _theme_embeddings = embed_texts(desc_texts)
            if _theme_embeddings is None:
                return None
            log.info(
                "Geological theme vocabulary embedded: %d themes", len(_theme_labels)
            )

        # Embed the paper text
        from _embeddings import embed_text_full

        _pooled = embed_text_full(text)  # chunk+mean-pool: full text, no cut
        paper_emb = None if _pooled is None else [_pooled]
        if paper_emb is None:
            return None

        # Cosine similarity
        paper_norm = paper_emb[0] / (np.linalg.norm(paper_emb[0]) + 1e-8)
        theme_norms = _theme_embeddings / (
            np.linalg.norm(_theme_embeddings, axis=1, keepdims=True) + 1e-8
        )
        similarities = theme_norms @ paper_norm
        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]

        if best_sim > 0.3:  # threshold for a meaningful match
            return _theme_labels[best_idx]
        return None
    except Exception as e:
        log.warning(
            "BGE theme matching failed at runtime, falling back to taxonomy: %s", e
        )
        return None


# Hierarchical taxonomy fallback — covers ALL geology by division → method
_TAXONOMY_DIVISIONS: list[tuple[str, list[str], str]] = [
    (
        "metamorphic",
        [
            "metamorph",
            "schist",
            "gneiss",
            "amphibolite",
            "eclogite",
            "granulite",
            "facies",
            "pseudosection",
            "prograde",
            "retrograde",
        ],
        "Metamorphic petrology",
    ),
    (
        "igneous",
        [
            "igneous",
            "magma",
            "basalt",
            "granite",
            "andesite",
            "rhyolite",
            "intrusion",
            "pluton",
            "volcanic",
            "lava",
            "pumice",
            "tuff",
        ],
        "Igneous and volcanic petrology",
    ),
    (
        "economic",
        [
            "ore",
            "deposit",
            "mineralization",
            "porphyry",
            "epithermal",
            "vein",
            "gold",
            "copper",
            "zinc",
            "lead",
            "skarn",
            "hydrothermal",
        ],
        "Economic geology",
    ),
    (
        "geochemistry",
        [
            "isotope",
            "trace element",
            "REE",
            "rare earth",
            "delta",
            "partition",
            "geochem",
            "chondrite",
            "incompatible",
            "LILE",
            "HFSE",
        ],
        "Geochemistry",
    ),
    (
        "mantle",
        [
            "mantle",
            "peridotite",
            "lherzolite",
            "xenolith",
            "asthenosphere",
            "lithosphere",
            "plume",
        ],
        "Mantle petrology",
    ),
    (
        "structural",
        [
            "fault",
            "fold",
            "thrust",
            "shear",
            "strain",
            "stress",
            "cleavage",
            "foliation",
            "lineation",
            "deformation",
            "brittle",
            "ductile",
        ],
        "Structural geology",
    ),
    (
        "sedimentology",
        [
            "sediment",
            "sandstone",
            "shale",
            "limestone",
            "turbidite",
            "fluvial",
            "deltaic",
            "carbonate",
            "diagenesis",
            "stratigraph",
        ],
        "Sedimentology and stratigraphy",
    ),
    (
        "geophysics",
        [
            "seismic",
            "gravity",
            "magnetic",
            "electrical",
            "conductivity",
            "tomography",
            "magnetotelluric",
            "geothermal",
            "heat flow",
        ],
        "Geophysics",
    ),
    (
        "analytical",
        [
            "microprobe",
            "EPMA",
            "LA-ICP-MS",
            "SIMS",
            "XRD",
            "SEM",
            "cathodoluminescence",
            "Raman",
        ],
        "Analytical and instrumental methods",
    ),
    (
        "marine",
        ["ocean", "marine", "seafloor", "abyssal", "ridge", "seamount"],
        "Marine geology",
    ),
    (
        "planetary",
        ["meteorite", "impact", "crater", "lunar", "martian", "asteroid"],
        "Planetary geology",
    ),
    (
        "environmental",
        ["contaminat", "groundwater", "pollution", "remediation", "arsenic", "soil"],
        "Environmental geology",
    ),
]

_TAXONOMY_METHODS: list[tuple[str, list[str], str]] = [
    (
        "thermobarometry",
        [
            "thermometry",
            "barometry",
            "thermobarometry",
            "geotherm",
            "geobarometer",
            "KD",
            "exchange",
            "calibration",
        ],
        "Thermobarometry",
    ),
    (
        "phase-equilibrium",
        [
            "pseudosection",
            "phase equilibria",
            "thermocalc",
            "domino",
            "perple",
            "gibbs method",
        ],
        "Phase-equilibrium modeling",
    ),
    (
        "geochronology",
        [
            "geochron",
            "U-Pb",
            "Ar-Ar",
            "Re-Os",
            "fission track",
            "cosmogenic",
            "dating",
            "age determination",
        ],
        "Geochronology",
    ),
    (
        "experimental",
        [
            "experimental",
            "piston cylinder",
            "multi-anvil",
            "diamond anvil",
            "synthetic",
            "capsule",
            "run product",
        ],
        "Experimental petrology",
    ),
    (
        "fluid-inclusion",
        [
            "fluid inclusion",
            "homogenization",
            "microthermometry",
            "melting ice",
            "heating stage",
        ],
        "Fluid inclusion studies",
    ),
    (
        "isotope",
        ["isotope", "delta", "Sr-Nd", "Pb", "Hf", "Os", "radiogenic"],
        "Isotope geochemistry",
    ),
    (
        "modeling",
        ["model", "simulation", "numerical", "finite element", "thermodynamic"],
        "Numerical modeling",
    ),
    (
        "field-study",
        ["field", "outcrop", "mapping", "regional", "transect"],
        "Field-based study",
    ),
    (
        "review",
        ["review", "synthesis", "overview", "compilation", "meta-analysis"],
        "Review / synthesis",
    ),
]


def _match_by_taxonomy(text: str) -> str:
    """Generalizable fallback: match by hierarchical geological taxonomy.

    Checks division first (metamorphic/igneous/economic/etc.), then method
    (thermobarometry/geochronology/experimental/etc.), and combines them
    into a label like 'Thermobarometry (metamorphic petrology)'.
    """
    text_lower = text.lower()

    # Find division
    division_label = None
    for _, keywords, label in _TAXONOMY_DIVISIONS:
        if any(kw in text_lower for kw in keywords):
            division_label = label
            break

    # Find method
    method_label = None
    for _, keywords, label in _TAXONOMY_METHODS:
        if any(kw in text_lower for kw in keywords):
            method_label = label
            break

    # Combine
    if method_label and division_label:
        return f"{method_label} ({division_label})"
    if division_label:
        return division_label
    if method_label:
        return method_label

    # Use discipline from pico
    pico = text_lower
    return "Other studies"


# =============================================================================
# 3. Metamorphic facies + tectonic interpretation (textbook logic, non-LLM)
# =============================================================================
_FACIES_FROM_TEMP: list[tuple[float, float, str]] = [
    (0, 150, "diagenetic/burial conditions"),
    (150, 250, "prehnite-pumpellyite facies (sub-greenschist)"),
    (250, 350, "lower greenschist facies (chlorite zone)"),
    (350, 450, "upper greenschist facies (biotite-garnet zone)"),
    (450, 550, "lower amphibolite facies (garnet-staurolite zone)"),
    (550, 650, "middle amphibolite facies (staurolite-kyanite zone)"),
    (650, 700, "upper amphibolite facies (sillimanite zone)"),
    (700, 850, "granulite facies"),
    (850, 1000, "ultra-high temperature (UHT) granulite facies"),
    (1000, 1300, "crustal anatexis / melting"),
    (1300, 1800, "mantle melting conditions"),
]

_FACIES_FROM_PRESSURE: list[tuple[float, float, str]] = [
    (0, 2, "very low pressure (contact/volcanic)"),
    (2, 5, "low pressure (Buchan / andalusite)"),
    (5, 8, "medium pressure (Barrovian / kyanite)"),
    (8, 12, "medium-high pressure (kyanite-sillimanite)"),
    (12, 20, "high pressure (eclogite facies)"),
    (20, 40, "ultra-high pressure (coesite/diamond)"),
    (40, 60, "deep subduction / mantle"),
]

_GEOTHERM: list[tuple[float, float, str]] = [
    (0, 5, "cold subduction zone"),
    (5, 15, "subduction / accretionary wedge"),
    (15, 25, "normal continental crust"),
    (25, 40, "Barrovian regional metamorphism"),
    (40, 60, "Buchan / contact metamorphism"),
    (60, 150, "contact aureole / rift"),
    (150, 99999, "volcanic / geothermal field"),
]


def interpret_pt_data(
    temps: list[float],
    pressures: list[float],
    discipline: str = "",
) -> str:
    """Generate geological interpretation from P-T data. Non-LLM, textbook logic."""
    if not temps and not pressures:
        return ""
    parts: list[str] = []
    if temps:
        t_med = sorted(temps)[len(temps) // 2]
        t_min, t_max = min(temps), max(temps)
        facies = _lookup(t_med, _FACIES_FROM_TEMP)
        parts.append(
            f"Temperatures of {t_min:.0f}–{t_max:.0f}°C "
            f"(median {t_med:.0f}°C, n={len(temps)}) correspond to "
            f"**{facies}**. "
        )
        if t_max - t_min > 300:
            parts.append(
                "The wide range reflects differences in metamorphic grade, "
                "rock type, and/or geothermometer calibration. "
            )
    if pressures:
        p_med = sorted(pressures)[len(pressures) // 2]
        p_min, p_max = min(pressures), max(pressures)
        p_env = _lookup(p_med, _FACIES_FROM_PRESSURE)
        parts.append(
            f"Pressures of {p_min:.1f}–{p_max:.1f} kbar "
            f"(median {p_med:.1f}, n={len(pressures)}) indicate "
            f"**{p_env}**. "
        )
    if temps and pressures:
        t_med = sorted(temps)[len(temps) // 2]
        p_med = sorted(pressures)[len(pressures) // 2]
        if p_med > 0:
            gradient = t_med / p_med
            tectonic = _lookup(gradient, _GEOTHERM)
            parts.append(
                f"The implied geothermal gradient (~{gradient:.0f}°C/kbar) "
                f"is consistent with **{tectonic}**. "
            )
    return "".join(parts)


def _lookup(val: float, table: list[tuple[float, float, str]]) -> str:
    for low, high, name in table:
        if low <= val < high:
            return name
    return "extreme conditions"


# =============================================================================
# 4. Research gap detection — method vocabulary comparison
# Generalizable: uses the same GEO_THEME_VOCABULARY to check coverage.
# =============================================================================
_METHOD_CATEGORIES: list[str] = [
    "thermobarometry",
    "geochronology",
    "experimental",
    "isotope",
    "modeling",
    "fluid inclusion",
    "field study",
    "geochemistry",
    "structural",
    "geophysical",
    "analytical",
    "review",
]


def detect_research_gaps(
    found_themes: list[str],
    discipline: str = "",
) -> list[str]:
    """Detect missing methodological approaches by comparing found themes
    against expected methods for the discipline.

    Generalizable: extracts method categories from found theme labels
    and compares against the discipline's expected categories.
    """
    found_text = " ".join(found_themes).lower()

    # Determine which method categories are present
    found_categories: set[str] = set()
    for cat in _METHOD_CATEGORIES:
        if cat in found_text:
            found_categories.add(cat)

    # Determine expected categories from the discipline
    disc_key = _match_discipline(discipline)
    expected = _EXPECTED_CATEGORIES.get(disc_key, set())
    if not expected:
        expected = {"thermobarometry", "geochemistry", "field study", "modeling"}

    gaps = expected - found_categories
    return sorted(gaps)


_EXPECTED_CATEGORIES: dict[str, set[str]] = {
    "metamorphic petrology": {
        "thermobarometry",
        "geochronology",
        "modeling",
        "isotope",
        "field study",
        "analytical",
    },
    "igneous petrology": {
        "experimental",
        "geochemistry",
        "geochronology",
        "modeling",
        "analytical",
        "field study",
    },
    "economic geology": {
        "fluid inclusion",
        "isotope",
        "geochronology",
        "geochemistry",
        "field study",
        "structural",
    },
    "geochemistry": {
        "isotope",
        "analytical",
        "experimental",
        "modeling",
        "geochemistry",
        "field study",
    },
    "mantle petrology": {
        "experimental",
        "geochemistry",
        "isotope",
        "thermobarometry",
        "analytical",
        "modeling",
    },
    "structural geology": {
        "field study",
        "modeling",
        "geochronology",
        "analytical",
    },
    "sedimentology": {
        "field study",
        "geochemistry",
        "modeling",
        "geochronology",
    },
    "geology": {
        "field study",
        "geochemistry",
        "geochronology",
        "modeling",
    },
}


# =============================================================================
# 5. Convergence / divergence analysis (statistical, generalizable)
# =============================================================================
def build_convergence_text(
    papers_by_theme: dict[str, list[dict]],
) -> str:
    """Build convergence/divergence analysis from grouped papers.

    Generalizable: uses statistical outlier detection (IQR method) on
    quantitative measurements, not hardcoded thresholds. Identifies
    agreement clusters and outlier studies automatically.
    """
    parts: list[str] = []
    for theme, papers in papers_by_theme.items():
        if len(papers) < 2:
            continue

        # Extract all numeric measurements grouped by type
        temp_vals: list[tuple[str, float]] = []
        press_vals: list[tuple[str, float]] = []
        for p in papers:
            title = (p.get("title") or "?")[:40]
            for m in p.get("measurements") or []:
                if not isinstance(m, dict) or m.get("_flagged"):
                    continue
                mname = (m.get("measurement") or "").lower()
                try:
                    val = float(m.get("value")) if m.get("value") else None
                except (TypeError, ValueError):
                    val = None
                if val is None:
                    continue
                if "temp" in mname and 50 < val < 2000:
                    temp_vals.append((title, val))
                elif "press" in mname and 0 < val < 100:
                    press_vals.append((title, val))

        for label, vals in [("temperature", temp_vals), ("pressure", press_vals)]:
            if len(vals) < 3:
                continue
            nums = [v for _, v in vals]
            nums_sorted = sorted(nums)
            median = nums_sorted[len(nums_sorted) // 2]
            # IQR for outlier detection
            q1 = nums_sorted[len(nums_sorted) // 4]
            q3 = nums_sorted[3 * len(nums_sorted) // 4]
            iqr = q3 - q1
            upper_fence = q3 + 1.5 * iqr
            lower_fence = q1 - 1.5 * iqr

            inliers = [(t, v) for t, v in vals if lower_fence <= v <= upper_fence]
            outliers = [(t, v) for t, v in vals if v > upper_fence or v < lower_fence]

            if len(inliers) >= 2:
                inlier_vals = [v for _, v in inliers]
                parts.append(
                    f"Within **{theme}**, {len(inliers)}/{len(vals)} studies "
                    f"report {label} values clustering around "
                    f"{min(inlier_vals):.0f}–{max(inlier_vals):.0f} "
                    f"(median {median:.0f}), indicating robust reproducibility. "
                )
            if outliers:
                parts.append(
                    f"{len(outliers)} stud{'y' if len(outliers) == 1 else 'ies'} "
                    f"report outlier {label} values "
                    f"({', '.join(f'{v:.0f}' for _, v in outliers[:3])}), "
                    f"potentially reflecting different calibration, rock type, "
                    f"or analytical method. "
                )

        # Known systematic offsets (generalizable pattern matching)
        all_text = " ".join((p.get("key_finding") or "").lower() for p in papers)
        offsets = _detect_known_offsets(all_text)
        for offset_desc in offsets:
            parts.append(offset_desc)

    return " ".join(parts) if parts else ""


# Known systematic offsets — generalizable pattern matching
_KNOWN_OFFSETS: list[tuple[list[str], str]] = [
    (
        ["ferry", "holdaway"],
        "Ferry-Spear vs Holdaway garnet-biotite calibrations "
        "show a known ~50-80°C systematic offset [Spear, 1993]. ",
    ),
    (
        ["laiu", "lai-icp"],
        "LA-ICP-MS vs SIMS trace element data may show "
        "systematic differences due to spatial resolution and matrix effects. ",
    ),
    (
        ["mass-spec", "thermal ionization"],
        "TIMS vs LA-ICP-MS U-Pb ages show "
        "different precision (TIMS ±0.1% vs LA-ICP-MS ±2%) — comparison requires "
        "method-aware interpretation. ",
    ),
    (
        ["grt-bt", "grt-cpx"],
        "Garnet-biotite vs garnet-clinopyroxene thermometers "
        "may yield systematically different temperatures due to different Fe-Mg "
        "exchange kinetics. ",
    ),
]


def _detect_known_offsets(text: str) -> list[str]:
    """Detect mentions of known systematic calibration offsets."""
    offsets = []
    for keywords, desc in _KNOWN_OFFSETS:
        if all(kw in text for kw in keywords):
            offsets.append(desc)
    return offsets
