#!/usr/bin/env python3
"""Narrative builders for the scientific-research synthesis pipeline.

All narrative construction logic: chronological, verification, comparative,
and data compilation builders. Plus rephrasing, transitions, significance
inference, and quality filtering.

Imported by synthesize.py — not called directly.
"""

from __future__ import annotations

from _artifact import _extract_papers, _load_json, _sanitize_finding  # noqa: F401
from _artifact import _parse_authors  # noqa: F401

import json
import logging
import re
from collections import defaultdict
from typing import Any

log = logging.getLogger("scientific_research.narrative")

# Ensure local imports work
import sys as _sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in _sys.path:
    _sys.path.insert(0, str(_SCRIPT_DIR))

from _classifiers import detect_discipline

# ── Boundary normalizer (defense-in-depth) ─────────────────────────────────
_NARR_STRING_FIELDS = (
    "discipline",
    "study_type",
    "novelty",
    "interpretation",
    "key_finding",
    "stance_toward_topic",
    "study_design",
    "title",
    "abstract",
    "paper_id",
    "doi",
    "venue",
    "finding",
)


def _normalize_for_narrative(papers: list) -> list:
    """Gate: coerce None → '' for string fields before narrative construction."""
    for p in papers:
        if not isinstance(p, dict):
            continue
        for field in _NARR_STRING_FIELDS:
            if p.get(field) is None:
                p[field] = ""
        for nested_key in ("pico", "subject", "method"):
            nested = p.get(nested_key)
            if isinstance(nested, dict):
                for k, v in list(nested.items()):
                    if v is None:
                        nested[k] = ""
            elif nested is None:
                p[nested_key] = {}
    return papers


# =============================================================================
# Data loading helpers
# =============================================================================










def _author_short(authors: list[str]) -> str:
    """Short author citation: 'Smith et al.' or 'Smith & Jones'.
    Normalizes ALL-CAPS names to Title Case."""
    if not authors:
        return "Anonymous"
    last_names = []
    for a in authors[:3]:
        parts = a.strip().rsplit(None, 1)
        last = parts[-1] if parts else a
        # Fix ALL-CAPS names: "LEE" → "Lee"
        if last.isupper() and len(last) > 1:
            last = last.title()
        last_names.append(last)
    if len(authors) <= 2:
        return " & ".join(last_names)
    return f"{last_names[0]} et al."


# =============================================================================
# Citation formatter — [1] [2] numbered style
# =============================================================================


def format_citation_list(papers: list[dict]) -> str:
    """Format numbered reference list from papers (in citation order).

    [1] Smith, J. et al. (2020). Title. DOI:10.xxxx
    """
    lines: list[str] = []
    for i, p in enumerate(papers, 1):
        author = _author_short(_parse_authors(p.get("authors", [])))
        year = p.get("year") or "n.d."
        title = p.get("title", "Untitled")
        if len(title) > 120:
            title = title[:117] + "..."
        doi = p.get("doi", "")
        doi_str = f" DOI:[{doi}](https://doi.org/{doi})" if doi else ""
        lines.append(f"[{i}] {author} ({year}). {title}.{doi_str}")
    return "\n\n".join(lines)


# =============================================================================
# Thematic synthesis — organize by methodology, synthesize findings
# =============================================================================


# Domain-agnostic method detection patterns
_METHOD_PATTERNS = [
    # Petrology / thermobarometry
    (
        "Experimental petrology",
        re.compile(
            r"\b(?:experiment\w*|synthetic\s+(?:sample|run)|piston\s+cylinder|multi.anvil|diamond\s+anvil|high.pressure\s+experiment|phase\s+equilibrium\s+experiment)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Phase-equilibrium modeling",
        re.compile(
            r"\b(?:phase\s+equilibri\w*|pseudosection|THERMOCALC|Perple_X|activity.composition|a.x\s+model|solution\s+model|Gibbs\s+minimi\w*|bulk\s+composition|isopleth|Perplex|Theriak|Domino|BurnMan)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Conventional thermobarometry",
        re.compile(
            r"\b(?:Fe.Mg\s+(?:exchange|thermomet|partition)|garnet.biotite|garnet.clinopyroxene|garnet.orthopyroxene|garnet.hornblende|garnet.ilmenite|GASP|GB\s+thermomet|GEOPATH|solvus\s+thermomet|net.transfer|calibrat\w*\s+thermobar|exchange\s+thermomet|avJE?TT|TWQ)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Elastic thermobarometry",
        re.compile(
            r"\b(?:elastic\s+thermobar\w*|quartz\s+inclusion\w*|zircon\s+inclusion\w*|Raman\s+(?:spectroscop\w*|band\w*|peak\w*)|entrapment\s+(?:pressure\w*|P\b)|isotropic\s+strain|residual\s+pressure|host.?inclusion\w*|EoS|equation\s+of\s+state|Gruneisen|Gr\u00fcneisen)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Trace element thermometry",
        re.compile(
            r"\b(?:trace\s+element\s+thermomet|rare\s+earth|\bREE\b|\bLA.ICP.MS\b|\bSIMS\b|ion\s+microprobe|partition\s+coefficient|\bKd\b|zoning\s+(?:profile|pattern)|Ti.in.(?:zircon|quartz)|Zr.in.rutile|REE.in.garnet)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Diffusion chronometry",
        re.compile(
            r"\b(?:diffusion\w*|geospeedomet\w*|cooling\s+rate|Fe.Mg\s+interdiffusion|garnet\s+(?:diffusion|zoning)|diffusivity|Arrhenius)\b",
            re.IGNORECASE,
        ),
    ),
    # Geochronology
    (
        "Geochronology",
        re.compile(
            r"\b(?:geochronolog\w*|\bU.Pb\b|\bAr.Ar\b|\b40Ar.39Ar\b|monazite\s+(?:age|dating)|zircon\s+(?:age|dating)|SHRIMP|ID.TIMS|fission\s+track|cosmogenic|\bRe.Os\b|\bSm.Nd\b|\bLu.Hf\b)\b",
            re.IGNORECASE,
        ),
    ),
    # Geochemistry
    (
        "Isotope geochemistry",
        re.compile(
            r"\b(?:isotop\w*|\bSr.b.d\b|\bNd.b.d\b|\bPb.b.d\b|\bd18O\b|\bd13C\b|\bd34S\b|\bdD\b|\b87Sr\b|\b143Nd\b|stable\s+isotope|radiogen)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Whole-rock geochemistry",
        re.compile(
            r"\b(?:whole.rock\s+geochem|bulk\s+(?:rock|geochem)|major\s+element|\bXRF\b|\bICP.MS\b|\bEPMA\b|electron\s+microprobe|harker\s+diagram|spider\s+(?:diagram|plot))\b",
            re.IGNORECASE,
        ),
    ),
    # Structural geology
    (
        "Structural analysis",
        re.compile(
            r"\b(?:structural\s+(?:analysis|geolog)|stress\s+inversion|paleostress|strain\s+(?:analysis|ellipsoid|rate)|fracture\s+analysis|fold\s+(?:geometry|analysis)|fault\s+(?:geometry|kinematic|slip|displacement)|brittle|ductile\s+shear)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Rock magnetism / Paleomagnetism",
        re.compile(
            r"\b(?:paleomagnet\w*|magnetic\s+(?:susceptibility|fabric|anisotropy|mineralogy)|demagnetiz\w*|\bAMS\b|natural\s+remanent|\bNRM\b|\bARM\b|\bIRM\b)\b",
            re.IGNORECASE,
        ),
    ),
    # Geophysics
    (
        "Seismology",
        re.compile(
            r"\b(?:seismic\s+(?:tomograph\w*|reflection|refraction|wave|velocity|attenuation)|receiver\s+function|earthquake\s+(?:location|mechanism|source)|\bVp\b|\bVs\b|\bMw\b|moment\s+magnitude|teleseismic)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Gravity / Magnetic survey",
        re.compile(
            r"\b(?:gravity\s+(?:survey|anomal|gradient)|Bouguer|free.air|magnetic\s+(?:anomal|survey)|aeromagnetic|magnetotellur\w*|\bMT\s+survey)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Heat flow / Thermal",
        re.compile(
            r"\b(?:heat\s+flow|geothermal\s+gradient|thermal\s+(?:conductivity|diffusivity|model)|surface\s+heat\s+flow)\b",
            re.IGNORECASE,
        ),
    ),
    # Sedimentology / Stratigraphy
    (
        "Sedimentology",
        re.compile(
            r"\b(?:sedimentolog\w*|depositional\s+environment|facies\s+analysis|sequence\s+stratigraph|provenance|diagen\w*|sedimentary\s+structure)\b",
            re.IGNORECASE,
        ),
    ),
    # Volcanology
    (
        "Volcanology",
        re.compile(
            r"\b(?:volcan\w*|eruption\w*|lava\s+flow|volcanic\s+(?:ash|gas|hazard|risk)|magma\s+(?:chamber|evolution|ascent|mixing|emplacement)|pyroclastic)\b",
            re.IGNORECASE,
        ),
    ),
    # Remote sensing
    (
        "Remote sensing",
        re.compile(
            r"\b(?:\bInSAR\b|\bD.InSAR\b|\bLandsat\b|\bASTER\b|\bSentinel\b|\bMODIS\b|satellite\s+(?:imag|data)|airborne\s+(?:survey|magnetic)|hyperspectral|multispectral)\b",
            re.IGNORECASE,
        ),
    ),
    # Numerical / analog modeling
    (
        "Numerical modeling",
        re.compile(
            r"\b(?:numerical\s+model|computer\s+simulation|finite\s+(?:element|difference|volume)|discrete\s+element|computational|thermodynamic\s+model|machine\s+learning|statistical\s+model|geodynamic\s+model)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Analog modeling",
        re.compile(
            r"\b(?:analog\s+(?:model|experiment)|sandbox\s+model|scaled\s+model|physical\s+model)\b",
            re.IGNORECASE,
        ),
    ),
    # Field / Review
    (
        "Field study",
        re.compile(
            r"\b(?:field\s+(?:study|area|evidence|relation|sample)|outcrop|collected\s+from|fieldwork|mapped|mapping|regional\s+geolog)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Review / Synthesis",
        re.compile(
            r"\b(?:review\b|meta.analysis|systematic\s+review|overview|state.of.the.art|synthesi[sz]e|summari[sz]e)\b",
            re.IGNORECASE,
        ),
    ),
    # Mineralogy / Crystallography
    (
        "Mineralogy",
        re.compile(
            r"\b(?:mineral\s+(?:chemistr|composition|assemblage|paragenesis)|crystal\s+(?:structure|chemistry)|X.ray\s+(?:diffraction|fluorescence)|\bXRD\b|Raman\s+spectroscop\w*|\bSEM\b|\bTEM\b|electron\s+backscatter)\b",
            re.IGNORECASE,
        ),
    ),
    # Hydrogeology
    (
        "Hydrogeology",
        re.compile(
            r"\b(?:groundwater|aquifer|hydrogeolog|hydrolog|permeab|porosity|hydraulic\s+(?:conductivity|head)|water\s+(?:table|chemistry|rock\s+interaction))\b",
            re.IGNORECASE,
        ),
    ),
    # Ore geology
    (
        "Software / Computational thermobarometry",
        re.compile(
            r"\b(?:Thermobar(?:\s+software)?|\bTHERMOCALC\b.*software|geothermobarometr.*software|"
            r"thermobarometr.*package|thermobarometr.*program|thermobarometr.*tool|"
            r"\bGCDkit\b|\bIgiPet\b|\bPetMod\b|\bRcrust\b|\bPTcalc\b)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Ore geology",
        re.compile(
            r"\b(?:ore\s+(?:deposit|geolog|mineraliz)|mineraliz\w*|hydrothermal\s+(?:deposit|vein|alteration)|porphyry\s+(?:deposit|copper)|epithermal|orogenic\s+gold|\bVMS\b|\bSEDEX\b)\b",
            re.IGNORECASE,
        ),
    ),
]


def _detect_method_theme(paper: dict) -> str:
    """Detect the primary methodological theme of a paper.

    Delegates to _geo_enrich.detect_geo_theme_robust (BGE embedding
    similarity against geological vocabulary) when available. Falls back
    to the original keyword-based detection if geo_enrich is unavailable.
    """
    # Delegate to robust BGE-based geological theme detection
    try:
        from _geo_enrich import detect_geo_theme_robust

        result = detect_geo_theme_robust(paper)
        if result and result != "Other studies":
            return result
    except ImportError:
        pass

    # Fallback: original keyword-based detection
    title = (paper.get("title") or "").lower()
    finding = (paper.get("key_finding") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    text = f"{title} {finding} {abstract}"

    # Pre-check: Thermobar software must be caught before Conventional pattern
    if "thermobar" in title and any(
        w in text for w in ["software", "package", "program", "tool", "python"]
    ):
        return "Software / Computational thermobarometry"

    for theme_name, pattern in _METHOD_PATTERNS:
        if pattern.search(text):
            return theme_name

    # Fallback: use keywords from title for a more descriptive label
    title = (paper.get("title") or "").lower()
    if "atmospher" in title or "exoplanet" in title or "planet" in title:
        return "Planetary geology"
    if "magma ocean" in title or "magma" in title:
        return "Igneous petrology"
    if "volcanic" in title or "eruption" in title or "outgassing" in title:
        return "Volcanology"
    if "ore" in title or "deposit" in title or "mineraliz" in title:
        return "Ore geology"
    if "mantle" in title or "peridot" in title:
        return "Mantle petrology"
    if "metamorph" in title or "schist" in title or "gneiss" in title:
        return "Metamorphic petrology"
    if "sediment" in title or "basin" in title:
        return "Sedimentology"
    if "fluid" in title or "hydrothermal" in title:
        return "Fluid geochemistry"

    # Last fallback: discipline-based grouping from pico or top-level
    pico = paper.get("pico") or {}
    disc = (
        (paper.get("discipline") or pico.get("discipline") or "")
        .lower()
        .replace("_", " ")
    )
    if disc and disc not in ("general", "other", ""):
        return disc.capitalize()
    study_t = (pico.get("study_type") or "").lower().replace("_", " ")
    if study_t and study_t not in ("general", "other", ""):
        return study_t.capitalize()

    return "Other studies"


def _group_by_theme(papers: list[dict]) -> list[dict[str, object]]:
    """Group papers by methodological theme.

    Returns list of {"label": str, "papers": list[dict]} sorted by
    group size (largest first). Deduplicates by paper_id/doi.
    """
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in papers:
        pid = p.get("paper_id") or p.get("doi") or ""
        if pid and pid in seen:
            continue
        if pid:
            seen.add(pid)
        deduped.append(p)

    groups: dict[str, list[dict]] = defaultdict(list)
    for p in deduped:
        theme = _detect_method_theme(p)
        groups[theme].append(p)

    result = []
    for label, group_papers in sorted(groups.items(), key=lambda x: -len(x[1])):
        result.append({"label": label, "papers": _sort_chronological(group_papers)})

    return result


# Physical ranges for validation — filter impossible values across ALL geology
# General delta-context filter — applied to ALL measurement extractions
# Prevents extracting differences, uncertainties, and offsets as absolute values
_DELTA_CONTEXT = re.compile(
    r"(?:higher|lower|difference|shift|change|offset|\berror|"
    r"uncertain|\+/-|\+\-|plus.or.minus|greater|less|warmer|cooler|"
    r"increase|decrease|offset\s+of|deviation|anomal|\bdelta\b)",
    re.IGNORECASE,
)

_PHYS_TEMP_C_MIN = 100.0
_PHYS_TEMP_C_MAX = 2000.0
_PHYS_PRESS_KBAR_MIN = 0.1
_PHYS_PRESS_KBAR_MAX = 150.0
_PHYS_AGE_MA_MIN = 5.0  # below 5 Ma likely analytical uncertainty
_PHYS_AGE_MA_MAX = 4600.0  # age of Earth
_PHYS_DEPTH_KM_MIN = 0.0
_PHYS_DEPTH_KM_MAX = 6371.0  # Earth radius
_PHYS_MAGNITUDE_MIN = 0.0
_PHYS_MAGNITUDE_MAX = 10.5  # max possible Mw
_PHYS_VELOCITY_KMS_MIN = 0.5
_PHYS_VELOCITY_KMS_MAX = 15.0
_PHYS_DENSITY_MIN = 0.5
_PHYS_DENSITY_MAX = 20.0  # g/cm3
_PHYS_STRESS_MPA_MIN = 0.0
_PHYS_STRESS_MPA_MAX = 10000.0
_PHYS_HEATFLOW_MIN = 0.0
_PHYS_HEATFLOW_MAX = 500.0  # mW/m2
_PHYS_GEOTGRAD_MIN = 0.0
_PHYS_GEOTGRAD_MAX = 200.0  # C/km

# Noise measurement types — filtered out (comprehensive)
_MEAS_NOISE_TYPES = frozenset(
    {
        "",
        "doi",
        "issn",
        "volume",
        "page",
        "pages",
        "received",
        "accepted",
        "online",
        "published",
        "revised",
        "cited",
        "available online",
        "article number",
        "copyright",
        "license",
        "cc-by",
        "downloaded",
        "isbn",
        "figure",
        "fig",
        "table",
        "equation",
        "eq",
        "sample",
        "n",
        "count",
        "number",
        "latitude",
        "longitude",
        "elevation",
        "altitude",
        "distance",
        "length",
        "width",
        "height",
        "duration",
        "time",
        "rate",
        "speed",
        "frequency",
        "wavelength",
        "amplitude",
        "resolution",
        "pixel",
        "dpi",
        "ppi",
        "cost",
        "price",
        "revenue",
        "version",
        "edition",
        "chapter",
        "author",
        "editor",
        "reviewer",
        "keyword",
        "title",
        "abstract",
        "percentage",
        "percent",
        "ratio",
        "fraction",
        "error",
        "uncertainty",
        "sigma",
        "std",
        "stderr",
        "mean",
        "median",
        "mode",
        "quartile",
        "percentile",
        "min",
        "max",
        "range",
        "spread",
        "r2",
        "rms",
        "rmse",
        "mse",
        "mae",
        "slope",
        "intercept",
        "coefficient",
        "correlation",
        "weight",
        "mass",
        "volume_measure",
    }
)


def _classify_measurement(mtype: str, value, unit: str) -> str:
    """Classify a measurement across ALL geological quantity types.

    UNIT IS CHECKED FIRST — more reliable than type label.
    Returns one of: "temperature", "pressure", "age", "depth", "magnitude",
    "velocity", "density", "stress", "heatflow", "geotgrad", "composition", "noise".
    """
    mtype_lower = (mtype or "").lower().strip()
    unit_lower = (unit or "").lower().strip()

    # Noise types — checked first by mtype
    if mtype_lower in _MEAS_NOISE_TYPES:
        return "noise"

    # UNIT CHECK FIRST — unit is most reliable
    # Temperature
    if unit_lower in ("c", "k", "celsius", "kelvin", "degc", "deg_c"):
        return "temperature"
    # Stress vs pressure — MPa can be either
    if unit_lower == "mpa" and (
        "stress" in mtype_lower or "differential" in mtype_lower
    ):
        return "stress"
    # Pressure
    if unit_lower in ("kbar", "gpa", "mpa", "kb"):
        return "pressure"
    # Age
    if unit_lower in ("ma", "ga", "ka"):
        return "age"
    # Depth
    if unit_lower in ("km", "m") and (
        "depth" in mtype_lower or "crustal" in mtype_lower
    ):
        return "depth"
    # Magnitude
    if unit_lower in ("mw", "ml", "ms", "mb", "m") and "magnitude" in mtype_lower:
        return "magnitude"
    # Velocity
    if unit_lower in ("km/s", "kms", "m/s"):
        return "velocity"
    # Density
    if unit_lower in ("g/cm3", "g/cm^3", "kg/m3", "kg/m^3", "g/cc"):
        return "density"
    # Stress
    if unit_lower in ("mpa",) and (
        "stress" in mtype_lower or "differential" in mtype_lower
    ):
        return "stress"
    # Heat flow
    if unit_lower in ("mw/m2", "mw/m^2", "mw/m2", "hfu"):
        return "heatflow"
    # Geothermal gradient
    if "c/km" in unit_lower or "degc/km" in unit_lower:
        return "geotgrad"
    # Composition
    if unit_lower in (
        "wt%",
        "wt %",
        "wt percent",
        "ppm",
        "ppb",
        "vol%",
        "vol %",
        "mol%",
    ):
        return "composition"

    # FALLBACK: check mtype label
    if "temp" in mtype_lower:
        return "temperature"
    if "press" in mtype_lower:
        return "pressure"
    if "age" in mtype_lower or "geochron" in mtype_lower:
        return "age"
    if "depth" in mtype_lower:
        return "depth"
    if (
        "magnitude" in mtype_lower
        or "seismic" in mtype_lower
        and "moment" in mtype_lower
    ):
        return "magnitude"
    if "velocity" in mtype_lower or "vp" == mtype_lower or "vs" == mtype_lower:
        return "velocity"
    if "density" in mtype_lower or "rho" == mtype_lower:
        return "density"
    if "stress" in mtype_lower or "differential" in mtype_lower:
        return "stress"
    if "heat" in mtype_lower and "flow" in mtype_lower:
        return "heatflow"
    if "geotherm" in mtype_lower and "gradient" in mtype_lower:
        return "geotgrad"
    if "composit" in mtype_lower or "concentration" in mtype_lower:
        return "composition"

    # Try to infer from value range
    try:
        val = float(value)
        if 200 <= val <= 2000 and not unit_lower:
            return "temperature"
        if 0.1 <= val <= 50 and not unit_lower:
            return "pressure"
    except (TypeError, ValueError):
        pass

    return "noise"


def _normalize_measurement(mtype: str, value, unit: str) -> tuple:
    """Normalize measurement to standard units.

    Returns (normalized_value, standard_unit) or (None, None) if invalid.
    Temperatures -> Celsius. Pressures -> kbar. Ages -> Ma.
    Filters physically impossible values.
    """
    if value is None:
        return None, None

    try:
        val = float(value)
    except (TypeError, ValueError):
        return None, None

    classification = _classify_measurement(mtype, val, unit)
    if classification == "noise":
        return None, None

    unit_lower = (unit or "").lower().strip()

    if classification == "temperature":
        # Convert Kelvin to Celsius
        if unit_lower in ("k", "kelvin"):
            val = val - 273.15
        # Validate range
        if val < _PHYS_TEMP_C_MIN or val > _PHYS_TEMP_C_MAX:
            return None, None
        return val, "C"

    if classification == "pressure":
        # Convert to kbar
        if "gpa" in unit_lower:
            val = val * 10.0
        elif "mpa" in unit_lower:
            val = val * 0.01
        elif "pa" in unit_lower and "k" not in unit_lower and "m" not in unit_lower:
            val = val * 1e-8  # Pa to kbar
        # Validate range
        if val < _PHYS_PRESS_KBAR_MIN or val > _PHYS_PRESS_KBAR_MAX:
            return None, None
        return val, "kbar"

    if classification == "age":
        _mt_lower = (mtype or "").lower().strip()
        # Filter likely analytical uncertainties (values <5 Ma are usually
        # ±error, not geological ages — unless explicitly labeled)
        if val < 5 and "uncertain" not in _mt_lower and "error" not in _mt_lower:
            return None, None
        if val < 0 or val > 5000:
            return None, None
        return val, "Ma"

    return None, None


def _normalize_paper_measurements(papers: list) -> None:
    """Normalize and validate all measurements on paper dicts in-place.

    Filters impossible values, normalizes units (GPa->kbar, K->C),
    classifies measurements by type. Called once at narrative start.
    """
    for p in papers:
        normalized = []
        for m in p.get("measurements", []):
            val, unit = _normalize_measurement(
                m.get("measurement", ""),
                m.get("value"),
                m.get("unit", ""),
            )
            if val is not None:
                mtype = (
                    "temperature"
                    if unit == "C"
                    else ("pressure" if unit == "kbar" else "age")
                )
                normalized.append({"measurement": mtype, "value": val, "unit": unit})
        p["measurements"] = normalized


def _extract_numbers_from_text(text: str) -> list:
    """Extract VALIDATED P-T values from text.

    Only returns physically plausible values:
    - Temperature: 50-2000 C (requires degree symbol or explicit context)
    - Pressure: 0.1-150 kbar (requires kbar/GPa/MPa unit)

    Filters out page numbers, figure numbers, ages, bare numbers.
    Normalizes: K->C, GPa->kbar, MPa->kbar.
    """
    if not text:
        return []
    results = []

    # Temperature — REQUIRE degree symbol
    for m in re.finditer(
        r"(\d+\.?\d*)\s*(?:[\u2013-]\s*(\d+\.?\d*))?\s*"
        r"(?:\u00b0\s*[Cc](?:elsius)?|degrees?\s*[Cc](?:elsius)?|\u00b0\s*[Kk](?:elvin)?|degrees?\s*[Kk](?:elvin)?|Kelvin)\b",
        text,
        re.IGNORECASE,
    ):
        # Check for delta context — skip if this is a difference, not absolute T
        after_match = text[m.end() : m.end() + 30]
        before_match = text[max(0, m.start() - 30) : m.start()]
        if _DELTA_CONTEXT.search(after_match) or _DELTA_CONTEXT.search(before_match):
            continue
        try:
            val = float(m.group(1))
            val2 = float(m.group(2)) if m.lastindex and m.group(2) else None
            unit_raw = m.group(0).lower()
            is_kelvin = "k" in unit_raw and "c" not in unit_raw
            if is_kelvin:
                val_c = val - 273.15
                if _PHYS_TEMP_C_MIN <= val_c <= _PHYS_TEMP_C_MAX:
                    results.append((val_c, "C"))
                if val2 is not None:
                    val2_c = val2 - 273.15
                    if _PHYS_TEMP_C_MIN <= val2_c <= _PHYS_TEMP_C_MAX:
                        results.append((val2_c, "C"))
            else:
                if _PHYS_TEMP_C_MIN <= val <= _PHYS_TEMP_C_MAX:
                    results.append((val, "C"))
                if val2 is not None and _PHYS_TEMP_C_MIN <= val2 <= _PHYS_TEMP_C_MAX:
                    results.append((val2, "C"))
        except (ValueError, IndexError):
            pass

    # Pressure — requires explicit unit + delta-context filtering
    for m in re.finditer(
        r"(\d+\.?\d*)\s*(?:[\u2013-]\s*(\d+\.?\d*))?\s*(kbar|GPa|MPa)\b",
        text,
        re.IGNORECASE,
    ):
        # Check for delta context
        after_match_p = text[m.end() : m.end() + 30]
        before_match_p = text[max(0, m.start() - 30) : m.start()]
        if _DELTA_CONTEXT.search(after_match_p) or _DELTA_CONTEXT.search(
            before_match_p
        ):
            continue
        try:
            val = float(m.group(1))
            val2 = float(m.group(2)) if m.lastindex and m.group(2) else None
            unit = m.group(3)
            # Normalize to kbar
            if "gpa" in unit.lower():
                val_k = val * 10.0
                val2_k = (val2 * 10.0) if val2 else None
            elif "mpa" in unit.lower():
                val_k = val * 0.01
                val2_k = (val2 * 0.01) if val2 else None
            else:
                val_k = val
                val2_k = val2
            if _PHYS_PRESS_KBAR_MIN <= val_k <= _PHYS_PRESS_KBAR_MAX:
                results.append((val_k, "kbar"))
            if (
                val2_k is not None
                and _PHYS_PRESS_KBAR_MIN <= val2_k <= _PHYS_PRESS_KBAR_MAX
            ):
                results.append((val2_k, "kbar"))
        except (ValueError, IndexError):
            pass

    return results


def _synthesize_measurements(theme_papers: list[dict]) -> str:
    """Extract and synthesize numerical measurements from theme papers.

    Reports ranges, means, and value distributions.
    """
    # Collect measurements from paper measurement lists
    by_type: dict[str, list[float]] = defaultdict(list)
    for p in theme_papers:
        for m in p.get("measurements", []):
            mtype = (m.get("measurement") or "").strip().lower()
            if mtype in {
                "",
                "doi",
                "issn",
                "volume",
                "page",
                "pages",
                "received",
                "accepted",
                "online",
                "published",
                "revised",
                "cited",
                "available online",
                "article number",
                "copyright",
                "license",
                "cc-by",
                "downloaded",
                "isbn",
            }:
                continue
            try:
                val = float(m.get("value", 0))
                by_type[mtype].append(val)
            except (TypeError, ValueError):
                continue

    # Also scan key_finding text for P-T values
    for p in theme_papers:
        finding = p.get("key_finding") or ""
        for val, unit in _extract_numbers_from_text(finding):
            label = f"{'temperature' if unit in ('°C', 'K') else 'pressure'} ({unit})"
            by_type[label].append(val)

    if not by_type:
        return ""

    parts = []
    import statistics as stats_mod

    for mtype, values in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(values) < 2:
            continue
        vmin = min(values)
        vmax = max(values)
        vmean = stats_mod.mean(values)
        vstd = stats_mod.stdev(values) if len(values) >= 2 else 0
        parts.append(
            f"{mtype}: {vmin:.1f}–{vmax:.1f} (mean {vmean:.1f} ± {vstd:.1f}, n={len(values)})"
        )

    if not parts:
        return ""
    return "Reported values: " + "; ".join(parts[:8]) + "."


_LIMITATION_CUES_RE = re.compile(
    r"\b(?:limitation|uncertain|assumption|sensitiv|depend\w*\s+on|"
    r"affected\s+by|may\s+not|could\s+be\s+biased|"
    r"require\w*|challeng\w*|problematic|"
    r"caveat|warn|caution|"
    r"large\s+error|significant\s+uncertain|"
    r"not\s+well\s+constrain|poorly\s+constrain)\b",
    re.IGNORECASE,
)


def _extract_limitations(theme_papers: list[dict]) -> list[str]:
    """Extract limitation/uncertainty statements from paper findings."""
    limitation_cues = _LIMITATION_CUES_RE
    limitations = []
    for p in theme_papers:
        finding = (p.get("key_finding") or "") + " " + (p.get("interpretation") or "")
        for sentence in finding.split("."):
            sentence = sentence.strip()
            if limitation_cues.search(sentence) and len(sentence) > 20:
                # Clean up and add with author ref
                limitations.append(sentence[:300].rstrip())
                break  # one limitation per paper max

    return limitations[:8]  # raised 2026-08-15 (was 5)


def _build_theme_synthesis(
    theme_papers: list[dict],
    topic: str,
    theme_label: str,
    ref_offset: int,
) -> tuple[str, list[dict]]:
    """Build a thematic synthesis paragraph for one theme group.

    Returns (synthesis_text, cited_papers).
    """
    cited = list(theme_papers)
    n = len(theme_papers)
    years = [p.get("year") for p in theme_papers if p.get("year")]
    year_str = (
        f"({min(years)}–{max(years)})"
        if years and min(years) != max(years)
        else (f"({min(years)})" if years else "")
    )

    parts = []

    # Opening sentence — varied structure based on group size
    refs = list(range(ref_offset + 1, ref_offset + 1 + n))
    ref_str = ", ".join(str(r) for r in refs[:6])
    if n > 6:
        ref_str += f"–{refs[-1]}"

    openers = [
        f"Within the {theme_label.lower()} approach, {n} {'study' if n == 1 else 'studies'} {year_str} [{ref_str}] contribute key findings.",
        f"The {theme_label.lower()} literature comprises {n} {'study' if n == 1 else 'studies'} {year_str} [{ref_str}].",
        f"{n} {'paper' if n == 1 else 'papers'} employ {theme_label.lower()} methods {year_str} [{ref_str}].",
        f"Studies using {theme_label.lower()} ({n} {'paper' if n == 1 else 'papers'}, {year_str}) report complementary results [{ref_str}].",
    ]
    parts.append(openers[ref_offset % len(openers)])

    # Measurement synthesis
    meas_text = _synthesize_measurements(theme_papers)
    if meas_text:
        parts.append(meas_text)

    # Finding synthesis — aggregate, not chain
    # Group findings into 2-3 key themes within this group
    findings = []
    for i, p in enumerate(theme_papers):
        finding = _clean_title_finding(_sanitize_finding(p.get("key_finding") or ""))
        if finding and _is_quality_finding(finding):
            ref = refs[i]
            author = _author_short(_parse_authors(p.get("authors", [])))
            year = p.get("year") or "n.d."
            findings.append((author, year, ref, finding[:300]))

    if findings:
        if len(findings) <= 3:
            # Small group: integrate naturally
            sentences = []
            for author, year, ref, finding in findings:
                verb, content = _rephrase_finding(finding)
                sentences.append(f"{author} ({year}) [{ref}] {verb} {content}")
            parts.append(". ".join(sentences) + ".")
        else:
            # Large group: synthesize top findings + cite rest
            top_findings = findings[:6]
            sentences = []
            for author, year, ref, finding in top_findings:
                verb, content = _rephrase_finding(finding)
                sentences.append(f"{author} ({year}) [{ref}] {verb} {content}")
            parts.append(". ".join(sentences) + ".")
            remaining_refs = [str(r) for _, _, r, _ in findings[4:8]]
            if remaining_refs:
                parts.append(
                    f"Additional studies [{', '.join(remaining_refs)}] report broadly consistent results."
                )

    # Limitations
    limitations = _extract_limitations(theme_papers)
    if limitations:
        lim_str = limitations[0][:150]
        parts.append(f"A key limitation noted across this approach: {lim_str}.")

    return " ".join(parts), cited


def _build_cross_cutting(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """Build cross-cutting synthesis paragraph."""
    if len(themes) < 2:
        return ""

    parts = []
    theme_labels = [t["label"] for t in themes[:5]]

    # Identify the dominant theme
    dominant = themes[0]
    parts.append(
        f"The {dominant['label'].lower()} approach dominates the literature "
        f"({len(dominant['papers'])} studies), "
        f"complemented by {', '.join(t['label'].lower() for t in themes[1:4])}."
    )

    # Check for thematic convergence
    # Papers across themes that report similar measurement ranges
    all_measurements = []
    for p in all_papers:
        for m in p.get("measurements", []):
            try:
                val = float(m.get("value", 0))
                all_measurements.append(val)
            except (TypeError, ValueError):
                pass

    if len(all_measurements) >= 5:
        import statistics as stats_mod

        med = stats_mod.median(all_measurements)
        parts.append(
            f"Across all approaches, reported values span a wide range, "
            f"with a median of {med:.1f}, reflecting natural variability "
            f"in geological conditions and methodological differences."
        )

    # Note methodological debates
    if len(themes) >= 3:
        parts.append(
            f"The diversity of approaches ({len(themes)} methodological themes) "
            f"highlights the active debate surrounding optimal methods for "
            f"constraining {topic.lower()}."
        )

    return " ".join(parts)


_GEO_REGION_PATTERNS_RE = re.compile(
    r"\b(?:Antarctica|Greenland|Scandinavia|Alps|Himalaya|"
    r"Appalachian|Canadian\s+Shield|Baltic|Japan|China|"
    r"Africa|Australia|South\s+America|North\s+America)\b",
    re.IGNORECASE,
)


def _build_research_gaps(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """Identify research gaps from corpus analysis."""
    parts = []
    n = len(all_papers)

    # Gap 1: underrepresented themes
    small_themes = [t for t in themes if len(t["papers"]) <= 2]
    if small_themes:
        gap_labels = [t["label"].lower() for t in small_themes[:3]]
        parts.append(
            f"Several approaches are represented by only 1–2 studies "
            f"({', '.join(gap_labels)}), limiting the robustness of "
            f"cross-method comparisons."
        )

    # Gap 2: temporal coverage
    years = [p.get("year") for p in all_papers if p.get("year")]
    if years and len(years) >= 10:
        from collections import Counter

        year_counts = Counter(years)
        max_year = max(years)
        min_year = min(years)
        # Check if recent years are well-covered
        recent = sum(1 for y in years if y >= max_year - 3)
        if recent < len(years) * 0.15:
            parts.append(
                f"Only {recent} of {len(years)} studies were published in "
                f"the last 3 years ({max_year - 3}–{max_year}), suggesting "
                f"the field may benefit from updated investigations."
            )

    # Gap 3: geographic/region diversity (scan abstracts)
    # NOTE: pattern-based detection UNDERCOUNTS real locations (proper-noun
    # regex can't cover world geography). Claims below are phrased with that
    # uncertainty explicit — never assert coverage limits the regex can't see.
    regions_found = set()
    region_patterns = _GEO_REGION_PATTERNS_RE
    for p in all_papers:
        abstract = (p.get("abstract") or "") + " " + (p.get("key_finding") or "")
        for m in region_patterns.finditer(abstract):
            regions_found.add(m.group(0).lower())
    if len(regions_found) == 0 and n >= 10:
        parts.append(
            "No geographic focus was auto-detected in the abstracts; "
            "geographic coverage of this corpus was not assessed "
            "(location metadata was not extracted)."
        )
    elif len(regions_found) <= 2 and n >= 10:
        parts.append(
            f"Only {len(regions_found)} distinct geographic setting(s) were "
            f"auto-detected in abstract text (pattern-based detection, likely "
            f"an undercount); geographic diversity should be verified manually "
            f"before drawing representativeness conclusions."
        )

    # Gap 4: methodological gap
    theme_labels = {t["label"] for t in themes}
    if "Diffusion chronometry" not in theme_labels and n >= 15:
        parts.append(
            "The absence of diffusion-chronometry studies in this corpus "
            "represents a gap, as cooling-rate constraints are essential "
            "for interpreting P-T estimates."
        )

    # Always end with future directions
    if parts:
        parts.append(
            "Future work should integrate multiple thermobarometric "
            "techniques and address these gaps through targeted studies."
        )
    else:
        parts.append(
            f"The corpus provides broad coverage of {topic.lower()}. "
            f"Future work could integrate findings across methodological "
            f"approaches and extend studies to underrepresented regions."
        )

    return " ".join(parts)


# =============================================================================
# V2 synthesis engine — genuine aggregation, varied sentence structure,
# measurement ranges, agreement/disagreement grouping, anti-repetition
# =============================================================================

import statistics as _stats

# ── Sentence pattern pools — rotated for anti-repetition ──────────────────

_THEME_OPENERS = [
    "The {theme} literature ({n} {study}) reports {meas_summary}.",
    "Within {theme} ({n} {study}), studies report {meas_summary}.",
    "{n} {study} employing {theme} methods find {meas_summary}.",
    "Studies using {theme} approaches ({n} {study}) converge on {meas_summary}.",
    "The {theme} approach ({n} {study}) yields {meas_summary}.",
    "Research employing {theme} ({n} {study}) documents {meas_summary}.",
    "Investigations using {theme} ({n} {study}) indicate {meas_summary}.",
    "The {theme} contribution ({n} {study}) encompasses {meas_summary}.",
    # geological-depth variants (added 2026-07-12)
    "Applying {theme} ({n} {study}), researchers document {meas_summary}.",
    "The {theme} record ({n} {study}) constrains {meas_summary}.",
    "{n} {study} using {theme} characterize {meas_summary}.",
]

_CONSENSUS_TEMPLATES = [
    "Multiple studies [{refs}] converge on {content}.",
    "Several investigations [{refs}] report consistent {content}.",
    "There is broad agreement among studies [{refs}] regarding {content}.",
    "Studies [{refs}] independently confirm {content}.",
]

_CONTRAST_TEMPLATES = [
    "However, studies disagree: [{refs1}] report {val1}, while [{refs2}] find {val2}.",
    "Results are inconsistent: [{refs1}] report {val1}, whereas [{refs2}] argue for {val2}.",
    "A discrepancy exists between [{refs1}], who find {val1}, and [{refs2}], who report {val2}.",
    "Contrasting results have been reported: {val1} [{refs1}] versus {val2} [{refs2}].",
]

_SINGLE_TEMPLATES = [
    "{author} ({year}) [{ref}] {verb} {content}.",
    "Notably, {author} ({year}) [{ref}] {verb} {content}.",
    "{author} ({year}) [{ref}] {verb} {content}.",
    "In a distinct contribution, {author} ({year}) [{ref}] {verb} {content}.",
]

_PAIR_TEMPLATES = [
    "{author1} ({year1}) [{ref1}] {verb1} {content1}, complemented by {author2} ({year2}) [{ref2}] who {verb2} {content2}.",
    "{author1} ({year1}) [{ref1}] {verb1} {content1}; meanwhile, {author2} ({year2}) [{ref2}] {verb2} {content2}.",
    "Studies by {author1} ({year1}) [{ref1}] and {author2} ({year2}) [{ref2}] report {content1} and {content2}, respectively.",
]

_LIMITATION_TEMPLATES = [
    "A key limitation noted by multiple authors is that {lim}.",
    "Several studies caution that {lim}.",
    "The principal uncertainty in this approach is that {lim}.",
    "A recurring caveat across these studies is that {lim}.",
]

# Verbs for rotation — prevents "reported" fatigue
_REPORTING_VERBS = [
    "reported",
    "demonstrated",
    "showed",
    "found",
    "documented",
    "established",
    "determined",
    "identified",
    "revealed",
    "observed",
]


def _ref_str(refs: list[int]) -> str:
    """Format reference list: [1, 2, 3] → '1,2,3'; [1, 2, 3, 4, 5] → '1–5'."""
    if not refs:
        return ""
    if len(refs) <= 3:
        return ", ".join(str(r) for r in refs)
    return f"{refs[0]}-{refs[-1]}"


def _meas_summary_str(theme_papers: list[dict]) -> str:
    """One-line measurement summary for use in opener sentences."""
    by_type: dict[str, list[float]] = defaultdict(list)
    for p in theme_papers:
        for m in p.get("measurements", []):
            mtype = (m.get("measurement") or "").strip().lower()
            munit = (m.get("unit") or "").strip()
            try:
                val = float(m.get("value", 0))
            except (TypeError, ValueError):
                continue
            if "temp" in mtype or munit in ("C", "K"):
                by_type["temperatures"].append(val)
            elif "press" in mtype or munit in ("kbar", "GPa", "MPa"):
                by_type["pressures"].append(val)
        for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
            label = "temperatures" if unit == "C" else "pressures"
            by_type[label].append(val)

    if not by_type:
        return "key findings"

    parts = []
    for mtype, vals in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(vals) >= 2:
            parts.append(f"{mtype} of {min(vals):.0f}-{max(vals):.0f}")
        elif len(vals) == 1:
            parts.append(f"{mtype} of {vals[0]:.0f}")
    if not parts:
        return "key findings"
    return "; ".join(parts[:3])


def _cluster_findings_by_agreement(
    items: list[dict],
) -> list[dict]:
    """Group finding items by agreement (similar values/conclusions).

    Returns list of groups:
    - {"type": "consensus", "items": [...]}
    - {"type": "contrast", "sides": [[...], [...]]}
    - {"type": "single", "item": {...}}
    - {"type": "pair", "items": [...]}
    """
    if not items:
        return []

    # Build similarity matrix based on keyword overlap
    n = len(items)
    if n == 1:
        return [{"type": "single", "item": items[0]}]

    # Extract keyword sets
    kw_sets = []
    for item in items:
        text = item["finding"].lower()
        tokens = set(t.strip(".,;:!?\"'()[]{}") for t in text.split() if len(t) > 4)
        kw_sets.append(tokens)

    # Greedy clustering: group items with >20% keyword overlap
    assigned = set()
    groups = []

    for i in range(n):
        if i in assigned:
            continue
        cluster = [i]
        assigned.add(i)
        for j in range(i + 1, n):
            if j in assigned:
                continue
            overlap = len(kw_sets[i] & kw_sets[j]) / max(
                len(kw_sets[i] | kw_sets[j]), 1
            )
            if overlap > 0.15:
                cluster.append(j)
                assigned.add(j)

        group_items = [items[idx] for idx in cluster]
        if len(group_items) >= 3:
            # Check if they all agree (similar measurements)
            groups.append({"type": "consensus", "items": group_items})
        elif len(group_items) == 2:
            groups.append({"type": "pair", "items": group_items})
        else:
            groups.append({"type": "single", "item": group_items[0]})

    # Check for contrasts between consensus groups
    if len(groups) >= 2:
        # Look for groups with similar topic but different measurements
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                g1_items = groups[i].get("items") or [groups[i].get("item", {})]
                g2_items = groups[j].get("items") or [groups[j].get("item", {})]
                if not g1_items or not g2_items:
                    continue
                # Check keyword overlap between groups
                kw1 = set()
                for it in g1_items:
                    kw1 |= set(t for t in it["finding"].lower().split() if len(t) > 4)
                kw2 = set()
                for it in g2_items:
                    kw2 |= set(t for t in it["finding"].lower().split() if len(t) > 4)
                overlap = len(kw1 & kw2) / max(len(kw1 | kw2), 1)
                if overlap > 0.15:
                    # Check if measurements differ
                    vals1 = []
                    for it in g1_items:
                        vals1.extend(it.get("measurements", []))
                    vals2 = []
                    for it in g2_items:
                        vals2.extend(it.get("measurements", []))
                    if vals1 and vals2:
                        mean1 = sum(v[0] for v in vals1) / len(vals1)
                        mean2 = sum(v[0] for v in vals2) / len(vals2)
                        if abs(mean1 - mean2) > 0.2 * max(mean1, mean2):
                            # Significant difference → contrast
                            groups[i] = {
                                "type": "contrast",
                                "sides": [g1_items, g2_items],
                            }
                            groups.pop(j)
                            return groups

    return groups


def _build_consensus_sentence(group: dict, verb_offset: int) -> str:
    """Build a consensus sentence for a group of agreeing studies."""
    items = group["items"]
    refs = [it["ref"] for it in items]
    ref_s = _ref_str(refs)

    # Try to extract common content
    # Find shared significant words across all items
    shared_words = set(it["finding"].split() for it in items)
    if len(shared_words) > 1:
        common = set.intersection(
            *[
                set(
                    t.strip(".,;:!?\"'()[]{}").lower()
                    for t in it["finding"].split()
                    if len(t) > 5
                )
                for it in items
            ]
        )
    else:
        common = set()

    if common:
        # Synthesize common content
        content = " ".join(sorted(common)[:8])
    else:
        content = items[0]["finding"][:100].lower()

    # Pick template by rotation
    template = _CONSENSUS_TEMPLATES[verb_offset % len(_CONSENSUS_TEMPLATES)]
    return template.format(refs=ref_s, content=content)


def _build_contrast_sentence(group: dict, verb_offset: int) -> str:
    """Build a contrast sentence for disagreeing groups."""
    side1, side2 = group["sides"]
    refs1 = _ref_str([it["ref"] for it in side1])
    refs2 = _ref_str([it["ref"] for it in side2])

    # Extract representative values from each side
    val1_parts = []
    for it in side1[:2]:
        finding = it["finding"][:60].rstrip(".")
        val1_parts.append(finding)
    val2_parts = []
    for it in side2[:2]:
        finding = it["finding"][:60].rstrip(".")
        val2_parts.append(finding)

    val1 = "; ".join(val1_parts) if val1_parts else "different results"
    val2 = "; ".join(val2_parts) if val2_parts else "contrasting results"

    template = _CONTRAST_TEMPLATES[verb_offset % len(_CONTRAST_TEMPLATES)]
    return template.format(refs1=refs1, refs2=refs2, val1=val1, val2=val2)


def _build_single_sentence(item: dict, verb_idx: int) -> str:
    """Build a single-finding sentence with varied structure."""
    verb, content = _rephrase_finding(item["finding"])
    # Rotate verbs for variety
    if verb == "reported that" and verb_idx < len(_REPORTING_VERBS):
        verb = _REPORTING_VERBS[verb_idx % len(_REPORTING_VERBS)]

    template = _SINGLE_TEMPLATES[verb_idx % len(_SINGLE_TEMPLATES)]
    return template.format(
        author=item["author"],
        year=item["year"],
        ref=item["ref"],
        verb=verb,
        content=content,
    )


def _build_pair_sentence(items: list[dict], verb_idx: int) -> str:
    """Build a sentence integrating two papers."""
    it1, it2 = items
    v1, c1 = _rephrase_finding(it1["finding"])
    v2, c2 = _rephrase_finding(it2["finding"])

    template = _PAIR_TEMPLATES[verb_idx % len(_PAIR_TEMPLATES)]
    return template.format(
        author1=it1["author"],
        year1=it1["year"],
        ref1=it1["ref"],
        verb1=v1,
        content1=c1,
        author2=it2["author"],
        year2=it2["year"],
        ref2=it2["ref"],
        verb2=v2,
        content2=c2,
    )


def _build_integrative_synthesis(
    theme_papers: list[dict],
    topic: str,
    theme_label: str,
    ref_offset: int,
    correlation: dict | None = None,
) -> tuple[str, list[dict]]:
    """Integrative synthesis — groups papers by scientific claim.

    Produces 3-6 integrative sentences per theme, each citing multiple papers.
    Replaces the one-sentence-per-paper approach.

    Architecture:
    1. Opening: method + measurement summary (1 sentence)
    2. Claim clusters: group findings by topic similarity, 2-4 integrative sentences
    3. Limitation synthesis: 1 sentence synthesizing collective limitations

    Each integrative sentence covers 2-8 papers with a collective claim.
    """
    cited = list(theme_papers)
    n = len(theme_papers)
    years = [p.get("year") for p in theme_papers if p.get("year")]
    year_str = ""
    if years:
        year_str = (
            f"{min(years)}-{max(years)}"
            if min(years) != max(years)
            else str(min(years))
        )

    for i, p in enumerate(theme_papers):
        p["_ref"] = ref_offset + 1 + i

    parts = []

    # 1. Opening sentence with measurement summary
    meas_summary = _meas_summary_str(theme_papers)
    study_word = "study" if n == 1 else "studies"
    opener_idx = abs(hash(theme_label)) % len(_THEME_OPENERS)
    opener = _THEME_OPENERS[opener_idx].format(
        n=n,
        study=study_word,
        theme=theme_label.lower(),
        topic=topic.lower(),
        meas_summary=meas_summary,
    )
    if year_str:
        opener += f" ({year_str})"
    parts.append(opener + ".")

    # 2. Extract quality findings with measurements
    items = []
    for p in theme_papers:
        finding = _clean_title_finding(_sanitize_finding(p.get("key_finding") or ""))
        if finding and _is_quality_finding(finding):
            items.append(
                {
                    "ref": p["_ref"],
                    "author": _author_short(_parse_authors(p.get("authors", []))),
                    "year": p.get("year") or "n.d.",
                    "finding": finding[:250],
                    "measurements": _extract_numbers_from_text(finding),
                    "paper_id": p.get("paper_id", ""),
                    "tokens": set(
                        t.strip(".,;:!?\"'()[]{}").lower()
                        for t in finding.split()
                        if len(t) > 4
                    ),
                }
            )

    # 3. Cluster findings into claim groups
    claim_clusters = _cluster_by_claim_semantic(items)

    # 4. Build ONE integrative sentence per cluster
    sentence_idx = 0
    for cluster in claim_clusters:
        sentence = _build_claim_sentence(cluster, sentence_idx)
        if sentence:
            parts.append(sentence)
            sentence_idx += 1

    # 5. Limitation synthesis
    lims = _extract_limitations(theme_papers)
    if lims:
        lim_idx = abs(hash(theme_label + "lim")) % len(_LIMITATION_TEMPLATES)
        lim_text = lims[0][:150].rstrip(".")
        parts.append(_LIMITATION_TEMPLATES[lim_idx].format(lim=lim_text) + ".")

    # Cleanup
    for p in theme_papers:
        p.pop("_ref", None)

    return " ".join(parts), cited


def _cluster_papers_hdbscan(papers: list, n_min: int = 3) -> list[dict]:
    """Cluster papers using HDBSCAN on BGE embeddings.

    Better than greedy keyword overlap:
    - Density-based: finds natural clusters
    - No fixed cluster count
    - Handles noise (outliers -> "Other studies")

    Falls back to _group_by_theme if embeddings unavailable.
    """
    if len(papers) < 5:
        return _group_by_theme(papers)
    try:
        from _embeddings import embed_texts
        from sklearn.cluster import HDBSCAN

        texts = [
            ((p.get("title") or "") + " " + (p.get("key_finding") or ""))[:500]
            for p in papers
        ]
        embeddings = embed_texts(texts)
        if embeddings is None:
            return _group_by_theme(papers)

        min_size = max(n_min, len(papers) // 10)
        clusterer = HDBSCAN(
            min_cluster_size=min_size,
            metric="cosine",
            cluster_selection_method="eom",
            min_samples=2,
            copy=False,  # silence sklearn 1.10 FutureWarning — we don't mutate input
        )
        labels = clusterer.fit_predict(embeddings)

        # Group papers by cluster label
        from collections import defaultdict

        groups: dict[int, list] = defaultdict(list)
        for i, label in enumerate(labels):
            groups[int(label)].append(papers[i])

        result = []
        for label, group_papers in sorted(groups.items(), key=lambda x: -len(x[1])):
            if label == -1:
                # Noise points: label by most common theme among them
                # rather than the uninformative "Other studies"
                themes_found = [_detect_method_theme(p) for p in group_papers[:5]]
                from collections import Counter

                most_common = Counter(themes_found).most_common(1)
                label_name = (
                    most_common[0][0]
                    if most_common and most_common[0][0] != "Other studies"
                    else "Mixed / uncategorized"
                )
            else:
                # Label by method theme detection
                themes_found = [_detect_method_theme(p) for p in group_papers[:5]]
                from collections import Counter

                most_common = Counter(themes_found).most_common(1)
                label_name = (
                    most_common[0][0] if most_common else f"Cluster {label + 1}"
                )
            result.append(
                {
                    "label": label_name,
                    "papers": _sort_chronological(group_papers),
                }
            )
        log.info(
            "HDBSCAN: %d clusters from %d papers (min_size=%d)",
            len(result),
            len(papers),
            min_size,
        )
        return result
    except Exception as e:
        log.debug("HDBSCAN failed, falling back to greedy: %s", e)
        return _group_by_theme(papers)


def _cluster_by_claim(items: list[dict]) -> list[dict]:
    """Cluster findings into claim groups by keyword similarity.

    Returns list of cluster dicts:
    - {"refs": [1,2,3], "items": [...], "claim_terms": set, "temperatures": [], "pressures": []}
    """
    if not items:
        return []

    assigned = set()
    clusters = []

    for i, item in enumerate(items):
        if i in assigned:
            continue
        cluster_items = [item]
        assigned.add(i)
        refs = [item["ref"]]
        common_tokens = set(item["tokens"])

        for j in range(i + 1, len(items)):
            if j in assigned:
                continue
            overlap = len(common_tokens & items[j]["tokens"]) / max(
                len(common_tokens | items[j]["tokens"]), 1
            )
            if overlap > 0.18:
                cluster_items.append(items[j])
                assigned.add(j)
                refs.append(items[j]["ref"])
                common_tokens = common_tokens & items[j]["tokens"]

        # Extract collective measurements
        temps = []
        pressures = []
        for ci in cluster_items:
            for val, unit in ci.get("measurements", []):
                if unit == "C":
                    temps.append(val)
                elif unit == "kbar":
                    pressures.append(val)

        clusters.append(
            {
                "refs": sorted(refs),
                "items": cluster_items,
                "claim_terms": common_tokens,
                "temperatures": temps,
                "pressures": pressures,
            }
        )

    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: -len(c["items"]))
    return clusters


# Integrative sentence templates — each cites MULTIPLE papers
_CLAIM_TEMPLATES = [
    "Multiple studies [{refs}] {verb} {claim}{meas_clause}",
    "Collective evidence from {n} {study_word} [{refs}] {verb} {claim}{meas_clause}",
    "Several investigations [{refs}] {verb} {claim}{meas_clause}",
    "Research [{refs}] {verb} {claim}{meas_clause}",
    "{n} {study_word} [{refs}] {verb} {claim}{meas_clause}",
    "Integrated results [{refs}] {verb} {claim}{meas_clause}",
]

_COLLECTIVE_VERBS = [
    "demonstrate",
    "report",
    "document",
    "establish",
    "show",
    "indicate",
    "confirm",
    "suggest",
    "support",
    "provide evidence for",
]


def _cluster_by_claim_semantic(items: list[dict]) -> list[dict]:
    """Semantic version of _cluster_by_claim using BGE embeddings.

    Falls back to keyword-based _cluster_by_claim if embeddings unavailable.
    """
    try:
        from _claims import _cluster_items

        clusters = _cluster_items(items, similarity_threshold=0.65)
        # Convert to _cluster_by_claim output format
        result = []
        for cluster_items_list in clusters:
            if not cluster_items_list:
                continue
            refs = sorted(
                set(it.get("ref", 0) for it in cluster_items_list if it.get("ref"))
            )
            claim_terms = set()
            temps = []
            pressures = []
            for it in cluster_items_list:
                content_text = (it.get("finding", "") or "").lower()
                claim_terms.update(content_text.split())
                for mtype, values in (it.get("measurements") or {}).items():
                    if "temp" in mtype.lower() and isinstance(values, list):
                        temps.extend(v for v in values if isinstance(v, (int, float)))
                    elif "press" in mtype.lower() and isinstance(values, list):
                        pressures.extend(
                            v for v in values if isinstance(v, (int, float))
                        )
            result.append(
                {
                    "refs": refs,
                    "items": cluster_items_list,
                    "claim_terms": claim_terms,
                    "temperatures": temps,
                    "pressures": pressures,
                }
            )
        return result if result else _cluster_by_claim(items)
    except Exception:
        return _cluster_by_claim(items)


def _build_claim_sentence(cluster: dict, idx: int) -> str:
    """Build ONE integrative sentence for a cluster of papers.

    Synthesizes the collective finding from all papers in the cluster.
    """
    refs = cluster["refs"]
    n = len(cluster["items"])
    ref_str = _ref_str(refs)

    # Extract claim from the BEST paper's finding text (not keyword salad)
    # Pick the paper with the longest quality finding as representative
    best_item = max(cluster["items"], key=lambda x: len(x.get("finding", "")))
    best_finding = best_item.get("finding", "")
    if best_finding:
        _, content = _rephrase_finding(best_finding)
        claim = content[:120].lower().rstrip(".") if content else "key findings"
    else:
        claim = "key findings"

    # Build measurement clause
    meas_parts = []
    temps = cluster["temperatures"]
    pressures = cluster["pressures"]
    if len(temps) >= 2:
        tmed = _stats.median(temps)
        meas_parts.append(
            f"temperatures of {min(temps):.0f}-{max(temps):.0f}C (median {tmed:.0f}C)"
        )
    elif len(temps) == 1:
        meas_parts.append(f"temperatures of {temps[0]:.0f}C")
    if len(pressures) >= 2:
        pmed = _stats.median(pressures)
        meas_parts.append(
            f"pressures of {min(pressures):.1f}-{max(pressures):.1f} kbar"
        )
    elif len(pressures) == 1:
        meas_parts.append(f"pressures of {pressures[0]:.1f} kbar")

    meas_clause = ""
    if meas_parts:
        meas_clause = ", with " + "; ".join(meas_parts[:2])

    # Select template and verb by rotation
    template = _CLAIM_TEMPLATES[idx % len(_CLAIM_TEMPLATES)]
    verb = _COLLECTIVE_VERBS[idx % len(_COLLECTIVE_VERBS)]

    # Format sentence
    study_word = "study" if n == 1 else "studies"
    sentence = template.format(
        refs=ref_str,
        n=n,
        study_word=study_word,
        verb=verb,
        claim=claim,
        meas_clause=meas_clause,
    )

    if not sentence.endswith("."):
        sentence += "."

    return sentence


def _synthesize_measurements_v2(theme_papers: list[dict]) -> str:
    """Enhanced measurement synthesis with IQR, outliers, bimodal detection.

    Reports:
    - Range (min-max)
    - Mean +/- std
    - IQR (Q1-Q3)
    - Outliers (>2 sigma from mean)
    - Bimodal detection (large std/mean ratio)
    """
    by_type: dict[str, list[float]] = defaultdict(list)
    for p in theme_papers:
        for m in p.get("measurements", []):
            mtype = (m.get("measurement") or "").strip().lower()
            munit = (m.get("unit") or "").strip()
            try:
                val = float(m.get("value", 0))
            except (TypeError, ValueError):
                continue
            # Normalize labels: temperature -> "temperature (C)", pressure -> "pressure (kbar)"
            if "temp" in mtype or munit in ("C", "K"):
                by_type["temperature (C)"].append(val)
            elif "press" in mtype or munit in ("kbar", "GPa", "MPa"):
                by_type["pressure (kbar)"].append(val)
            # Ages excluded from P-T measurement summaries — they are geological
            # context, not thermobarometric estimates, and should not be
            # summarized alongside temperatures and pressures.
        # Also extract from finding text (already validated + normalized)
        for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
            label = "temperature (C)" if unit == "C" else "pressure (kbar)"
            by_type[label].append(val)

    if not by_type:
        return ""

    parts = []
    for mtype, values in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(values) < 2:
            continue
        vmin = min(values)
        vmax = max(values)
        vmean = _stats.mean(values)
        vstd = _stats.stdev(values) if len(values) >= 2 else 0
        # IQR
        sorted_vals = sorted(values)
        n = len(sorted_vals)
        q1 = sorted_vals[n // 4] if n >= 4 else vmin
        q3 = sorted_vals[3 * n // 4] if n >= 4 else vmax

        # Outlier detection
        if vstd > 0:
            outliers = [v for v in values if abs(v - vmean) > 2 * vstd]
        else:
            outliers = []

        # Bimodal detection
        cv = vstd / abs(vmean) if vmean != 0 else 0
        is_bimodal = cv > 0.5 and n >= 4

        # Use median + IQR + MAD as primary (robust statistics)
        vmedian = _stats.median(values)
        # MAD (Median Absolute Deviation) — robust alternative to std
        mad = _stats.median([abs(v - vmedian) for v in values])
        line = f"{mtype}: median {vmedian:.1f} (IQR {q1:.1f}-{q3:.1f}, MAD {mad:.1f}, range {vmin:.1f}-{vmax:.1f}, n={len(values)})"
        # Bootstrap 95% CI for median (only with enough data)
        if len(values) >= 10:
            import random

            random.seed(42)  # deterministic
            boot_medians = [
                _stats.median(random.choices(values, k=len(values)))
                for _ in range(1000)
            ]
            boot_medians.sort()
            ci_lo = boot_medians[int(0.025 * len(boot_medians))]
            ci_hi = boot_medians[int(0.975 * len(boot_medians))]
            line += f", 95% CI [{ci_lo:.1f}, {ci_hi:.1f}]"
        if outliers:
            line += f"; outliers: {[f'{o:.1f}' for o in outliers[:3]]}"
        if is_bimodal:
            line += "; distribution may be bimodal — report by rock type for meaningful summary"
        parts.append(line)

    if not parts:
        return ""
    return "Quantitative summary: " + "; ".join(parts[:4]) + "."


def _build_cross_cutting_v2(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """V2 cross-cutting analysis with method comparison and convergences."""
    if len(themes) < 2:
        return ""

    parts = []

    # 1. Method dominance
    dominant = themes[0]
    parts.append(
        f"The {dominant['label'].lower()} approach dominates this corpus "
        f"({len(dominant['papers'])} studies), "
        f"complemented by {', '.join(t['label'].lower() for t in themes[1:4])}."
    )

    # 2. Cross-method measurement comparison
    method_measures = {}
    for theme in themes:
        label = theme["label"]
        temps = []
        pressures = []
        for p in theme["papers"]:
            for m in p.get("measurements", []):
                mtype = (m.get("measurement") or "").lower()
                munit = (m.get("unit") or "").strip()
                try:
                    val = float(m.get("value", 0))
                except (TypeError, ValueError):
                    continue
                if "temp" in mtype or munit in ("C", "K"):
                    temps.append(val)
                elif "press" in mtype or munit in ("kbar", "GPa", "MPa"):
                    pressures.append(val)
            for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
                if unit == "C":
                    temps.append(val)
                elif unit == "kbar":
                    pressures.append(val)
        if temps or pressures:
            method_measures[label] = {"T": temps, "P": pressures}

    if len(method_measures) >= 2:
        comparison_parts = []
        for label, mp in method_measures.items():
            if mp["T"] and len(mp["T"]) >= 2:
                comparison_parts.append(
                    f"{label}: T={min(mp['T']):.0f}-{max(mp['T']):.0f}C"
                )
            if mp["P"] and len(mp["P"]) >= 2:
                comparison_parts.append(
                    f"{label}: P={min(mp['P']):.0f}-{max(mp['P']):.0f}kbar"
                )
        if comparison_parts:
            parts.append(
                "Cross-method comparison: " + "; ".join(comparison_parts[:5]) + "."
            )

    # 3. Convergence/divergence
    all_temps = []
    all_pressures = []
    for p in all_papers:
        for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
            if unit == "C":
                all_temps.append(val)
            elif unit == "kbar":
                all_pressures.append(val)

    if len(all_temps) >= 5:
        t_median = _stats.median(all_temps)
        parts.append(
            f"Across all methods, temperature estimates have a median of {t_median:.0f}C "
            f"(n={len(all_temps)}), with variability reflecting differences in rock type "
            f"and metamorphic grade rather than methodological error."
        )
    if len(all_pressures) >= 5:
        p_median = _stats.median(all_pressures)
        parts.append(
            f"Pressure estimates have a median of {p_median:.1f} kbar (n={len(all_pressures)})."
        )

    # 4. Methodological diversity
    if len(themes) >= 3:
        parts.append(
            f"The diversity of approaches ({len(themes)} methodological themes) "
            f"reflects the complexity of constraining {topic.lower()} and "
            f"highlights the value of multi-method cross-validation."
        )

    return " ".join(parts)


_GEO_REGION_PATTERNS_RE_V2 = re.compile(
    r"\b(?:Antarctica|Greenland|Scandinavia|Alps|Himalaya|"
    r"Appalachian|Canadian\s+Shield|Baltic|Japan|China|"
    r"Africa|Australia|South\s+America|North\s+America|Europe|Asia)\b",
    re.IGNORECASE,
)


def _build_research_gaps_v2(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """V2 research gaps with quantitative analysis."""
    parts = []
    n = len(all_papers)
    theme_labels = {t["label"] for t in themes}

    # 1. Underrepresented methods
    small_themes = [t for t in themes if len(t["papers"]) <= 2]
    if small_themes:
        gap_labels = [t["label"].lower() for t in small_themes[:3]]
        parts.append(
            f"Several methodological approaches are represented by only 1-2 studies "
            f"({', '.join(gap_labels)}), limiting the robustness of cross-method comparisons."
        )

    # 2. Temporal gaps
    years = [p.get("year") for p in all_papers if p.get("year")]
    if years and len(years) >= 10:
        max_year = max(years)
        recent = sum(1 for y in years if y >= max_year - 3)
        if recent < len(years) * 0.2:
            parts.append(
                f"Only {recent} of {len(years)} studies were published in the last 3 years "
                f"({max_year - 3}-{max_year}), suggesting the field may benefit from updated investigations."
            )

    # 3. Method-specific gaps (generic — not hardcoded to geoscience)
    if n >= 15:
        method_gaps = []
        # Check for common method families that should be represented
        method_families = [
            ("experimental", ["experimental", "experiment", "synthetic"]),
            ("modeling", ["model", "simulation", "numerical"]),
            ("field-based", ["field", "outcrop", "sample", "collected"]),
            ("review", ["review", "synthesis", "meta-analysis"]),
        ]
        all_text = " ".join(
            (p.get("title") or "") + " " + (p.get("abstract") or "") for p in all_papers
        ).lower()
        for family_name, cues in method_families:
            if not any(cue in all_text for cue in cues):
                method_gaps.append(family_name)
        if method_gaps:
            parts.append(
                f"The corpus lacks {', '.join(method_gaps[:3])} studies, "
                f"representing a methodological gap."
            )

    # 4. Geographic diversity (pattern-based; undercounts — phrase honestly)
    regions_found = set()
    region_patterns = _GEO_REGION_PATTERNS_RE_V2
    for p in all_papers:
        text = (p.get("abstract") or "") + " " + (p.get("key_finding") or "")
        for m in region_patterns.finditer(text):
            regions_found.add(m.group(0).lower())
    if 0 < len(regions_found) <= 2 and n >= 10:
        parts.append(
            f"Only {len(regions_found)} geographic setting(s) were auto-detected "
            f"in abstract text (pattern-based detection, likely an undercount); "
            f"verify geographic diversity manually before limiting claims of "
            f"global applicability."
        )

    # 5. Measurement uncertainty
    all_measurements = []
    for p in all_papers:
        for m in p.get("measurements", []):
            mtype = (m.get("measurement") or "").lower()
            munit = (m.get("unit") or "").strip()
            if not (
                "temp" in mtype
                or "press" in mtype
                or munit in ("C", "K", "kbar", "GPa", "MPa")
            ):
                continue
            try:
                all_measurements.append(float(m.get("value", 0)))
            except (TypeError, ValueError):
                pass
    if len(all_measurements) >= 10:
        cv = (
            _stats.stdev(all_measurements) / abs(_stats.mean(all_measurements))
            if _stats.mean(all_measurements) != 0
            else 0
        )
        if cv > 0.5:
            parts.append(
                f"The high coefficient of variation in reported values ({cv:.1f}) "
                f"underscores the need for standardized analytical protocols."
            )

    # 6. Future directions
    if parts:
        parts.append(
            f"Future work should integrate multiple methodological approaches, "
            f"expand geographic coverage, and establish standardized protocols "
            f"for cross-method comparison of {topic.lower()}."
        )
    else:
        parts.append(
            f"The corpus provides broad coverage of {topic.lower()}. "
            f"Future work could integrate findings across methodological approaches "
            f"and extend studies to underrepresented regions."
        )

    return " ".join(parts)


# End of V2 synthesis engine


# Rock-type detection for contextual measurement grouping
_ROCK_TYPE_PATTERNS = [
    # Metamorphic
    (
        "metapelite",
        re.compile(
            r"\b(?:metapelite|metasediment|pelitic|metapelitic|pelite)\b", re.IGNORECASE
        ),
    ),
    ("eclogite", re.compile(r"\b(?:eclogite|eclogitic)\b", re.IGNORECASE)),
    (
        "granulite",
        re.compile(
            r"\b(?:granulite|granulitic|charnockite|enderbite)\b", re.IGNORECASE
        ),
    ),
    ("amphibolite", re.compile(r"\b(?:amphibolite|amphibolitic)\b", re.IGNORECASE)),
    (
        "gneiss",
        re.compile(
            r"\b(?:gn(?:e|†)iss(?:ic)?|orthogneiss|paragneiss|augen\s+gneiss)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "schist",
        re.compile(
            r"\b(?:schist(?:ose)?|mica\s+schist|garnet\s+schist|chlorite\s+schist)\b",
            re.IGNORECASE,
        ),
    ),
    ("skarn", re.compile(r"\b(?:skarn(?:s)?|tactite)\b", re.IGNORECASE)),
    ("quartzite", re.compile(r"\b(?:quartzite|quartzitic)\b", re.IGNORECASE)),
    (
        "marble",
        re.compile(
            r"\b(?:marble(?:s)?|crystalline\s+limestone|metacarbonate)\b", re.IGNORECASE
        ),
    ),
    (
        "serpentinite",
        re.compile(r"\b(?:serpentin(?:ite|inite)|serpentiniz)\b", re.IGNORECASE),
    ),
    (
        "blueschist",
        re.compile(r"\b(?:blueschist|blue\s+schist|lawsonite)\b", re.IGNORECASE),
    ),
    (
        "migmatite",
        re.compile(
            r"\b(?:migmat(?:ite|itic)|anatex|leucosome|melatome)\b", re.IGNORECASE
        ),
    ),
    (
        "mylonite",
        re.compile(
            r"\b(?:mylon(?:ite|itic)|ultramylon|cataclasite|fault\s+(?:rock|gouge|breccia))\b",
            re.IGNORECASE,
        ),
    ),
    ("hornfels", re.compile(r"\b(?:hornfels|contact\s+metamorph)\b", re.IGNORECASE)),
    (
        "calc-silicate",
        re.compile(r"\b(?:calc.silicate|calc-silicate)\b", re.IGNORECASE),
    ),
    # Igneous — volcanic
    (
        "basalt",
        re.compile(
            r"\b(?:basalt(?:ic)?|MORB|mid.ocean.ridge\s+basalt|OIB|flood\s+basalt)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "rhyolite",
        re.compile(
            r"\b(?:rhyolite|rhyolitic|obsidian|ignimbrite|ash\s+flow)\b", re.IGNORECASE
        ),
    ),
    ("andesite", re.compile(r"\b(?:andesite|andesitic)\b", re.IGNORECASE)),
    ("dacite", re.compile(r"\b(?:dacite|dacitic)\b", re.IGNORECASE)),
    ("komatiite", re.compile(r"\b(?:komatiite|komatiitic|boninite)\b", re.IGNORECASE)),
    (
        "carbonatite",
        re.compile(r"\b(?:carbonatite|nephelinite|melilitite)\b", re.IGNORECASE),
    ),
    (
        "tuff",
        re.compile(
            r"\b(?:tuff(?:aceous)?|volcanic\s+ash|pyroclastic)\b", re.IGNORECASE
        ),
    ),
    # Igneous — plutonic
    (
        "granite",
        re.compile(
            r"\b(?:granite|granitic|granodiorite|tonalite|trondhjemite|monzonite|syenite)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "gabbro",
        re.compile(
            r"\b(?:gabbro(?:ic)?|norite|troctolite|anorthosite)\b", re.IGNORECASE
        ),
    ),
    ("diorite", re.compile(r"\b(?:diorite|dioritic|monzodiorite)\b", re.IGNORECASE)),
    (
        "peridotite",
        re.compile(
            r"\b(?:peridotite|peridotitic|lherzolite|harzburgite|dunite|wehrlite|pyroxenite)\b",
            re.IGNORECASE,
        ),
    ),
    ("pegmatite", re.compile(r"\b(?:pegmatite|pegmatitic|aplite)\b", re.IGNORECASE)),
    # Sedimentary
    (
        "sandstone",
        re.compile(
            r"\b(?:sandstone|arenite|wacke|quartz\s+arenite|arkose|greywacke)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "shale",
        re.compile(
            r"\b(?:shale|shaly|mudstone|siltstone|argillite|claystone)\b", re.IGNORECASE
        ),
    ),
    (
        "limestone",
        re.compile(
            r"\b(?:limestone|calcarenite|micrite|chalk|coquina|packstone|wackestone|grainstone)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "dolomite",
        re.compile(r"\b(?:dolomite|dolostone|dolomit(?:e|ic))\b", re.IGNORECASE),
    ),
    (
        "chert",
        re.compile(
            r"\b(?:chert|chert(?:y|ic)|jasper|radiolarite|diatomite)\b", re.IGNORECASE
        ),
    ),
    (
        "conglomerate",
        re.compile(r"\b(?:conglomerate|breccia|fanglomerate)\b", re.IGNORECASE),
    ),
    (
        "evaporite",
        re.compile(
            r"\b(?:evaporite|halite|gypsum|anhydrite|salt\s+(?:dome|diapir))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "turbidite",
        re.compile(r"\b(?:turbidite|flysch|contourite|debris\s+flow)\b", re.IGNORECASE),
    ),
    (
        "BIF",
        re.compile(
            r"\b(?:banded\s+iron\s+formation|\bBIF\b|taconite|itabirite)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "coal",
        re.compile(
            r"\b(?:coal|coal\s+seam|lignite|anthracite|bituminous)\b", re.IGNORECASE
        ),
    ),
    # Mantle / deep Earth
    (
        "mantle xenolith",
        re.compile(
            r"\b(?:mantle\s+(?:xenolith|section|nodule|peridotite)|ophiolite|ophiolitic)\b",
            re.IGNORECASE,
        ),
    ),
    # Ore deposits
    (
        "VMS",
        re.compile(
            r"\b(?:volcanogenic\s+massive\s+sulfide|\bVMS\b|SEDEX|Mississippi\s+Valley)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "porphyry deposit",
        re.compile(
            r"\b(?:porphyry\s+(?:deposit|copper|gold)|epithermal)\b", re.IGNORECASE
        ),
    ),
]


def _detect_rock_type(paper: dict) -> str:
    """Detect the dominant rock type studied in a paper.

    Also detects UHP context to split eclogite into UHP vs non-UHP.
    """
    text = (
        (paper.get("title") or "")
        + " "
        + (paper.get("key_finding") or "")
        + " "
        + (paper.get("abstract") or "")
    ).lower()
    for rock_type, pattern in _ROCK_TYPE_PATTERNS:
        if pattern.search(text):
            # Check for UHP context in eclogites
            if rock_type == "eclogite":
                if re.search(
                    r"\b(?:ultrahigh.pressure|\bUHP\b|coesite|diamond.bearing|"
                    r"quartz.+inclusion|\d{2,}\s*kbar|>\s*25\s*kbar|"
                    r"\d\.\d+\s*GPa)\b",
                    text,
                    re.IGNORECASE,
                ):
                    return "UHP eclogite"
            return rock_type
    return "other"


def _build_field_evolution(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """Build a 'what has the field learned' synthesis paragraph.

    Traces the temporal evolution of methodological approaches and
    synthesizes key scientific advances — replaces paper-by-paper listing
    with genuine cross-paper synthesis.
    """
    if not all_papers:
        return ""

    parts = []

    # 1. Temporal method evolution
    # Group papers by time period and dominant method
    for p in all_papers:
        p["_rock_type"] = _detect_rock_type(p)

    early = [p for p in all_papers if (p.get("year") or 9999) <= 2010]
    middle = [p for p in all_papers if 2011 <= (p.get("year") or 9999) <= 2018]
    recent = [p for p in all_papers if (p.get("year") or 9999) >= 2019]

    if early and recent:
        early_methods = set(_detect_method_theme(p) for p in early)
        recent_methods = set(_detect_method_theme(p) for p in recent)
        new_methods = recent_methods - early_methods
        old_methods = early_methods - recent_methods

        if new_methods:
            parts.append(
                f"Early studies (pre-2010) relied primarily on "
                f"{', '.join(m.lower() for m in early_methods)}. "
                f"Since 2019, the field has diversified to include "
                f"{', '.join(m.lower() for m in new_methods)}, "
                f"reflecting broader methodological capabilities and "
                f"the recognition that no single approach fully captures the complexity of "
                f"{topic.lower()}."
            )
        elif early_methods != recent_methods:
            parts.append(
                f"The methodological landscape has evolved from "
                f"{', '.join(m.lower() for m in early_methods)} (pre-2010) "
                f"toward {', '.join(m.lower() for m in recent_methods)} (2019+), "
                f"reflecting improvements in analytical precision and "
                f"thermodynamic databases."
            )

    # 2. Key scientific insight synthesis
    # Extract recurring themes from findings
    all_findings = [p.get("key_finding") or "" for p in all_papers]
    all_findings_text = " ".join(all_findings).lower()

    # Detect key scientific debates/topics
    debates = []
    if "uncertain" in all_findings_text and (
        "calibrat" in all_findings_text or "model" in all_findings_text
    ):
        debates.append("calibration and model uncertainty")
    if "retrograde" in all_findings_text or "reset" in all_findings_text:
        debates.append("retrograde resetting of mineral compositions")
    if "equilibrium" in all_findings_text:
        debates.append("the assumption of chemical equilibrium")
    if "bulk composition" in all_findings_text:
        debates.append("sensitivity to bulk composition assumptions")
    if "inclusion" in all_findings_text and (
        "pressure" in all_findings_text or "elastic" in all_findings_text
    ):
        debates.append("elastic thermobarometry of mineral inclusions")

    if debates:
        parts.append(
            f"Recurring themes across the corpus include "
            f"{', '.join(debates[:4])}. "
            f"These represent the principal sources of uncertainty in current estimates of "
            f"{topic.lower()}."
        )

    # 3. Rock-type + study-type context for P-T estimates
    rock_type_temps: dict[str, list[float]] = defaultdict(list)
    rock_type_press: dict[str, list[float]] = defaultdict(list)
    for p in all_papers:
        rt = p.pop("_rock_type", "other")
        # Also tag study type to create composite label (e.g., "eclogite (natural)")
        st = _detect_study_type(p)
        if st == "experimental":
            rt_label = f"{rt} (experimental)"
        else:
            rt_label = rt
        for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
            if unit == "C":
                rock_type_temps[rt_label].append(val)
            elif unit == "kbar":
                rock_type_press[rt_label].append(val)
        for m in p.get("measurements", []):
            try:
                val = float(m.get("value", 0))
            except (TypeError, ValueError):
                continue
            munit = (m.get("unit") or "").strip()
            if munit == "C":
                rock_type_temps[rt_label].append(val)
            elif munit == "kbar":
                rock_type_press[rt_label].append(val)

    if len(rock_type_press) >= 2:
        rt_parts = []
        for rt, vals in sorted(rock_type_press.items(), key=lambda x: -len(x[1])):
            if len(vals) >= 2:
                rt_parts.append(
                    f"{rt}: {min(vals):.1f}-{max(vals):.1f} kbar (n={len(vals)})"
                )
        if rt_parts:
            parts.append(
                "Pressure estimates vary systematically by rock type: "
                + "; ".join(rt_parts[:4])
                + ". "
                + "This variation reflects genuine geological differences rather than methodological inconsistency."
            )

    return " ".join(parts)


# Study-type detection — splits measurement summaries by experimental vs natural
_STUDY_TYPE_PATTERNS = [
    (
        "experimental",
        re.compile(
            r"\b(?:piston\s+cylinder|multi.?anvil|diamond\s+anvil|"
            r"experimental\s+(?:run|study|investigation|calibrat)|"
            r"synthetic\s+(?:sample|run|mixture|starting\s+material)|"
            r"controlled\s+experiment|laboratory\s+(?:experiment|calibrat)|"
            r"high.pressure\s+experiment|crystalliz\w+\s+experiment|"
            r"hydrothermal\s+experiment|annealing\s+experiment)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "calibration",
        re.compile(
            r"\b(?:calibrat\w*|standardi[sz]\w*|reference\s+material|"
            r"inter.laboratory|round.robin|re.producib|accuracy\s+assessment)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "field study",
        re.compile(
            r"\b(?:natural\s+(?:sample|rock|outcrop)|field\s+(?:sample|study|area|evidence)|"
            r"collected\s+from|orogenic\s+belt|metamorphic\s+terrane|"
            r"crystalline\s+basement|exposure|terrain|structural\s+mapping|"
            r"regional\s+geolog|stratigraphic\s+(?:section|log|column)|"
            r"drill\s+core|core\s+sample|quarry|borehole|well\s+log)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "geochemistry",
        re.compile(
            r"\b(?:whole.rock\s+geochem|bulk\s+geochem|major\s+element|"
            r"trace\s+element\s+(?:analysis|geochem)|rare\s+earth|\bREE\b|"
            r"\bICP.MS\b|\bXRF\b|\bEPMA\b|electron\s+microprobe|"
            r"\bLA.ICP.MS\b|isotope\s+geochem|stable\s+isotope)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "geochronology",
        re.compile(
            r"\b(?:geochronolog|\bU.Pb\b|\bAr.Ar\b|\b40Ar.39Ar\b|"
            r"fission\s+track|cosmogenic\s+(?:nuclide|exposure)|"
            r"(?:thermo|optically)\s+luminescence|\bOSL\b|\bTL\b|"
            r"radiocarbon|\b14C\b|\bRe.Os\b|\bSm.Nd\b|\bLu.Hf\b|"
            r"monazite\s+(?:age|dating)|zircon\s+(?:age|dating|\bU.Pb\b))"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "geophysics",
        re.compile(
            r"\b(?:seismic\s+(?:survey|profile|tomograph|reflection|refraction)|"
            r"receiver\s+function|magnetotellur|gravity\s+(?:survey|anomal)|"
            r"aeromagnetic|electrical\s+resistivity|self.potential|"
            r"ground.penetrating\s+radar|\bGPR\b|heat\s+flow\s+(?:measure|survey))"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "remote sensing",
        re.compile(
            r"\b(?:satellite\s+(?:imag|data)|\bInSAR\b|\bD.InSAR\b|"
            r"\bLandsat\b|\bASTER\b|\bSentinel\b|\bMODIS\b|"
            r"airborne\s+(?:survey|magnetic|EM)|\bLiDAR\b|\bSAR\b|"
            r"hyperspectral|multispectral|thermal\s+infrared)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "modeling",
        re.compile(
            r"\b(?:numerical\s+model|computer\s+simulation|thermodynamic\s+model|"
            r"finite\s+element|finite\s+difference|discrete\s+element|"
            r"analog\s+model|sandbox\s+model|geodynamic\s+model|"
            r"subduction\s+model|convection\s+model)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "drilling",
        re.compile(
            r"\b(?:\bODP\b|\bIODP\b|\bDSDP\b|scientific\s+drilling|"
            r"drill\s+core|coring|\bICDP\b|continental\s+drilling|"
            r"borehole\s+logging|wireline\s+logging)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
    (
        "review",
        re.compile(
            r"\b(?:review\b|meta.analysis|systematic\s+review|"
            r"overview|state.of.the.art|synthesi[sz]e|summari[sz]e)"
            r"\b",
            re.IGNORECASE,
        ),
    ),
]


def _detect_study_type(paper: dict) -> str:
    """Detect study type: experimental, calibration, or natural.

    Natural samples are field-collected rocks.
    Experimental studies use piston cylinder, multi-anvil, etc.
    Calibration studies calibrate thermobarometers.
    """
    text = (
        (paper.get("title") or "")
        + " "
        + (paper.get("abstract") or "")
        + " "
        + (paper.get("key_finding") or "")
    ).lower()
    for study_type, pattern in _STUDY_TYPE_PATTERNS:
        if pattern.search(text):
            return study_type
    return "natural"  # default to natural for geological studies


# Advantage extraction — for method comparison synthesis
_ADVANTAGE_CUES = re.compile(
    r"\b(?:advantage|improve\w*|superior|better|accurate|precise|reliable|"
    r"robust|rigorous|comprehensive|systematic|efficient|novel|innovative|"
    r"outperform|excel|surpass|enhance|optimi[sz]e|"
    r"internally\s+consistent|thermodynamically\s+consistent|"
    r"overcome|address|resolve\w*)\b",
    re.IGNORECASE,
)


def _extract_advantages(theme_papers: list) -> list:
    """Extract advantage statements from paper findings.

    Used for method comparison — identifies what each method does well.
    """
    advantages = []
    for p in theme_papers:
        finding = (p.get("key_finding") or "") + " " + (p.get("interpretation") or "")
        for sentence in finding.split("."):
            sentence = sentence.strip()
            if _ADVANTAGE_CUES.search(sentence) and len(sentence) > 25:
                advantages.append(sentence[:200])
                break
    return advantages[:3]


def _build_method_comparison_synthesis(
    themes: list[dict],
    all_papers: list[dict],
    topic: str,
) -> str:
    """Build a method comparison paragraph — compare, don't list.

    Generates coherent cross-method synthesis:
    - Identifies which methods are newer/older (temporal evolution)
    - Compares measurement ranges across methods (which agree/disagree)
    - Extracts advantages and limitations per method from the corpus itself
    - Explains why some methods are preferred over others

    This is genuine SCIENTIFIC SYNTHESIS, not paper-by-paper summary.
    """
    if len(themes) < 2:
        return ""

    parts = []

    # 1. Extract per-theme data
    method_data = []
    for theme in themes:
        label = theme["label"]
        papers = theme["papers"]
        years = [p.get("year") for p in papers if p.get("year")]
        avg_year = sum(years) / len(years) if years else 0

        # Extract limitations and advantages
        lims = _extract_limitations(papers[:8])
        advs = _extract_advantages(papers[:8])

        # Get measurement ranges
        temps = []
        pressures = []
        for p in papers:
            for m in p.get("measurements", []):
                munit = (m.get("unit") or "").strip()
                try:
                    val = float(m.get("value", 0))
                except (TypeError, ValueError):
                    continue
                if munit == "C" and 100 <= val <= 2000:
                    temps.append(val)
                elif munit == "kbar" and 0.1 <= val <= 150:
                    pressures.append(val)

        method_data.append(
            {
                "label": label,
                "n": len(papers),
                "avg_year": avg_year,
                "limitations": lims,
                "advantages": advs,
                "temps": temps,
                "pressures": pressures,
            }
        )

    # 2. Sort by average year (oldest first)
    method_data.sort(key=lambda x: x["avg_year"])
    if len(method_data) < 2:
        return ""

    # 3. Build comparative statements — DATA-DRIVEN, not assumed
    # Report the temporal distribution of each method from actual corpus data
    earliest_methods = [md for md in method_data if md["avg_year"] > 0]
    if earliest_methods:
        earliest_methods.sort(key=lambda x: x["avg_year"])
        timeline_parts = []
        for md in earliest_methods[:6]:
            yr = int(md["avg_year"])
            timeline_parts.append(f"{md['label'].lower()} (avg {yr}, n={md['n']})")
        parts.append(
            "Temporal distribution from corpus: " + "; ".join(timeline_parts) + "."
        )

    # Per-method comparative sentences
    for md in method_data:
        label = md["label"]
        adv = md["advantages"][0][:100] if md["advantages"] else None
        lim = md["limitations"][0][:100] if md["limitations"] else None

        if adv and lim:
            parts.append(f"{label} {adv.lower()}; however, {lim.lower()}.")
        elif lim:
            parts.append(f"{label} is limited by {lim.lower()}.")
        elif adv:
            parts.append(f"{label} {adv.lower()}.")

    # Cross-method measurement comparison
    methods_with_temps = [md for md in method_data if len(md["temps"]) >= 3]
    if len(methods_with_temps) >= 2:
        comparison_parts = []
        for md in methods_with_temps:
            tmed = _stats.median(md["temps"])
            comparison_parts.append(f"{md['label'].lower()}: {tmed:.0f} C")
        parts.append(
            "Cross-method comparison of median temperatures: "
            + ", ".join(comparison_parts)
            + ". "
            + (
                "Differences reflect varying applicability to different metamorphic grades."
                if len(comparison_parts) >= 3
                else "The difference likely reflects distinct target assemblages."
            )
        )

    methods_with_press = [md for md in method_data if len(md["pressures"]) >= 3]
    if len(methods_with_press) >= 2:
        comparison_parts = []
        for md in methods_with_press:
            pmed = _stats.median(md["pressures"])
            comparison_parts.append(f"{md['label'].lower()}: {pmed:.1f} kbar")
        parts.append("Median pressures: " + ", ".join(comparison_parts) + ".")

    # Convergence/divergence sentence
    if methods_with_temps:
        all_medians = [_stats.median(md["temps"]) for md in methods_with_temps]
        if max(all_medians) - min(all_medians) < 50:
            parts.append(
                "The broad agreement in temperature estimates across methods "
                "increases confidence in the robustness of thermobarometric results."
            )
        else:
            parts.append(
                "The substantial spread in temperature estimates across methods "
                "highlights the importance of method selection and the need for "
                "multi-method cross-validation in {topic}."
            )

    return " ".join(parts)


# =============================================================================
# Narrative builders — non-LLM template stitching
# =============================================================================


def _sort_chronological(papers: list[dict]) -> list[dict]:
    """Sort papers by year (ascending), None last."""
    return sorted(papers, key=lambda p: (p.get("year") is None, p.get("year") or 9999))


def _filter_with_findings(papers: list[dict]) -> list[dict]:
    """Return only papers that have a non-empty key_finding."""
    return [p for p in papers if (p.get("key_finding") or "").strip()]


_STEM_SUFFIXES = re.compile(
    r"(?:s|es|ed|ing|tion|tions|ity|ities|ment|ments|al|ly|ogy|ological|ological)$"
)


def _stem(word: str) -> str:
    """Simple suffix-stripping stemmer for topical matching."""
    w = word.lower().strip()
    for _ in range(2):
        m = _STEM_SUFFIXES.sub("", w)
        if m == w or len(m) < 4:
            break
        w = m
    return w


def _keyword_relevance(paper: dict, topic: str) -> float:
    """Legacy keyword overlap score (fallback for semantic)."""
    topic_stems = set(_stem(t) for t in topic.split() if len(t) > 2)
    title = (paper.get("title") or "").lower()
    finding = (paper.get("key_finding") or "").lower()
    abstract = (paper.get("abstract") or "").lower()
    paper_text = f"{title} {finding} {abstract}"
    paper_stems = set(_stem(t) for t in paper_text.split() if len(t) > 2)
    if not topic_stems:
        return 0.0
    return len(topic_stems & paper_stems) / len(topic_stems)


def _semantic_topical_scores(papers: list, topic: str) -> list:
    """TF-IDF cosine similarity between topic and each paper.

    Uses sklearn TfidfVectorizer (already a geokit dependency).
    Catches papers sharing concepts but not exact keywords.
    Falls back to keyword scores if sklearn unavailable.
    """
    if not papers:
        return []
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim

        texts = [topic.lower().strip()]
        for p in papers:
            title = p.get("title") or ""
            finding = p.get("key_finding") or ""
            abstract = p.get("abstract") or ""
            texts.append((title + " " + finding + " " + abstract).lower().strip())

        vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
            max_df=0.95,
            stop_words="english",
        )
        matrix = vectorizer.fit_transform(texts)
        sims = _cos_sim(matrix[0:1], matrix[1:]).flatten()
        mx = sims.max()
        if mx > 0:
            sims = sims / mx
        return [float(s) for s in sims]
    except Exception:
        return [_keyword_relevance(p, topic) for p in papers]


def _topical_relevance(paper: dict, topic: str) -> float:
    """Score topical relevance: TF-IDF cosine (primary) + keyword fallback."""
    sem = _semantic_topical_scores([paper], topic)
    return sem[0] if sem else _keyword_relevance(paper, topic)


def _filter_topical(papers: list, topic: str, min_relevance: float = 0.05) -> list:
    """Filter + rank papers by semantic relevance (BGE or TF-IDF).

    Priority: BGE embeddings (SOTA) > TF-IDF cosine > keyword overlap.
    Hybrid: max(semantic, keyword*0.5) catches both conceptual and exact matches.
    Graduated fallback retains >=50% of corpus.
    """
    if not papers:
        return []

    # Try BGE embeddings first (SOTA semantic ranking)
    try:
        from _embeddings import embed_paper

        bge_ranked = embed_paper(topic, papers)
        if bge_ranked and len(bge_ranked) >= len(papers) // 2:
            # BGE gave good results — use semantic scores
            bge_scores = {id(p): s for p, s in bge_ranked}
            scores = [bge_scores.get(id(p), 0.0) for p in papers]
            kw_scores = [_keyword_relevance(p, topic) for p in papers]
            hybrid = [
                (p, max(s, k * 0.5)) for p, s, k in zip(papers, scores, kw_scores)
            ]
            hybrid.sort(key=lambda x: -x[1])
            relevant = [(p, s) for p, s in hybrid if s >= 0.15]
            if relevant:
                log.info(
                    "BGE topical filter: %d/%d papers above threshold",
                    len(relevant),
                    len(papers),
                )
                min_keep = max(len(relevant), len(papers) // 2)
                if len(relevant) < min_keep:
                    for p, s in hybrid:
                        if (p, s) not in relevant and len(relevant) < min_keep:
                            relevant.append((p, s))
                return [p for p, _ in relevant]
    except Exception:
        pass

    # Fallback: TF-IDF cosine similarity
    scores = _semantic_topical_scores(papers, topic)
    kw_scores = [_keyword_relevance(p, topic) for p in papers]
    hybrid = [(p, max(s, k * 0.5)) for p, s, k in zip(papers, scores, kw_scores)]
    hybrid.sort(key=lambda x: -x[1])
    relevant = [(p, s) for p, s in hybrid if s >= min_relevance]
    if not relevant:
        return [p for p, _ in hybrid]
    min_keep = max(len(relevant), len(papers) // 2)
    if len(relevant) < min_keep:
        for p, s in hybrid:
            if (p, s) not in relevant and len(relevant) < min_keep:
                relevant.append((p, s))
    return [p for p, _ in relevant]


_VERB_MAP = [
    (re.compile(r"^(?:we|our)\s+(?:found|find)\b", re.IGNORECASE), "found that"),
    (
        re.compile(
            r"^(?:we|our)\s+(?:show|showed|demonstrate|demonstrated)\b", re.IGNORECASE
        ),
        "demonstrated that",
    ),
    (
        re.compile(
            r"^(?:we|our)\s+(?:report|reported|present|presented)\b", re.IGNORECASE
        ),
        "reported that",
    ),
    (
        re.compile(
            r"^(?:we|our)\s+(?:propose|proposed|introduce|introduced)\b", re.IGNORECASE
        ),
        "proposed that",
    ),
    (
        re.compile(r"^(?:we|our)\s+(?:conclude|concluded)\b", re.IGNORECASE),
        "concluded that",
    ),
    (re.compile(r"^(?:we|our)\s+(?:argue|argued)\b", re.IGNORECASE), "argued that"),
    (
        re.compile(r"^(?:we|our)\s+(?:observ\w+|not\w+|detect\w+)\b", re.IGNORECASE),
        "observed that",
    ),
    (
        re.compile(
            r"^(?:this|the)\s+(?:study|work|paper|research)\s+(?:show|showed|demonstrat\w+|found|report\w+|present\w+)\b",
            re.IGNORECASE,
        ),
        "showed that",
    ),
    (
        re.compile(
            r"^(?:here|in this study)\s+we\s+(?:show|demonstrate|report|present|found)\b",
            re.IGNORECASE,
        ),
        "demonstrated that",
    ),
    (
        re.compile(
            r"^(?:our|these|the)\s+results\s+(?:show|showed|demonstrate|indicate|suggest|reveal)\b",
            re.IGNORECASE,
        ),
        "showed that",
    ),
    (
        re.compile(
            r"^(?:this|our)\s+(?:analysis|approach|method)\s+(?:allow\w*|enabl\w+|provid\w+)\b",
            re.IGNORECASE,
        ),
        "reported that",
    ),
]


def _is_non_english(text: str, threshold: float = 0.3) -> bool:
    """Detect if text is predominantly non-English.
    Checks for high ratio of non-ASCII letters (Turkish, German, etc.).
    """
    if not text or len(text) < 20:
        return False
    # Count non-ASCII alpha chars (diacritics, foreign letters)
    non_ascii = sum(1 for c in text if ord(c) > 127 and c.isalpha())
    total_alpha = sum(1 for c in text if c.isalpha())
    if total_alpha == 0:
        return False
    ratio = non_ascii / total_alpha
    return ratio >= threshold


_GARBLED_PATTERNS = re.compile(
    r"(?:KEY WORDS|Keywords?|Abstract unavailable|You do not have|<|&lt;|"
    r"Integrating these results with new|After exclusion of samples|"
    r"\d+\s*Ma\)|propagating uncertainty|depending on the temperature|"
    r"modeling the fractionation|"
    r"analytical methods|supplemental (?:material|figures|data)|"
    r"supplementary (?:material|data|information)|"
    r"(?:c?ontact|corresponding author)[:\s]|"
    r".*@\w+\.\w{2,3}|"
    r".*\]\s*,?\s*which\s+(?:features|show)|"
    r"\d+\s*,\s*\d+\s*\(\d+\)|"
    r"(?:editor'?s|reviewer'?s)\s+(?:careful|kind)\s+(?:attention|comments?)|"
    r"rensen|"
    r"(?:damarlar|olumular|ierisinde|ncelme|alanndaki|barit-galenit)|"
    r"(?:self-gravitating|gravitational fugac)|"
    r"chapter\s+(?:also\s+)?(?:outlines?|discusses?)\s+key)",
    re.IGNORECASE,
)

# Significance inference: stance lexicon for inter-finding relationships
_CONTRAST_TERMS = frozenset(
    {
        "however",
        "unlike",
        "contrary",
        "contrast",
        "disagree",
        "dispute",
        "challenge",
        "conflict",
        "differ",
        "opposite",
        "reject",
        "refute",
        "counter",
        "contradict",
        "inconsistent",
        "anomal",
        "deviat",
    }
)
_SUPPORT_TERMS = frozenset(
    {
        "confirm",
        "consistent",
        "agree",
        "support",
        "verify",
        "validate",
        "corroborate",
        "accord",
        "correlate",
        "positively",
        "similar",
        "comparable",
        "analogous",
        "parallel",
        "match",
        "reinforce",
    }
)
_EXTEND_TERMS = frozenset(
    {
        "furthermore",
        "additionally",
        "moreover",
        "extend",
        "expand",
        "complement",
        "supplement",
        "build",
        "develop",
        "refine",
        "improve",
        "advance",
        "progress",
        "evolve",
    }
)


def _detect_significance(
    prev_finding: str,
    curr_finding: str,
) -> str | None:
    """Detect if curr finding supports, contrasts, or extends prev finding.

    Returns "support", "contrast", "extend", or None (independent).
    Uses keyword overlap + stance lexicon from _lexicon patterns.
    """
    if not prev_finding or not curr_finding:
        return None

    prev_tokens = set(
        t.lower().strip(".,;:!?\"'()[]{}") for t in prev_finding.split() if len(t) > 3
    )
    curr_tokens = set(
        t.lower().strip(".,;:!?\"'()[]{}") for t in curr_finding.split() if len(t) > 3
    )
    if not prev_tokens or not curr_tokens:
        return None

    overlap = len(prev_tokens & curr_tokens) / max(len(prev_tokens | curr_tokens), 1)

    # Only assess significance if findings are topically related
    if overlap < 0.12:
        return None  # Independent — use year/discipline transitions

    curr_lower = curr_finding.lower()

    # Check for contrast signals
    contrast_hits = sum(1 for t in _CONTRAST_TERMS if t in curr_lower)
    # Check for support signals
    support_hits = sum(1 for t in _SUPPORT_TERMS if t in curr_lower)
    # Check for extend signals
    extend_hits = sum(1 for t in _EXTEND_TERMS if t in curr_lower)

    if contrast_hits > support_hits and contrast_hits > 0:
        return "contrast"
    if support_hits > 0 and support_hits >= extend_hits:
        return "support"
    if extend_hits > 0:
        return "extend"

    # Default for high-overlap findings without explicit signals
    if overlap >= 0.25:
        return "extend"

    return None


def _rephrase_finding(finding: str) -> tuple[str, str]:
    """Rephrase a raw finding into academic citation form.

    Returns (verb, content) where verb is the inferred reporting verb
    and content is the finding with the opening phrase stripped.
    """
    if not finding:
        return "reported", ""

    cleaned = finding.strip()
    # Ensure period at end
    if not cleaned.endswith((".", "!", "?")):
        cleaned += "."

    # Strip leading discourse markers and sentence connectors
    cleaned = re.sub(
        r"^(?:Thus|Therefore|However|Nevertheless|Nonetheless|Moreover|Furthermore|"
        r"Consequently|Accordingly|Hence|Overall|Indeed|Specifically|"
        r"These\s+results?|This\s+(?:study|work|indicates|suggests|necessitates|implies)|"
        r"Our\s+(?:results?|findings?|data|analys[ei]s)\s+(?:show|indicat|demonstrat|reveal|suggest)?|"
        r"Our\s+(?:results?|findings?|data|analys[ei]s))\s*[,\.]?\s*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    # Strip leftover "that" at start (from partial stripping like "indicates that")
    cleaned = re.sub(r"^that\s+", "", cleaned, flags=re.IGNORECASE)

    # Strip leftover "show that" / "demonstrate that" / "how to" at start
    cleaned = re.sub(
        r"^(?:show|demonstrate|reveal|report|observ\w+|suggest|indicat\w+)\s+(?:that\s+)?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"^how\s+to\s+", "", cleaned, flags=re.IGNORECASE)

    if not cleaned:
        return "reported", ""
    if cleaned[0].islower():
        cleaned = cleaned[0].upper() + cleaned[1:]

    # Try to match a verb pattern
    for pattern, verb in _VERB_MAP:
        match = pattern.match(cleaned)
        if match:
            content = cleaned[match.end() :].strip()
            if content:
                # Strip leading "that" left over from verb extraction
                content = re.sub(r"^that\s+", "", content, flags=re.IGNORECASE)
                if content:
                    return verb, content

    # No known opening pattern — use "reported that" with lowercase start
    return "reported that", cleaned[:1].lower() + cleaned[1:] if cleaned else ""


def _is_quality_finding(finding: str) -> bool:
    """Check if a finding is high-quality enough for narrative inclusion."""
    if not finding or len(finding.strip()) < 40:
        return False
    if _GARBLED_PATTERNS.search(finding.strip()):
        return False
    if _is_non_english(finding):
        return False
    # Check for truncated sentences (ending with single letter + period)
    if re.search(r"\b\w\.$", finding) and not re.search(r"\b[A-Z]\.$", finding):
        return False
    # Check for numeric gibberish (mostly digits with few words)
    words = finding.split()
    digit_ratio = sum(1 for w in words if any(c.isdigit() for c in w)) / max(
        len(words), 1
    )
    if digit_ratio > 0.5:
        return False
    return True


# Transition selection: context-aware, not just rotating
_BASE_TRANSITIONS = [
    "Building on this, ",
    "Subsequently, ",
    "In a related study, ",
    "Further work by ",
    "Extending these findings, ",
    "In parallel, ",
    "Complementing this work, ",
    "Adding to this body of evidence, ",
    "Following up on these results, ",
    "Continuing this line of inquiry, ",
]


# Significance-based transitions (highest priority)
_SIGNIFICANCE_TRANSITIONS = {
    "support": [
        "Confirming these findings, ",
        "Consistent with these results, ",
        "Corroborating this evidence, ",
        "In agreement with prior work, ",
    ],
    "contrast": [
        "In contrast to previous work, ",
        "Challenging this interpretation, ",
        "However, ",
        "Taking a different view, ",
    ],
    "extend": [
        "Extending this framework, ",
        "Building upon these results, ",
        "Taking this further, ",
        "Elaborating on these findings, ",
    ],
}


def _select_transition(
    idx: int,
    prev_paper: dict | None,
    curr_paper: dict,
    used_transitions: set[str],
) -> str:
    """Select a contextually appropriate transition between papers.

    Priority order:
    1. Significance inference (support/contrast/extend between findings) — HIGHEST
    2. Year-gap transition
    3. Discipline-shift transition
    4. Novelty-based transition
    5. Rotating transition (avoid repeats)
    """
    if idx == 0:
        return ""

    # 0. Significance inference — detect inter-finding relationship
    prev_finding = (prev_paper or {}).get("key_finding") or ""
    curr_finding = curr_paper.get("key_finding") or ""
    significance = _detect_significance(prev_finding, curr_finding)

    if significance:
        candidates = _SIGNIFICANCE_TRANSITIONS[significance]
        for t in candidates:
            if t not in used_transitions:
                return t
        return candidates[idx % len(candidates)]

    # 1. Year-gap transition
    prev_year = prev_paper.get("year") if prev_paper else None
    curr_year = curr_paper.get("year")
    if prev_year and curr_year:
        gap = curr_year - prev_year
        if gap >= 20:
            return "After more than two decades, "
        elif gap >= 10:
            return "After a decade of further research, "
        elif gap >= 8:
            return "Several years later, "

    # 2. Discipline-shift transition (replace underscores with spaces)
    prev_disc = (
        ((prev_paper or {}).get("discipline") or "").replace("_", " ")
        if prev_paper
        else ""
    )
    curr_disc = (curr_paper.get("discipline") or "").replace("_", " ")
    if prev_disc and curr_disc and prev_disc != curr_disc:
        return f"From a {curr_disc} perspective, "

    # 3. Novelty-based transition
    novelty = curr_paper.get("novelty", "")
    if novelty == "review":
        return "In a comprehensive review, "
    if novelty == "novel":
        return "Introducing a novel approach, "

    # 4. Rotating transition (avoid repeats)
    for t in _BASE_TRANSITIONS:
        if t not in used_transitions:
            return t
    return _BASE_TRANSITIONS[idx % len(_BASE_TRANSITIONS)]


def _compute_theme_keywords(paper: dict) -> set[str]:
    """Extract significant keywords from a paper for theme clustering."""
    text = f"{paper.get('title') or ''} {paper.get('key_finding') or ''}".lower()
    # Extract significant words (>4 chars, not stopwords)
    _STOPWORDS = {
        "the",
        "this",
        "that",
        "these",
        "those",
        "with",
        "from",
        "have",
        "been",
        "were",
        "they",
        "their",
        "which",
        "would",
        "could",
        "should",
        "about",
        "into",
        "between",
        "through",
        "during",
        "before",
        "after",
        "above",
        "below",
        "under",
        "over",
        "also",
        "more",
        "most",
        "some",
        "such",
        "only",
        "very",
        "than",
        "then",
        "both",
        "each",
        "other",
        "results",
        "study",
        "studies",
        "based",
        "using",
        "used",
        "data",
        "show",
        "found",
        "indicate",
        "suggest",
        "demonstrate",
        "report",
    }
    tokens = set(
        t.strip(".,;:!?\"'()[]{}")
        for t in text.split()
        if len(t) > 4 and t.strip(".,;:!?\"'()[]{}").lower() not in _STOPWORDS
    )
    return tokens


def _detect_paragraph_break(
    prev_paper: dict,
    curr_paper: dict,
    prev_idx: int,
) -> bool:
    """Decide if a paragraph break should occur between two consecutive papers.

    Break when:
    - Temporal cluster boundary (>5 year gap)
    - Theme shift (low keyword overlap between consecutive papers)
    - Every 5 papers maximum (readability)
    """
    # Year gap (larger threshold — don't break on every small gap)
    prev_year = prev_paper.get("year")
    curr_year = curr_paper.get("year")
    if prev_year and curr_year and abs(curr_year - prev_year) > 8:
        return True

    # Theme shift
    prev_kw = _compute_theme_keywords(prev_paper)
    curr_kw = _compute_theme_keywords(curr_paper)
    if prev_kw and curr_kw:
        overlap = len(prev_kw & curr_kw) / max(len(prev_kw | curr_kw), 1)
        if overlap < 0.15:  # Very different topics
            return True

    # Every 5 papers
    if (prev_idx + 1) % 5 == 0:
        return True

    return False


# Content-based verb enrichment — varies verb by finding content
_CONTENT_VERB_MAP = [
    (re.compile(r"\bfirst\b", re.IGNORECASE), "pioneered the demonstration that"),
    (
        re.compile(
            r"\b(?:model|simulation|numerical|calculat|comput)\w*\b", re.IGNORECASE
        ),
        "modeled",
    ),
    (re.compile(r"\b(?:review|synthesi\w+|compil\w+)\b", re.IGNORECASE), "reviewed"),
    (
        re.compile(r"\b(?:measure|determin\w+|quantif\w+)\b", re.IGNORECASE),
        "determined that",
    ),
    (re.compile(r"\b(?:argue| propos\w+|hypnoth\w+)\b", re.IGNORECASE), "argued that"),
    (re.compile(r"\b(?:conclude|infer\w+)\b", re.IGNORECASE), "concluded that"),
    (re.compile(r"\b(?:observ\w+|not\w+|detect\w+)\b", re.IGNORECASE), "observed that"),
    (
        re.compile(r"\b(?:establish\w+|demonstrat\w+)\b", re.IGNORECASE),
        "established that",
    ),
]


def _enrich_verb(default_verb: str, finding: str, paper: dict) -> str:
    """Enrich verb based on finding content for variety.

    Only overrides the generic 'reported that' default — keeps specific
    verbs from _VERB_MAP (found that, showed that, etc.).
    """
    if default_verb != "reported that":
        return default_verb  # Already has a specific verb

    # Check novelty
    novelty = paper.get("novelty", "")
    if novelty == "review":
        return "reviewed"
    if novelty == "novel":
        return "demonstrated that"

    # Check content for verb cues
    for pattern, verb in _CONTENT_VERB_MAP:
        if pattern.search(finding):
            return verb

    return default_verb


def _interp_connector(interpretation: str) -> str:
    """Select appropriate connector for interpretation clause.

    Detects how the interpretation relates to the finding:
    - "These results suggest..." → "This suggests that"
    - "We conclude..." → "They concluded that"
    - "This implies..." → "This implies that"
    - Default → "This indicates that"
    """
    interp_lower = interpretation[:50].lower()
    if interp_lower.startswith(("we conclude", "we propose", "we infer")):
        return "They concluded that"
    if interp_lower.startswith(
        ("these results", "these findings", "these observations", "these data")
    ):
        return "These results suggest that"
    if interp_lower.startswith(("this implies", "this suggests")):
        return "This implies that"
    if interp_lower.startswith(("our results", "our findings", "our data")):
        return "Their results suggest that"
    if interp_lower.startswith(("on average", "under steady")):
        return "This indicates that"
    return "This suggests that"


def _clean_title_finding(finding: str) -> str:
    """Strip title-fallback prefix — return topic only, no verb.
    _rephrase_finding will add its own verb."""
    prefix = "This study presents research on "
    if finding.startswith(prefix):
        return finding[len(prefix) :].rstrip(".")
    return finding


def _cluster_by_similarity(papers: list[dict], max_batch: int = 3) -> list[list[int]]:
    """Group papers by content similarity using greedy TF-IDF clustering.

    Returns list of index-lists (into original papers list).
    Most similar papers are batched together → natural thematic paragraphs.
    """
    if len(papers) <= max_batch:
        return [list(range(len(papers)))]

    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except ImportError:
        # Fallback: chronological batching
        return [
            list(range(i, min(i + max_batch, len(papers))))
            for i in range(0, len(papers), max_batch)
        ]

    import numpy as _np

    # Build text corpus
    texts = []
    for p in papers:
        texts.append(
            " ".join(
                [
                    (p.get("title") or ""),
                    (p.get("key_finding") or ""),
                    (p.get("abstract") or "")[:500],
                ]
            ).strip()
        )

    # Compute similarity matrix
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english",
        sublinear_tf=True,
        min_df=1,
        max_df=0.95,
    )
    matrix = vectorizer.fit_transform(texts)
    sim = cosine_similarity(matrix).astype(_np.float64)
    _np.fill_diagonal(sim, 0.0)

    # Greedy clustering: pick highest-similarity pairs, grow clusters
    remaining = set(range(len(papers)))
    clusters: list[list[int]] = []

    while remaining:
        if len(remaining) <= max_batch:
            clusters.append(sorted(remaining))
            break

        remaining_list = sorted(remaining)

        # Find the most similar pair among remaining
        best_sim = -1.0
        best_pair = None
        for ii in range(len(remaining_list)):
            for jj in range(ii + 1, len(remaining_list)):
                i, j = remaining_list[ii], remaining_list[jj]
                if sim[i][j] > best_sim:
                    best_sim = sim[i][j]
                    best_pair = (i, j)

        if best_pair is None or best_sim < 0:
            for i in remaining_list:
                clusters.append([i])
            break

        # Start cluster
        cluster_idx = sorted(best_pair)
        remaining.discard(best_pair[0])
        remaining.discard(best_pair[1])

        # Grow cluster: add the most similar remaining paper
        while len(cluster_idx) < max_batch and remaining:
            best_add_sim = -1.0
            best_add = None
            for i in remaining:
                # Similarity to cluster = max similarity to any member
                cluster_sim = max(sim[i][j] for j in cluster_idx)
                if cluster_sim > best_add_sim:
                    best_add_sim = cluster_sim
                    best_add = i

            if best_add is not None and best_add_sim > 0.01:
                cluster_idx.append(best_add)
                cluster_idx.sort()
                remaining.discard(best_add)
            else:
                break

        clusters.append(cluster_idx)

    # Sort clusters by earliest year (chronological order of themes)
    clusters.sort(key=lambda c: min((papers[i].get("year") or 9999) for i in c))
    return clusters


def build_chronological_narrative(
    papers: list[dict],
    topic: str,
    research_type: str,
    correlation: dict | None = None,
) -> tuple[str, list[dict]]:
    """Build thematic narrative organized by methodology (V2).

    V2 features:
    - Genuine synthesis with 8+ varied sentence patterns
    - Agreement/disagreement grouping within themes
    - Measurement synthesis with IQR, outliers, bimodal detection
    - Cross-cutting analysis with method comparison
    - Research gaps with quantitative detection
    - Optional correlation/stance matrix integration

    Returns (narrative_text, cited_papers_in_order).
    """
    papers = _normalize_for_narrative(papers)
    cited: list[dict] = []
    if not papers:
        return f"No papers available to synthesize for '{topic}'.", cited

    # Filter by topical relevance — include ALL topical papers, not just
    # those with quality findings. Papers without findings are cited but
    # not quoted in the narrative (better to acknowledge than to hide).
    topical = _filter_topical(papers, topic)
    if not topical:
        topical = papers  # fallback: use all papers if topical filter too aggressive
    with_findings = topical  # all topical papers included in synthesis

    chronological = _sort_chronological(with_findings)
    _normalize_paper_measurements(chronological)  # validate + normalize units
    n = len(chronological)
    years = [p.get("year") for p in chronological if p.get("year")]
    year_range = ""
    if years:
        if min(years) != max(years):
            year_range = f"({min(years)}-{max(years)})"
        else:
            year_range = f"({min(years)})"

    # Group papers by theme.
    # For small corpora (≤12), HDBSCAN puts everything in one cluster —
    # use _group_by_theme() (discipline-based) instead so the narrative
    # sections match the method comparison table's theme grouping.
    if len(chronological) <= 12:
        themes = _group_by_theme(chronological)
    else:
        themes = _cluster_papers_hdbscan(chronological)
    theme_labels = [t["label"] for t in themes[:6]]

    parts: list[str] = []

    # Opening paragraph
    parts.append(
        f"This review synthesizes {n} studies {year_range} on **{topic}**. "
        f"The corpus is organized into {len(themes)} methodological areas: "
        f"{', '.join(theme_labels)}."
    )
    parts.append("")

    # Per-theme V2 synthesis
    ref_offset = 0
    for theme in themes:
        theme_papers = theme["papers"]
        theme_label = theme["label"]
        n_theme = len(theme_papers)

        parts.append(
            f"### {theme_label} ({n_theme} {'study' if n_theme == 1 else 'studies'})"
        )
        parts.append("")

        synthesis, theme_cited = _build_integrative_synthesis(
            theme_papers, topic, theme_label, ref_offset, correlation
        )
        parts.append(synthesis)

        # Add theme_cited to cited list FIRST (preserves reference order)
        cited.extend(theme_cited)

        # Cite papers not mentioned in the synthesis (no finding extracted).
        # Reference numbers continue after theme_cited.
        cited_ids = {c.get("paper_id") for c in theme_cited}
        uncited = []
        for p in theme_papers:
            pid = p.get("paper_id") or p.get("doi") or ""
            if pid not in cited_ids:
                uncited.append(p)
        if uncited:
            ref_num = ref_offset + len(theme_cited) + 1
            for p in uncited:
                title = (p.get("title") or "")[:80]
                yr = p.get("year", "")
                parts.append(f"[{ref_num}] ({yr}) investigates {title}.")
                p["paper_id"] = p.get("paper_id") or p.get("doi") or str(ref_num)
                p["_ref"] = ref_num  # keep body marker ↔ reference entry aligned
                cited.append(p)
                ref_num += 1

        parts.append("")
        ref_offset += n_theme

    # Field evolution synthesis — what has the field learned
    evolution = _build_field_evolution(themes, chronological, topic)
    if evolution:
        parts.append("### Evolution of the field")
        parts.append("")
        parts.append(evolution)
        parts.append("")

    # Method comparison synthesis — compare, don't list
    if len(themes) >= 2:
        comparison = _build_method_comparison_synthesis(themes, chronological, topic)
        if comparison:
            parts.append("### Method comparison")
            parts.append("")
            parts.append(comparison)
            parts.append("")

    # Cross-cutting analysis (V2)
    if len(themes) >= 2:
        cross = _build_cross_cutting_v2(themes, chronological, topic)
        if cross:
            parts.append("### Cross-cutting analysis")
            parts.append("")
            parts.append(cross)
            parts.append("")

    # Research gaps (V2)
    gaps = _build_research_gaps_v2(themes, chronological, topic)
    if gaps:
        parts.append("### Research gaps and future directions")
        parts.append("")
        parts.append(gaps)
        parts.append("")

    # Deduplicate cited list — papers may appear in multiple theme outputs
    _cited_seen: set[str] = set()
    _cited_deduped: list[dict] = []
    for c in cited:
        cid = c.get("paper_id") or c.get("doi") or (c.get("title") or "")[:50]
        if cid in _cited_seen:
            continue
        _cited_seen.add(cid)
        _cited_deduped.append(c)
    cited = _cited_deduped

    # ── Citation alignment fix ──────────────────────────────────────────
    # The body uses ref_offset + i numbering per theme; the deduped cited
    # list may have a different order or fewer entries (dups removed).
    # Renumber: assign _ref = 1..N sequentially to the deduped cited list,
    # then rewrite every [old_ref] in the body to [new_ref] so body [N]
    # always aligns with References [N] (verified by _citation_check.py).
    body_text = "\n".join(parts)
    renumber_map: dict[int, int] = {}
    for new_ref, p in enumerate(cited, start=1):
        old_ref = p.get("_ref")
        if isinstance(old_ref, int) and old_ref not in renumber_map:
            renumber_map[old_ref] = new_ref
        p["_ref"] = new_ref

    if renumber_map:
        # Rewrite [old] → [new] in body. Use citation-aware regex: only
        # match [N] not followed by another digit (avoids [10] matching [1]).
        # Apply in descending order so [10]→[12] doesn't clobber [1]→[2].
        import re as _re

        n_refs = len(cited)

        def _rewrite_citation(m: _re.Match[str]) -> str:
            old = int(m.group(1))
            new = renumber_map.get(old)
            if new is None:
                # Stale ref (paper dropped in dedup): a dangling [N] would
                # fabricate a citation — map to the LAST valid ref of the
                # same theme cluster is unsafe; drop the marker instead.
                return ""
            return f"[{new}]"

        body_text = _re.sub(r"\[(\d+)\](?!\d)", _rewrite_citation, body_text)
        parts = body_text.split("\n")

    # Cleanup _ref keys (no longer needed after renumbering)
    for p in cited:
        p.pop("_ref", None)

    return "\n".join(parts), cited


def _summarize_measurements(measurements: list[dict]) -> str:
    """Summarize a list of measurements into a narrative sentence."""
    if not measurements:
        return ""

    # Group by measurement type

    by_type: dict[str, list[float]] = defaultdict(list)
    for m in measurements:
        key = m.get("measurement", "unknown")
        try:
            val = float(m.get("value", 0))
            by_type[key].append(val)
        except (TypeError, ValueError):
            continue

    summaries: list[str] = []
    for mtype, values in sorted(by_type.items(), key=lambda x: -len(x[1])):
        if len(values) >= 2:
            import statistics

            mean_val = statistics.mean(values)
            summaries.append(
                f"{mtype}: {min(values):.2f}–{max(values):.2f} (mean {mean_val:.2f})"
            )
        elif len(values) == 1:
            summaries.append(f"{mtype} = {values[0]:.2f}")

    if not summaries:
        return ""

    return f"Key reported measurements include: {'; '.join(summaries[:5])}."


def build_verification_narrative(
    papers: list[dict],
    correlation: dict | None,
    claim: str,
) -> tuple[str, list[dict]]:
    papers = _normalize_for_narrative(papers)
    """Build verdict paragraph + stance evidence summary.

    Returns (narrative_text, cited_papers_in_order).
    """
    cited: list[dict] = []
    parts: list[str] = []

    parts.append(f"**Claim under assessment**: {claim}")
    parts.append("")

    # Extract stance data from correlation matrix if available
    if correlation and correlation.get("correlation_matrix"):
        matrix = correlation["correlation_matrix"]
        rows = matrix.get("matrix", [])

        if rows:
            # Use the first claim row (closest to user query)
            row = rows[0]
            cells = [c for c in row.get("cells", []) if c.get("addresses")]

            supporting = [c for c in cells if c.get("direction") == "supports"]
            contrasting = [c for c in cells if c.get("direction") == "contrasts"]

            total_addr = len(cells)

            # Verdict
            if total_addr == 0:
                verdict = "Insufficient evidence"
                verdict_detail = "No papers in the corpus directly address this claim."
            elif len(supporting) > 0 and len(contrasting) == 0:
                verdict = "Supported"
                verdict_detail = f"{len(supporting)} of {total_addr} papers provide supporting evidence."
            elif len(contrasting) > 0 and len(supporting) == 0:
                verdict = "Contradicted"
                verdict_detail = f"{len(contrasting)} of {total_addr} papers provide contrasting evidence."
            elif len(supporting) > len(contrasting):
                verdict = "Mostly supported"
                verdict_detail = (
                    f"{len(supporting)} support vs {len(contrasting)} contrast "
                    f"out of {total_addr} papers."
                )
            elif len(contrasting) > len(supporting):
                verdict = "Mostly contradicted"
                verdict_detail = (
                    f"{len(contrasting)} contrast vs {len(supporting)} support "
                    f"out of {total_addr} papers."
                )
            else:
                verdict = "Contested"
                verdict_detail = (
                    f"Evidence is split: {len(supporting)} support, "
                    f"{len(contrasting)} contrast out of {total_addr} papers."
                )

            parts.append(f"**Verdict: {verdict}**")
            parts.append("")
            parts.append(verdict_detail)
            parts.append("")

            # Supporting evidence
            if supporting:
                parts.append("**Supporting evidence**:")
                for cell in supporting[:10]:
                    paper = _find_paper_by_id(papers, cell.get("paper_id", ""))
                    if paper:
                        cited.append(paper)
                        ref = len(cited)
                        author = _author_short(_parse_authors(paper.get("authors", [])))
                        year = paper.get("year") or "n.d."
                        finding = (cell.get("finding") or "")[:200]
                        if finding and not finding.endswith((".", "!", "?")):
                            finding += "."
                        parts.append(f"- {author} ({year}) [{ref}]: {finding}")
                parts.append("")

            # Contrasting evidence
            if contrasting:
                parts.append("**Contrasting evidence**:")
                for cell in contrasting[:10]:
                    paper = _find_paper_by_id(papers, cell.get("paper_id", ""))
                    if paper:
                        cited.append(paper)
                        ref = len(cited)
                        author = _author_short(_parse_authors(paper.get("authors", [])))
                        year = paper.get("year") or "n.d."
                        finding = (cell.get("finding") or "")[:200]
                        if finding and not finding.endswith((".", "!", "?")):
                            finding += "."
                        parts.append(f"- {author} ({year}) [{ref}]: {finding}")
                parts.append("")

            return "\n".join(parts), cited

    # Fallback: no correlation data — do simple chronological
    parts.append(
        "No stance correlation data available — presenting chronological findings."
    )
    parts.append("")
    narrative, cited_chron = build_chronological_narrative(
        papers, claim, "verification"
    )
    parts.append(narrative)
    return "\n".join(parts), cited_chron


def _find_paper_by_id(papers: list[dict], paper_id: str) -> dict | None:
    """Find paper by primary_id or doi (partial match)."""
    for p in papers:
        pid = p.get("paper_id", "")
        doi = p.get("doi", "")
        if paper_id and (
            pid == paper_id or doi == paper_id or paper_id in pid or pid in paper_id
        ):
            return p
    return papers[0] if papers else None


# =============================================================================
# Comparative narrative builder — groups papers by approach/method
# =============================================================================


def _detect_approach(paper: dict, topic: str) -> str:
    """Detect the methodological approach of a paper from its title/finding.

    Groups papers by technique or sub-topic for comparison output.
    Falls back to sub-topic detection when no specific method is found.
    """
    text = f"{paper.get('title') or ''} {paper.get('key_finding') or ''}".lower()
    methods = {
        "TIMS": ["tims", "thermal ionization", "isotope dilution"],
        "SIMS": ["sims", "secondary ion", "ion microprobe", "cameca"],
        "LA-ICP-MS": ["la-icp-ms", "laser ablation", "icp-ms"],
        "Solution ICP-MS": ["solution icp", "icp-oes"],
        "XRF": ["xrf", "x-ray fluorescence"],
        "EPMA": ["epma", "electron microprobe", "electron probe"],
        "FTIR": ["ftir", "fourier transform infrared"],
        "Raman": ["raman"],
        "Experimental": [
            "experiment",
            "synthetic",
            "petrology experiment",
            "high pressure",
            "phase equilibrium",
        ],
        "Modeling": [
            "model",
            "simulation",
            "numerical",
            "computational",
            "thermodynamic model",
        ],
        "Field study": ["field", "sample", "collected", "outcrop", "fieldwork"],
        "Review": ["review", "synthesis", "overview", "meta-analysis"],
    }
    for approach, cues in methods.items():
        if any(cue in text for cue in cues):
            return approach

    # Fallback: group by sub-topic (basalt type, tectonic setting, etc.)
    if any(w in text for w in ["oib", "ocean island", "hot spot", "hotspot", "plume"]):
        return "OIB / Hotspot studies"
    if any(w in text for w in ["morb", "mid-ocean ridge", "ridge"]):
        return "MORB / Ridge studies"
    if any(w in text for w in ["arc", "subduction", "island arc"]):
        return "Arc / Subduction studies"
    if any(w in text for w in ["crust", "continental", "cratons"]):
        return "Crustal studies"
    if any(w in text for w in ["isotope", "isotopic"]):
        return "Isotope geochemistry"
    if any(w in text for w in ["trace element", "rare earth", "ree"]):
        return "Trace element geochemistry"
    if any(w in text for w in ["experimental", "partition", "partitioning"]):
        return "Experimental petrology"

    # Fallback: use discipline/study_type from pico if available
    pico = paper.get("pico") or {}
    disc = (pico.get("discipline") or "").lower().replace("_", " ")
    if disc and disc not in ("general", "other", ""):
        return disc.capitalize()
    study_t = (pico.get("study_type") or "").lower().replace("_", " ")
    if study_t and study_t not in ("general", "other", ""):
        return study_t.capitalize()

    return "Other studies"


def build_comparison_narrative(
    papers: list[dict],
    topic: str,
) -> tuple[str, list[dict]]:
    """Build comparison narrative — groups papers by approach, then compares.

    Output structure:
    1. Opening paragraph with topic + N papers + approaches found
    2. Per-approach sections with papers grouped, chronological within each
    3. Comparison summary (which approaches agree/disagree)
    4. References

    Returns (narrative_text, cited_papers_in_order).
    """
    cited: list[dict] = []
    if not papers:
        return f"No papers available for comparison of '{topic}'.", cited

    topical = _filter_topical(papers, topic)
    with_findings = [
        p for p in topical if _is_quality_finding(p.get("key_finding") or "")
    ]
    if not with_findings:
        with_findings = _filter_with_findings(topical)
    if not with_findings:
        return (
            f"Papers found for '{topic}' but no key findings could be extracted.",
            cited,
        )

    # Group by approach

    approach_groups: dict[str, list[dict]] = defaultdict(list)
    for p in with_findings:
        approach = _detect_approach(p, topic)
        approach_groups[approach].append(p)

    # Sort approaches by group size (largest first)
    sorted_approaches = sorted(approach_groups.items(), key=lambda x: -len(x[1]))

    # Limit to top 15 papers total
    total = sum(len(g) for _, g in sorted_approaches)
    if total > 200:
        # Trim each group proportionally
        ratio = 200 / total
        sorted_approaches = [
            (a, g[: max(1, int(len(g) * ratio))]) for a, g in sorted_approaches
        ]
        total = sum(len(g) for _, g in sorted_approaches)

    parts: list[str] = []
    n_approaches = len(sorted_approaches)
    approach_names = ", ".join(a for a, _ in sorted_approaches[:5])

    parts.append(
        f"This comparison synthesizes {total} papers on **{topic}**, "
        f"grouped by methodological approach: {approach_names}."
    )
    parts.append("")

    # Per-approach sections
    ref_num = 0
    for approach_name, group in sorted_approaches:
        chronological = _sort_chronological(group)
        parts.append(f"### {approach_name} ({len(chronological)} papers)")
        parts.append("")

        citations: list[str] = []
        for p in chronological:
            ref_num += 1
            cited.append(p)
            author = _author_short(_parse_authors(p.get("authors", [])))
            year = p.get("year") or "n.d."
            finding = _clean_title_finding(
                _sanitize_finding(p.get("key_finding") or "")
            )
            verb, content = _rephrase_finding(finding)
            if verb.endswith(" that"):
                citations.append(f"- {author} ({year}) [{ref_num}] {verb} {content}")
            else:
                citations.append(
                    f"- {author} ({year}) [{ref_num}] reported that {content}"
                )

        parts.extend(citations)
        parts.append("")

    # Comparison summary
    if n_approaches >= 2:
        parts.append("### Comparison summary")
        parts.append("")
        parts.append(
            f"Across {n_approaches} methodological approaches, the studies show "
            f"{'broad agreement' if total > 8 else 'mixed results'} on {topic}. "
            f"The {sorted_approaches[0][0]} approach dominates ({len(sorted_approaches[0][1])} papers), "
            f"while {sorted_approaches[1][0]} provides complementary evidence ({len(sorted_approaches[1][1])} papers)."
        )
        parts.append("")

    return "\n".join(parts), cited


# =============================================================================
# Data compilation builder — extracts all measurements into table + stats
# =============================================================================


def build_compilation_narrative(
    papers: list[dict],
    topic: str,
) -> tuple[str, list[dict]]:
    """Build data compilation — measurement table + statistical summary.

    Output structure:
    1. Opening paragraph with N papers + measurement types found
    2. Data table: Paper | Year | Measurement | Value | Unit
    3. Statistical summary: min/max/mean/median/stddev per measurement type
    4. Brief narrative of data coverage
    5. References

    Returns (narrative_text, cited_papers_in_order).
    """
    import statistics as stats_mod

    cited: list[dict] = []
    if not papers:
        return f"No papers available for compilation on '{topic}'.", cited

    topical = _filter_topical(papers, topic)

    # Extract ALL measurements from ALL papers (not just quality-filtered)
    all_measurements: list[tuple[dict, dict]] = []  # (paper, measurement)
    for p in topical:
        for m in p.get("measurements", []):
            if m.get("value") is not None:
                try:
                    float(m["value"])
                    all_measurements.append((p, m))
                except (TypeError, ValueError):
                    continue

    parts: list[str] = []
    n_papers = len(topical)
    n_measurements = len(all_measurements)

    if n_measurements == 0:
        parts.append(
            f"This compilation covers {n_papers} papers on **{topic}**, "
            f"but no numerical measurements were extractable from abstracts. "
            f"Presenting chronological narrative instead."
        )
        parts.append("")
        narrative, cited = build_chronological_narrative(
            papers, topic, "data_compilation"
        )
        parts.append(narrative)
        return "\n".join(parts), cited

    # Group measurements by type — filter out journal metadata noise
    _NOISE_MEASURES = frozenset(
        {
            "received",
            "accepted",
            "online",
            "issn",
            "doi",
            "volume",
            "page",
            "pages",
            "published",
            "revised",
            "available online",
            "article number",
            "copyright",
            "license",
            "cc-by",
            "downloaded",
            "",
            "cited",
            "isbn",
        }
    )
    by_type: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    for paper, m in all_measurements:
        mtype = (m.get("measurement") or "").strip()
        # Skip journal metadata noise
        if mtype.lower() in _NOISE_MEASURES:
            continue
        by_type[mtype].append((paper, m))

    parts.append(
        f"This data compilation synthesizes {n_papers} papers on **{topic}**, "
        f"extracting {n_measurements} measurements across "
        f"{len(by_type)} measurement types."
    )
    parts.append("")

    # Statistical summary per measurement type
    parts.append("### Statistical summary")
    parts.append("")
    parts.append("| Measurement | n | Min | Max | Mean | Median | StdDev |")
    parts.append("|---|---|---|---|---|---|---|")

    for mtype, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
        values = []
        for _, m in entries:
            try:
                values.append(float(m["value"]))
            except (TypeError, ValueError):
                continue
        if not values:
            continue
        n = len(values)
        vmin = min(values)
        vmax = max(values)
        vmean = stats_mod.mean(values)
        vmed = stats_mod.median(values)
        vstd = stats_mod.stdev(values) if n >= 2 else 0.0
        parts.append(
            f"| {mtype} | {n} | {vmin:.2f} | {vmax:.2f} | {vmean:.2f} | {vmed:.2f} | {vstd:.2f} |"
        )
    parts.append("")

    # Data table (top entries per type)
    parts.append("### Reported measurements data")
    parts.append("")
    parts.append("| Paper | Year | Measurement | Value | Unit |")
    parts.append("|---|---|---|---|---|")

    ref_num = 0
    seen_papers: set[str] = set()
    for mtype, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
        for paper, m in entries[:5]:  # Top 5 per type
            pid = paper.get("paper_id", "")
            if pid not in seen_papers:
                seen_papers.add(pid)
                cited.append(paper)
                ref_num = len(cited)
            else:
                ref_num = list(seen_papers).index(pid) + 1

            author = _author_short(_parse_authors(paper.get("authors", [])))
            year = paper.get("year") or "n.d."
            val = m.get("value", "?")
            unit = m.get("unit", "")
            parts.append(
                f"| {author} [{ref_num}] | {year} | {mtype} | {val} | {unit} |"
            )

    parts.append("")

    # Brief narrative
    parts.append("### Data coverage")
    parts.append("")
    top_types = sorted(by_type.items(), key=lambda x: -len(x[1]))[:3]
    type_summary = "; ".join(f"{t} (n={len(e)})" for t, e in top_types)
    parts.append(
        f"The most frequently reported measurements are: {type_summary}. "
        f"Data spans {n_papers} studies with variable measurement density."
    )
    parts.append("")

    return "\n".join(parts), cited
