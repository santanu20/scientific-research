#!/usr/bin/env python3
"""Field-agnostic effect size parser for scientific-research skill.

Extracts numerical measurements from scientific abstracts across ALL fields:
- Geological: δ18O = +5.2‰, Fe3+/ΣFe = 0.15, T = 1200°C, P = 2.5 GPa, 45.2 ± 0.3 Ma
- Physics: Tc = 92K, σ = 5.2×10⁻²⁶ cm², B = 2.5 T
- Chemistry: yield = 85%, k = 3.2×10⁻³ s⁻¹, λmax = 520 nm
- Biology: 5.2 ± 1.1 cells/mL, 85% survival, p < 0.001
- Clinical: 5.2±1.1 kg vs 4.5±1.0 kg, n=100, 95% CI: 0.3-1.1
- CS: accuracy = 92%, F1 = 0.87, AUC = 0.95

3-layer architecture:
1. Find ALL numbers with context (value, uncertainty, unit, measurement name)
2. Detect comparison markers and pair into intervention vs control
3. Return structured effect sizes
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

log = logging.getLogger("scientific_research.effect_parser")

# =============================================================================
# Regex patterns — field-agnostic
# =============================================================================

# Number with ± uncertainty: "5.2±1.1", "5.2 ± 1.1", "5.2 +1.1/-0.8"
RE_MEAN_SD = re.compile(r"(\d+\.?\d*)\s*[±\+\-]\s*(\d+\.?\d*)")

# Value with CI: "95% CI: 0.3 to 1.1" or "(0.3-1.1)" or "95% CI=0.3–1.1"
RE_CI = re.compile(
    r"(?:95\s*%\s*CI)[:\s=]+(\d+\.?\d*)\s*(?:to|–|-|—)\s*(\d+\.?\d*)",
    re.IGNORECASE,
)
RE_CI_PARENS = re.compile(r"\((\d+\.?\d*)\s*(?:–|-|—|to)\s*(\d+\.?\d*)\)")

# p-value: "p < 0.001", "p=0.05", "P ≤ 0.01"
RE_P_VALUE = re.compile(r"[pP]\s*[<≤>=]\s*(\d+\.?\d*)")

# Sample size: "n=100", "N = 50", "(n=30)"
RE_N_VALUE = re.compile(r"[nN]\s*=\s*(\d+)")

# Percentage: "85%", "85 %"
RE_PERCENTAGE = re.compile(r"(\d+\.?\d*)\s*%")

# Scientific notation: "5.2 × 10⁻²⁶", "3.2 x 10^-3", "1.2e-5"
RE_SCIENTIFIC = re.compile(
    r"(\d+\.?\d*)\s*[×x*]\s*10[⁻^-]?(\d+)",
    re.IGNORECASE,
)

# Isotope delta notation: "δ18O = +5.2‰", "δ¹⁸O = -2.3 ‰"
RE_ISOTOPE_UNICODE = re.compile(
    r"δ[⁰¹²³⁴⁵⁶⁷⁸⁹]+[A-Z][a-z]?\s*[=:]\s*([+-]?\d+\.?\d*)\s*‰", re.IGNORECASE
)
RE_ISOTOPE = re.compile(
    r"(?:δ|delta)\s*\d*\s*[A-Z][a-z]?\s*[=:]\s*([+-]?\d+\.?\d*)\s*(?:‰|per\s+mil)",
    re.IGNORECASE,
)

# Subscript/superscript Unicode maps — ported from pdf-ocr skill
# Used to normalize scientific notation from OCR'd or unicode-rich abstracts
_SUB_DIGITS = "₀₁₂₃₄₅₆₇₈₉"
_SUP_DIGITS = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUB_TO_ASCII = str.maketrans(_SUB_DIGITS, "0123456789")
_SUP_TO_ASCII = str.maketrans(_SUP_DIGITS, "0123456789")

# Chemical formula patterns: "SiO₂", "Al₂O₃", "Fe₂O₃", "H₂O", "CO₂"
# Handles both Unicode subscripts (SiO₂) and ASCII (SiO2)
RE_CHEM_FORMULA = re.compile(
    r"\b([A-Z][a-z]?\d*(?:[₀-₉]*)(?:[A-Z][a-z]?\d*(?:[₀-₉]*))*)\s*[=:]?\s*"
    r"(?:(\d+\.?\d*)\s*(?:wt%|wt\.?\s*%|weight\s*percent|%))",
    re.IGNORECASE,
)

# Oxide weight percent: "SiO2 = 50.3 wt%", "Al2O3: 15.2%", "FeOT = 10.5"
RE_OXIDE_WT = re.compile(
    r"(SiO[₂2]|Al[₂2]O[₃3]|Fe[₂2]O[₃3]|FeO[Tt]?|CaO|MgO|Na[₂2]O|K[₂2]O|"
    r"TiO[₂2]|P[₂2]O[₅5]|MnO|Cr[₂2]O[₃3]|H[₂2]O|CO[₂2])"
    r"\s*[=:]?\s*(\d+\.?\d*)\s*(?:wt%|wt\.?\s*%|%|percent)?",
    re.IGNORECASE,
)

# Scientific notation: "1.23 × 10⁻⁵", "1.23e-5", "1.23E-5", "1.23 x 10^-5"
RE_SCI_NOTATION = re.compile(
    r"(\d+\.?\d*)\s*(?:[×x*]|e|E)\s*(?:10)?[×x]?\s*[?^]?\s*([⁺⁻\-+])?(\d+)",
    re.IGNORECASE,
)

# Isotope ratio with unicode: "⁸⁷Sr/⁸⁶Sr = 0.703", "¹⁴³Nd/¹⁴⁴Nd"
RE_ISOTOPE_RATIO_UNICODE = re.compile(
    r"([⁰¹²³⁴⁵⁶⁷⁸⁹]+[A-Z][a-z]?/[⁰¹²³⁴⁵⁶⁷⁸⁹]+[A-Z][a-z]?)\s*[=:]\s*(\d+\.?\d*)",
)


def _normalize_unicode(text: str) -> str:
    """Convert Unicode subscripts/superscripts to ASCII for consistent parsing.

    Ported from pdf-ocr skill. Handles:
    - ₀₁₂₃₄₅₆₇₈₉ → 0123456789 (subscripts)
    - ⁰¹²³⁴⁵⁶⁷⁸⁹ → 0123456789 (superscripts)
    - δ¹⁸O → delta18O (isotope notation)
    """
    result = text
    # Convert subscripts
    result = result.translate(_SUB_TO_ASCII)
    # Convert superscripts
    result = result.translate(_SUP_TO_ASCII)
    return result


# Ratio notation: "Fe3+/SigmaFe = 0.15", "La/Yb = 5.2", "87Sr/86Sr = 0.703"
RE_RATIO = re.compile(
    r"([A-Z][A-Za-z0-9+\-/Σ]{2,15})\s*[=:]\s*(\d+\.?\d*)",
)

# Temperature: "1200°C", "298 K", "25 °C"
# Temperature: requires context to avoid false positives (not just any number before °C)
# Matches: "temperature = 1200°C", "T = 1200°C", "heated to 1200°C", "at 1200°C"
# Does NOT match: "stored at 3°C", "incubated at 37°C" (unless preceded by temperature/T/heated/measured)
RE_TEMP_C = re.compile(
    r"(?:temperature\s*[=:]|T\s*=|heated\s+to|measured\s+at|oven\s+at|furnace\s+at)"
    r"\s*(\d+\.?\d*)\s*°?\s*[Cc]",
    re.IGNORECASE,
)
RE_TEMP_K = re.compile(
    r"(?:Tc\s*=|temperature\s*[=:]|T\s*=|measured\s+at|transition\s+at)"
    r"\s*(\d+\.?\d*)\s*[Kk]\b",
    re.IGNORECASE,
)

# Pressure: "2.5 GPa", "1 atm", "100 MPa"
RE_PRESSURE = re.compile(r"(\d+\.?\d*)\s*[MG]?Pa", re.IGNORECASE)

# United bare numbers (B5a, 2026-08-15): "750 °C", "6 kbar", "3.2 wt %",
# "12 ‰" — the most common scientific phrasing, previously NOT captured
# (only keyword-prefixed forms were). Runs AFTER specific regexes;
# position-guarded so it never double-captures. Label comes from the
# reconcile pass (unit family default).
RE_UNITED = re.compile(
    r"(\d+\.?\d*)\s{0,2}(°C|°\s?[Cc]|[Cc]elsius|deg\s?[Cc]\b|GPa|MPa|kPa|kbar"
    r"|Ma\b|Ga\b|ka\b|wt\s?%|ppm|ppb|‰|per\s?mil|km\b|mm\b|cm\b|µm|μm|nm\b|%)"
)
_UNIT_CANON = {
    "°c": "°C",
    "° C": "°C",
    "c": "°C",
    "celsius": "°C",
    "deg c": "°C",
    "gpa": "GPa",
    "mpa": "MPa",
    "kpa": "kPa",
    "kbar": "kbar",
    "ma": "Ma",
    "ga": "Ga",
    "ka": "ka",
    "wt%": "wt%",
    "wt %": "wt%",
    "ppm": "ppm",
    "ppb": "ppb",
    "‰": "‰",
    "per mil": "‰",
    "km": "km",
    "mm": "mm",
    "cm": "cm",
    "µm": "µm",
    "μm": "µm",
    "nm": "nm",
    "%": "%",
}

# Age: "45.2 ± 0.3 Ma", "2.5 Ga"
RE_AGE = re.compile(r"(\d+\.?\d*)\s*[MG]a\b", re.IGNORECASE)

# Metric assignment: "accuracy = 92%", "F1 = 0.87", "yield = 85%", "SO2 flux = 500"
RE_METRIC = re.compile(
    r"(accuracy|yield|selectivity|conversion|efficiency|F1|AUC|SO2 flux|flux|emission rate|discharge|"
    r"precision|recall|sensitivity|specificity)\s*[=:]\s*(\d+\.?\d*)\s*%?",
    re.IGNORECASE,
)

# Oxide weight percent (geochemistry): "SiO2 = 50.3 wt%", "Al2O3: 15.2%", "FeOT = 10.5"
# After _normalize_unicode, SiO₂ becomes SiO2
RE_OXIDE = re.compile(
    r"(SiO2|Al2O3|Fe2O3|FeO[Tt]?|CaO|MgO|Na2O|K2O|TiO2|P2O5|MnO|Cr2O3|H2O|CO2|LOI)"
    r"\s*[=:]\s*(\d+\.?\d*)\s*(?:wt%|wt\.?\s*%|%|percent)?",
    re.IGNORECASE,
)

# Comparison markers (field-agnostic)
COMPARISON_PATTERNS = [
    re.compile(r"\bvs\.?\b", re.IGNORECASE),
    re.compile(r"\bversus\b", re.IGNORECASE),
    re.compile(r"\bcompared?\s+(?:to|with)\b", re.IGNORECASE),
    re.compile(r"\brelative\s+to\b", re.IGNORECASE),
    re.compile(r"\bin\s+contrast\s+to\b", re.IGNORECASE),
    re.compile(r"\bcontrol\s+group\b", re.IGNORECASE),
    re.compile(r"\bplacebo\b", re.IGNORECASE),
    re.compile(r"\btreatment\s+group\b", re.IGNORECASE),
    re.compile(r"\bbaseline\b", re.IGNORECASE),
]

# Known measurement labels
MEASUREMENT_LABELS = {
    "accuracy": "accuracy",
    "yield": "yield",
    "selectivity": "selectivity",
    "conversion": "conversion",
    "efficiency": "efficiency",
    "f1": "F1 score",
    "auc": "AUC",
    "precision": "precision",
    "recall": "recall",
    "sensitivity": "sensitivity",
    "specificity": "specificity",
}


@dataclass
class ExtractedNumber:
    """A number extracted from text with its context."""

    value: float
    uncertainty: float | None = None
    unit: str = ""
    measurement: str = ""
    n: int | None = None
    ci_lower: float | None = None
    ci_upper: float | None = None
    p_value: float | None = None
    position: int = 0  # character position in text
    raw_text: str = ""
    # R4 (2026-08-15): True when this number is one BOUND of a range/interval
    # ("cooling interval of 190–270 °C", "1200–1350°C experiments"). Interval
    # bounds are NOT point measurements — pooling them as points fabricated
    # phantom 190 °C / 1350 °C "temperatures" in the E2E brief.
    is_interval_bound: bool = False


def _find_all_numbers(text: str) -> list[ExtractedNumber]:
    """Layer 1: Find ALL numbers in text with their context.

    Extracts: mean±SD, CI, p-value, n, percentage, scientific notation,
    isotope ratios, temperatures, pressures, ages, and named metrics.
    """
    results: list[ExtractedNumber] = []

    # Mean ± SD
    for m in RE_MEAN_SD.finditer(text):
        val = float(m.group(1))
        sd = float(m.group(2))
        # Get surrounding context for unit/measurement
        context = text[max(0, m.start() - 40) : m.end() + 20]
        unit = _detect_unit(context)
        measurement = _detect_measurement(context)
        results.append(
            ExtractedNumber(
                value=val,
                uncertainty=sd,
                unit=unit,
                measurement=measurement,
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # CI
    for m in RE_CI.finditer(text):
        lo, hi = float(m.group(1)), float(m.group(2))
        results.append(
            ExtractedNumber(
                value=(lo + hi) / 2,
                ci_lower=lo,
                ci_upper=hi,
                unit="CI",
                measurement="mean difference",
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Percentage
    for m in RE_PERCENTAGE.finditer(text):
        val = float(m.group(1))
        context = text[max(0, m.start() - 30) : m.end()]
        measurement = _detect_measurement(context)
        # Skip if already captured as part of mean±SD
        if not any(
            abs(r.value - val) < 0.01 and r.position == m.start() for r in results
        ):
            results.append(
                ExtractedNumber(
                    value=val,
                    unit="%",
                    measurement=measurement,
                    position=m.start(),
                    raw_text=m.group(0),
                )
            )

    # Scientific notation
    for m in RE_SCIENTIFIC.finditer(text):
        mantissa = float(m.group(1))
        exp = int(m.group(2))
        val = mantissa * (10**-exp)
        context = text[max(0, m.start() - 30) : m.end() + 20]
        unit = _detect_unit(context)
        measurement = _detect_measurement(context)
        results.append(
            ExtractedNumber(
                value=val,
                unit=unit,
                measurement=measurement,
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Isotope delta (both ASCII δ18O and unicode δ¹⁸O)
    for pattern in [RE_ISOTOPE, RE_ISOTOPE_UNICODE]:
        for m in pattern.finditer(text):
            val = float(m.group(1))
            element_match = re.search(
                r"δ\s*(\d*\s*[A-Z][a-z]?)", text[max(0, m.start() - 10) : m.end()]
            )
            element = element_match.group(1).strip() if element_match else "isotope"
            results.append(
                ExtractedNumber(
                    value=val,
                    unit="‰",
                    measurement=f"δ{element}",
                    position=m.start(),
                    raw_text=m.group(0),
                )
            )

    # Ratio
    for m in RE_RATIO.finditer(text):
        ratio_name = m.group(1)
        val = float(m.group(2))
        results.append(
            ExtractedNumber(
                value=val,
                unit="ratio",
                measurement=ratio_name,
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Named metrics (accuracy, yield, F1, AUC, etc.)
    for m in RE_METRIC.finditer(text):
        name = m.group(1).lower()
        val = float(m.group(2))
        unit = "%" if "%" in m.group(0) else ""
        results.append(
            ExtractedNumber(
                value=val,
                unit=unit,
                measurement=MEASUREMENT_LABELS.get(name, name),
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Oxide weight percent (geochemistry): SiO2, Al2O3, FeO, etc.
    for m in RE_OXIDE.finditer(text):
        oxide = m.group(1).upper()
        val = float(m.group(2))
        unit = "wt%" if "%" in m.group(0).lower() else "wt%"
        results.append(
            ExtractedNumber(
                value=val,
                unit=unit,
                measurement=f"{oxide}",
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # p-values
    for m in RE_P_VALUE.finditer(text):
        p = float(m.group(1))
        results.append(
            ExtractedNumber(
                value=p,
                p_value=p,
                unit="",
                measurement="p-value",
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Sample size
    for m in RE_N_VALUE.finditer(text):
        n = int(m.group(1))
        results.append(
            ExtractedNumber(
                value=float(n),
                n=n,
                unit="",
                measurement="sample size",
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # Temperature
    for m in RE_TEMP_C.finditer(text):
        val = float(m.group(1))
        # Skip if already captured
        if not any(
            abs(r.value - val) < 0.1 and r.position == m.start() for r in results
        ):
            results.append(
                ExtractedNumber(
                    value=val,
                    unit="°C",
                    measurement="temperature",
                    position=m.start(),
                    raw_text=m.group(0),
                )
            )
    for m in RE_TEMP_K.finditer(text):
        val = float(m.group(1))
        if val > 10:  # avoid matching "k" as in "kPa"
            if not any(
                abs(r.value - val) < 0.1 and r.position == m.start() for r in results
            ):
                results.append(
                    ExtractedNumber(
                        value=val,
                        unit="K",
                        measurement="temperature",
                        position=m.start(),
                        raw_text=m.group(0),
                    )
                )

    # Age
    for m in RE_AGE.finditer(text):
        val = float(m.group(1))
        unit = "Ga" if "Ga" in m.group(0) else "Ma"
        results.append(
            ExtractedNumber(
                value=val,
                unit=unit,
                measurement="age",
                position=m.start(),
                raw_text=m.group(0),
            )
        )

    # United bare numbers ("750 °C", "6 kbar") — position-guarded catch-all
    for m in RE_UNITED.finditer(text):
        val = float(m.group(1))
        raw_unit = m.group(2).strip()
        canon = _UNIT_CANON.get(raw_unit.lower()) or _UNIT_CANON.get(raw_unit) or ""
        if not canon:
            continue
        # skip if a specific regex already captured this exact number+position
        if any(abs(r.value - val) < 1e-9 and r.position == m.start(1) for r in results):
            continue
        results.append(
            ExtractedNumber(
                value=val,
                unit=canon,
                measurement="",  # filled by reconcile from unit family
                position=m.start(1),
                raw_text=m.group(0),
            )
        )

    # Sort by position
    results.sort(key=lambda r: r.position)

    # R4: mark interval/range bounds: number-dash-number-unit spans
    # (e.g. 1200-1350 C, 0.5-1.0 GPa, 190-270 C). Both endpoint numbers
    # become is_interval_bound=True and are excluded from point pools,
    # because they describe a RANGE, not a measured value.
    _range_re = re.compile(
        r"(\d+\.?\d*)\s*[\u2013\u2014-]\s*(\d+\.?\d*)\s*([°A-Za-zµ%‰]*)"
    )
    span_by_val: dict[float, bool] = {}
    for m in _range_re.finditer(text):
        try:
            lo, hi = float(m.group(1)), float(m.group(2))
        except ValueError:
            continue
        if lo == hi:
            continue
        # unit right after range OR shared family unit on either side
        u_after = (m.group(3) or "").strip()
        ctx = text[max(0, m.start() - 8) : m.end() + 8]
        looks_ranged = bool(u_after) or _detect_unit(ctx)
        if looks_ranged:
            span_by_val[lo] = True
            span_by_val[hi] = True
    for r in results:
        if span_by_val.get(r.value):
            r.is_interval_bound = True

    _reconcile_units_and_measurements(results, text)
    return results


# =============================================================================
# Unit/measurement reconciliation (B5a fix, 2026-08-15)
# Wide-context detection lets NEIGHBORING numbers' units/labels leak in
# ("temperature ... 5 kbar" → 5 kbar bound to "temperature"). Two-part fix:
#   1. rebind unit from the TIGHTEST adjacent window (after, then before)
#   2. enforce unit-family compatibility: unit wins conflicts (locally bound
#      and more reliable than a label phrase up to 40 chars away)
# =============================================================================
_UNIT_FAMILIES: dict[str, set[str]] = {
    "temperature": {"°C", "K"},
    "pressure": {"GPa", "MPa", "kPa", "kbar", "bar"},
    "age": {"Ma", "Ga", "ka"},
    "fraction": {"%", "wt%", "‰", "‰ VSMOW", "fold"},
    "length": {"km", "m", "cm", "mm", "µm", "nm"},
    "concentration": {"ppm", "ppb", "mol/L", "mM"},
}

_FAMILY_DEFAULT_MEASUREMENT = {
    "temperature": "temperature",
    "pressure": "pressure",
    "age": "age",
    "fraction": "percentage/composition",
    "length": "distance",
    "concentration": "concentration",
}

_MEASUREMENT_FAMILY = {
    "temperature": "temperature",
    "pressure": "pressure",
    "age": "age",
    "melt composition": "fraction",
    "yield": "fraction",
    "conversion": "fraction",
    "efficiency": "fraction",
    "accuracy": "fraction",
    "survival rate": "fraction",
    "mortality": "fraction",
    "weight loss": "fraction",
    "weight gain": "fraction",
}


def _unit_family(unit: str) -> str | None:
    for family, members in _UNIT_FAMILIES.items():
        if unit in members:
            return family
    return None


def _rebind_unit_adjacent(text: str, start: int, end: int) -> str:
    """Unit from the token(s) immediately after (then before) the number.

    15-char window: long enough for '°C', ' wt %', ' kbar'; too short for a
    neighboring number's unit to leak in. Returns '' when nothing adjacent.
    """
    after = text[end : end + 15]
    u = _detect_unit(after)
    if u and (
        after.lstrip().lower().startswith(u.lower())
        or "%" in after[:3]
        or "‰" in after[:3]
    ):
        return u
    before = text[max(0, start - 15) : start]
    u2 = _detect_unit(before)
    if u2 and (
        before.rstrip().lower().endswith(u2.lower())
        or "%" in before[-3:]
        or "‰" in before[-3:]
    ):
        return u2
    return ""


def _reconcile_units_and_measurements(numbers: list, text: str) -> None:
    """In-place: adjacent unit rebind + family-compatibility remap."""
    for n in numbers:
        if n.measurement in ("p-value", "sample size", "mean difference"):
            continue
        adj_unit = _rebind_unit_adjacent(text, n.position, n.position + len(n.raw_text))
        if adj_unit:
            n.unit = adj_unit
        fam_u = _unit_family(n.unit)
        fam_m = _MEASUREMENT_FAMILY.get(n.measurement or "")
        # incompatible label vs unit → trust the unit, remap the label
        if fam_u and fam_m and fam_u != fam_m or not n.measurement and fam_u:
            n.measurement = _FAMILY_DEFAULT_MEASUREMENT[fam_u]
        # unit in a family but label still outside mapping? leave as-is


def _detect_unit(context: str) -> str:
    """Detect the unit from surrounding context."""
    context_lower = context.lower()
    units = [
        ("‰", "‰"),
        ("vsmow", "‰ VSMOW"),
        ("‰ vsmow", "‰ VSMOW"),
        ("gpa", "GPa"),
        ("mpa", "MPa"),
        ("kpa", "kPa"),
        ("°c", "°C"),
        ("deg c", "°C"),
        ("celsius", "°C"),
        ("ma", "Ma"),
        ("ga", "Ga"),
        ("wt%", "wt%"),
        ("wt %", "wt%"),
        ("ppm", "ppm"),
        ("ppb", "ppb"),
        ("kg", "kg"),
        ("mg", "mg"),
        ("ml", "mL"),
        ("µl", "µL"),
        ("mm", "mm"),
        ("cm", "cm"),
        ("nm", "nm"),
        ("µm", "µm"),
        ("km", "km"),
        ("ev", "eV"),
        ("mev", "MeV"),
        ("gev", "GeV"),
        ("mol/l", "mol/L"),
        ("mmol", "mM"),
        ("mM".lower(), "mM"),
        ("kj/mol", "kJ/mol"),
        ("s-1", "s⁻¹"),
        ("s^-1", "s⁻¹"),
        ("hz", "Hz"),
        ("khz", "kHz"),
        ("mhz", "MHz"),
        ("ghz", "GHz"),
        ("tesla", "T"),
        (" t\b", "T"),
        ("fold", "fold"),
        ("cells/ml", "cells/mL"),
        ("cells/μl", "cells/µL"),
        ("bar", "bar"),
        ("%", "%"),
    ]
    for pattern, unit in units:
        if pattern in context_lower:
            return unit
    return ""


def _detect_measurement(context: str) -> str:
    """Detect what is being measured from surrounding context."""
    context_lower = context.lower()
    measurements = [
        ("weight loss", "weight loss"),
        ("weight gain", "weight gain"),
        ("bmi", "BMI"),
        ("blood pressure", "blood pressure"),
        ("heart rate", "heart rate"),
        ("temperature", "temperature"),
        ("pressure", "pressure"),
        ("yield", "yield"),
        ("selectivity", "selectivity"),
        ("conversion", "conversion"),
        ("efficiency", "efficiency"),
        ("accuracy", "accuracy"),
        ("precision", "precision"),
        ("auc", "AUC"),
        ("f1", "F1 score"),
        ("age", "age"),
        ("duration", "duration"),
        ("concentration", "concentration"),
        ("δ18o", "δ18O"),
        ("δ13c", "δ13C"),
        ("δd", "δD"),
        ("δ15n", "δ15N"),
        ("δ34s", "δ34S"),
        ("fe3+", "Fe3+/ΣFe"),
        ("fe2+", "Fe2+"),
        ("redox", "redox state"),
        ("fugacity", "oxygen fugacity"),
        ("melt", "melt composition"),
        ("conductivity", "conductivity"),
        ("resistivity", "resistivity"),
        ("magnetization", "magnetization"),
        ("absorbance", "absorbance"),
        ("emission", "emission"),
        ("survival", "survival rate"),
        ("mortality", "mortality"),
        ("expression", "gene expression"),
    ]
    for pattern, name in measurements:
        if pattern in context_lower:
            return name
    return ""


def _find_comparisons(text: str, numbers: list[ExtractedNumber]) -> list[dict]:
    """Layer 2: Detect comparison markers and pair numbers into groups.

    Finds "vs", "compared to", "control vs treatment" patterns
    and pairs numbers on either side.
    """
    pairs = []

    for pattern in COMPARISON_PATTERNS:
        for m in pattern.finditer(text):
            split_pos = m.start()

            # Numbers before the comparison marker = group 1
            group1 = [n for n in numbers if n.position < split_pos]
            # Numbers after = group 2
            group2 = [n for n in numbers if n.position >= m.end()]

            # Take the nearest number from each side (within 100 chars)
            g1_candidates = [n for n in group1 if split_pos - n.position < 200]
            g2_candidates = [n for n in group2 if n.position - m.end() < 200]

            if g1_candidates and g2_candidates:
                # Take the closest meaningful number from each side
                # Prefer numbers with uncertainty (mean±SD) or named metrics
                def best_candidate(candidates, ref_pos):
                    # Sort by distance to reference position
                    scored = sorted(candidates, key=lambda n: abs(n.position - ref_pos))
                    # Prefer numbers with uncertainty
                    with_unc = [n for n in scored if n.uncertainty is not None]
                    if with_unc:
                        return with_unc[0]
                    return scored[0] if scored else None

                g1 = best_candidate(g1_candidates, split_pos)
                g2 = best_candidate(g2_candidates, m.end())

                if g1 and g2:
                    # Extract n from nearby "n=X" pattern
                    n1 = _find_nearest_n(text, g1.position)
                    n2 = _find_nearest_n(text, g2.position)
                    # Get p-value from nearby
                    p_val = _find_nearest_p(text, g1.position, g2.position)

                    pairs.append(
                        {
                            "m1": g1.value,
                            "sd1": g1.uncertainty,
                            "n": n1 or g1.n,
                            "m2": g2.value,
                            "sd2": g2.uncertainty,
                            "n2": n2 or g2.n,
                            "ci_lower": g1.ci_lower or g2.ci_lower,
                            "ci_upper": g1.ci_upper or g2.ci_upper,
                            "p_value": p_val,
                            "outcome": g1.measurement or g2.measurement or "comparison",
                            "unit": g1.unit or g2.unit,
                            "comparison": m.group(0).strip(),
                        }
                    )

    return pairs


def _find_nearest_n(text: str, pos: int, window: int = 100) -> int | None:
    """Find the nearest 'n=X' value to the given position."""
    search_start = max(0, pos - window)
    search_end = min(len(text), pos + window)
    segment = text[search_start:search_end]
    matches = list(RE_N_VALUE.finditer(segment))
    if matches:
        return int(matches[-1].group(1))  # take the last one in the segment
    return None


def _find_nearest_p(text: str, pos1: int, pos2: int, window: int = 150) -> float | None:
    """Find the nearest p-value to the given positions."""
    center = (pos1 + pos2) // 2
    search_start = max(0, center - window)
    search_end = min(len(text), center + window)
    segment = text[search_start:search_end]
    matches = list(RE_P_VALUE.finditer(segment))
    if matches:
        return float(matches[0].group(1))
    return None


def extract_effect_sizes(text: str) -> dict:
    """Extract all effect sizes from text.

    Returns dict with:
    - mean_sd_groups: paired comparisons (intervention vs control)
    - single_measurements: standalone numerical values
    - p_values: all p-values found
    """
    if not text:
        return {"mean_sd_groups": [], "single_measurements": [], "p_values": []}

    # Normalize Unicode subscripts/superscripts first (δ¹⁸O → delta18O, SiO₂ → SiO2)
    text = _normalize_unicode(text)

    # Layer 1: Find all numbers
    numbers = _find_all_numbers(text)

    # Layer 2: Find comparisons + pair groups
    pairs = _find_comparisons(text, numbers)

    # Layer 3: Single measurements (not part of any pair)
    paired_positions = set()
    for pair in pairs:
        # Mark positions used in pairs (approximate)
        pass

    singles = []
    for n in numbers:
        # Skip p-values and sample sizes from singles
        if n.measurement in ("p-value", "sample size"):
            continue
        # R4: interval bounds are not point measurements — never pool
        if getattr(n, "is_interval_bound", False):
            continue
        # Skip if this number is part of a pair (check by value proximity)
        in_pair = any(
            abs(pair.get("m1", 999) - n.value) < 0.01
            or abs(pair.get("m2", 999) - n.value) < 0.01
            for pair in pairs
        )
        if not in_pair:
            # B5a: a number with NO unit and NO measurement label is unbound
            # noise — keeping it poisons downstream pooling with phantom rows
            if not n.unit and not n.measurement:
                continue
            singles.append(
                {
                    "value": n.value,
                    "uncertainty": n.uncertainty,
                    "unit": n.unit,
                    "measurement": n.measurement,
                    "n": n.n,
                    "ci_lower": n.ci_lower,
                    "ci_upper": n.ci_upper,
                }
            )

    # Extract p-values
    p_values = [float(m.group(1)) for m in RE_P_VALUE.finditer(text)]

    return {
        "mean_sd_groups": pairs,
        "single_measurements": singles,
        "p_values": p_values,
    }


if __name__ == "__main__":
    # Test on different scientific texts
    test_texts = [
        (
            "Clinical",
            "The treatment group lost 5.2±1.1 kg vs 4.5±1.0 kg in controls (n=100 per group, p=0.001, 95% CI: 0.3-1.1).",
        ),
        (
            "Geological",
            "The δ18O values range from +5.2‰ to +8.1‰ VSMOW. Fe3+/ΣFe = 0.15 ± 0.02. Temperature = 1200°C at 2.5 GPa.",
        ),
        (
            "Chemistry",
            "The reaction yield = 85% with selectivity >95%. Rate constant k = 3.2×10⁻³ s⁻¹ at 25°C.",
        ),
        (
            "Physics",
            "Tc = 92K ± 1K for the optimally doped sample. σ = 5.2×10⁻²⁶ cm² at B = 2.5 T.",
        ),
        (
            "CS",
            "Our model achieves accuracy = 92% with F1 = 0.87 and AUC = 0.95, compared to 85% accuracy for the baseline.",
        ),
    ]

    for field_name, text in test_texts:
        result = extract_effect_sizes(text)
        print(f"\n[{field_name}] {text[:70]}...")
        print(f"  Paired groups: {len(result['mean_sd_groups'])}")
        for p in result["mean_sd_groups"]:
            print(
                f"    {p.get('outcome', '?')}: {p.get('m1')}±{p.get('sd1')} vs {p.get('m2')}±{p.get('sd2')} (p={p.get('p_value')})"
            )
        print(f"  Single measurements: {len(result['single_measurements'])}")
        for s in result["single_measurements"][:5]:
            unc = f" ±{s['uncertainty']}" if s.get("uncertainty") else ""
            print(
                f"    {s.get('measurement', '?')}: {s['value']}{unc} {s.get('unit', '')}"
            )
        if result["p_values"]:
            print(f"  p-values: {result['p_values']}")