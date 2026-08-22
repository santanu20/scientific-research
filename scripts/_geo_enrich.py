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



# Pre-computed embeddings cache
_theme_embeddings: list | None = None
_theme_labels: list[str] | None = None


# Geological theme vocabulary + BGE matcher removed — zero callers after the
# dynamic-context relabeling (2026-08-22).

# Restored from git history — live dependencies of interpret_pt_data
# (collateral of the 2026-08-22 dead-theme-vocabulary removal).

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


def _lookup(val: float, table: list[tuple[float, float, str]]) -> str:
    for low, high, name in table:
        if low <= val < high:
            return name


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
                lo, hi = min(inlier_vals), max(inlier_vals)
                unit = "°C" if label == "temperature" else " kbar"
                rel_spread = (hi - lo) / median if median else 0.0
                if rel_spread <= 0.5:
                    parts.append(
                        f"Within **{theme}**, {len(inliers)}/{len(vals)} studies "
                        f"report {label} values clustering around "
                        f"{lo:.0f}–{hi:.0f}{unit} (median {median:.0f}{unit}), "
                        f"indicating consistency across studies. "
                    )
                else:
                    # A 250-1600 span is NOT "robust reproducibility" — wide
                    # inlier ranges reflect genuinely heterogeneous conditions.
                    parts.append(
                        f"Within **{theme}**, {label} values span "
                        f"{lo:.0f}–{hi:.0f}{unit} (median {median:.0f}{unit}) "
                        f"across {len(inliers)}/{len(vals)} studies — a "
                        f"heterogeneous range reflecting differing rock types, "
                        f"calibrations, or geological processes, not a single "
                        f"reproducible value. "
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
        parts.extend(offsets)

    return " ".join(parts) if parts else ""


# Known systematic offsets — generalizable pattern matching
_KNOWN_OFFSETS: list[tuple[list[str], str]] = [
    (
        ["ferry", "holdaway"],
        ("Ferry-Spear vs Holdaway garnet-biotite calibrations "
        "show a known ~50-80°C systematic offset [Spear, 1993]. "),
    ),
    (
        ["laiu", "lai-icp"],
        ("LA-ICP-MS vs SIMS trace element data may show "
        "systematic differences due to spatial resolution and matrix effects. "),
    ),
    (
        ["mass-spec", "thermal ionization"],
        ("TIMS vs LA-ICP-MS U-Pb ages show "
        "different precision (TIMS ±0.1% vs LA-ICP-MS ±2%) — comparison requires "
        "method-aware interpretation. "),
    ),
    (
        ["grt-bt", "grt-cpx"],
        ("Garnet-biotite vs garnet-clinopyroxene thermometers "
        "may yield systematically different temperatures due to different Fe-Mg "
        "exchange kinetics. "),
    ),
]


def _detect_known_offsets(text: str) -> list[str]:
    """Detect mentions of known systematic calibration offsets."""
    offsets = []
    for keywords, desc in _KNOWN_OFFSETS:
        if all(kw in text for kw in keywords):
            offsets.append(desc)
    return offsets
