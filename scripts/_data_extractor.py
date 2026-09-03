"""Domain-specific quantitative data extraction from research papers.

ARCHITECTURE:
    PDF Table Parser (shared)
        → Domain Header Detector (per-domain column dictionaries)
        → Domain Data Model (structured dataclass)
        → Domain Validator (geological plausibility checks)
        → GeoKit Integration (feed to existing calculation engines)

Each domain knows:
  - WHAT tables to look for (header patterns)
  - HOW to parse values (units, formats)
  - WHAT to calculate (GeoKit modules to invoke)
  - WHAT ranges are plausible (validation bounds)

Usage:
    from _data_extractor import extract_domain_data
    data = extract_domain_data(pdf_path, domain="geochemistry")
    # Returns structured GeochemistryData with major_elements, trace_elements, etc.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

log = logging.getLogger("scientific_research.data_extractor")


# =============================================================================
# Domain Definitions — what data to extract for each discipline
# =============================================================================


@dataclass
class DomainSpec:
    """Specification for domain-specific data extraction."""

    name: str
    # Header patterns that identify this domain's tables
    table_headers: dict[str, list[str]]
    # Plausible value ranges [min, max] for validation
    valid_ranges: dict[str, tuple[float, float]]
    # Unit variants to normalize (e.g., "wt%" → "%", "ppm" → "ppm")
    unit_aliases: dict[str, str] = field(default_factory=dict)
    # GeoKit module to invoke for calculations
    calc_module: str = ""
    # GeoKit function to call with extracted data
    calc_function: str = ""


# ── Geochemistry ─────────────────────────────────────────────────────────────
GEOCHEMISTRY = DomainSpec(
    name="geochemistry",
    table_headers={
        # Major element oxides (wt%)
        "SiO2": ["sio2", "sio₂", "silica", "sio2"],
        "TiO2": ["tio2", "tio₂", "titanium dioxide"],
        "Al2O3": ["al2o3", "al₂o₃", "alumina", "aluminium oxide"],
        "Fe2O3": ["fe2o3", "fe₂o₃", "ferric oxide", "fe2o3t", "fe2o3*"],
        "FeO": ["feo", "ferrous oxide"],
        "Fe2O3T": ["fe2o3t", "fe2o3*", "feot", "fe₂o₃t", "total iron as fe2o3"],
        "MnO": ["mno", "manganese oxide"],
        "MgO": ["mgo", "magnesia", "magnesium oxide"],
        "CaO": ["cao", "lime", "calcium oxide"],
        "Na2O": ["na2o", "na₂o", "soda", "sodium oxide"],
        "K2O": ["k2o", "k₂o", "potash", "potassium oxide"],
        "P2O5": ["p2o5", "p₂o₅", "phosphorus pentoxide"],
        "LOI": ["loi", "loss on ignition"],
        "Total": ["total", "sum"],
        # Trace elements (ppm)
        "Ba": ["ba", "barium"],
        "Rb": ["rb", "rubidium"],
        "Sr": ["sr", "strontium"],
        "Y": ["y", "yttrium"],
        "Zr": ["zr", "zirconium"],
        "Nb": ["nb", "niobium"],
        "La": ["la", "lanthanum"],
        "Ce": ["ce", "cerium"],
        "Nd": ["nd", "neodymium"],
        "Sm": ["sm", "samarium"],
        "Eu": ["eu", "europium"],
        "Gd": ["gd", "gadolinium"],
        "Dy": ["dy", "dysprosium"],
        "Yb": ["yb", "ytterbium"],
        "Lu": ["lu", "lutetium"],
        "Th": ["th", "thorium"],
        "U": ["u", "uranium"],
        "Pb": ["pb", "lead"],
        "Cs": ["cs", "cesium"],
        "Cr": ["cr", "chromium"],
        "Ni": ["ni", "nickel"],
        "Co": ["co", "cobalt"],
        "V": ["v", "vanadium"],
        "Sc": ["sc", "scandium"],
        "Ga": ["ga", "gallium"],
        "Zn": ["zn", "zinc"],
        "Cu": ["cu", "copper"],
        # Isotopes
        "Sr87_Sr86": ["87sr/86sr", "⁸⁷sr/⁸⁶sr", "sr ratio", "87sr/86sr(i)"],
        "Nd143_Nd144": ["143nd/144nd", "εnd", "epsilon nd", "143nd/144nd(i)"],
        "Pb206_Pb204": ["206pb/204pb", "²⁰⁶pb/²⁰⁴pb"],
        "Pb207_Pb204": ["207pb/204pb", "²⁰⁷pb/²⁰⁴pb"],
        "Pb208_Pb204": ["208pb/204pb", "²⁰⁸pb/²⁰⁴pb"],
        "d18O": ["δ18o", "d18o", "delta 18", "o-18"],
        "d13C": ["δ13c", "d13c", "delta 13", "c-13"],
        "d34S": ["δ34s", "d34s", "delta 34", "s-34"],
    },
    valid_ranges={
        "SiO2": (30, 100),
        "TiO2": (0, 10),
        "Al2O3": (0, 40),
        "Fe2O3": (0, 30),
        "FeO": (0, 30),
        "MnO": (0, 5),
        "MgO": (0, 50),
        "CaO": (0, 40),
        "Na2O": (0, 15),
        "K2O": (0, 15),
        "P2O5": (0, 5),
        "Total": (95, 102),
    },
    calc_module=None,
    calc_function="cipw_hb",
)

# ── Geochronology ────────────────────────────────────────────────────────────
GEOCHRONOLOGY = DomainSpec(
    name="geochronology",
    table_headers={
        "Sample": ["sample", "sample id", "spot", "analysis", "grain"],
        "Mineral": ["mineral", "phase", "dated mineral"],
        "Method": ["method", "system", "isotope system", "dating method"],
        "Age_Ma": [
            "age",
            "age (ma)",
            "206pb/238u age",
            "207pb/206pb age",
            "40ar/39ar age",
            "re-os age",
            "age (ga)",
        ],
        "Error": ["±", "error", "1σ", "2σ", "uncertainty", "age error"],
        "Concordance": ["concordance", "conc. %", "discordance"],
        "Interpretation": ["interpretation", "geological meaning", "age type"],
    },
    valid_ranges={
        "Age_Ma": (0, 4600),  # 0 to age of Earth
        "Error": (0, 500),
        "Concordance": (50, 110),  # 50-110% concordant
    },
)

# ── Structural Geology ───────────────────────────────────────────────────────
STRUCTURAL = DomainSpec(
    name="structural",
    table_headers={
        "Station": ["station", "locality", "site", "outcrop", "location"],
        "Type": ["type", "structure", "feature", "measurement type"],
        "Strike": ["strike", "azimuth", "dip direction"],
        "Dip": ["dip", "inclination", "dip amount"],
        "Trend": ["trend", "plunge direction", "lineation trend"],
        "Plunge": ["plunge", "lineation plunge", "rake"],
        "Pitch": ["pitch", "rake", "slip rake"],
        "Sense": ["sense", "slip sense", "movement sense"],
        "Rx": ["rx", "strain x", "principal strain x"],
        "Ry": ["ry", "strain y", "principal strain y"],
        "Rz": ["rz", "strain z", "principal strain z"],
        "Rs": ["rs", "strain ratio", "flinn k"],
    },
    valid_ranges={
        "Strike": (0, 360),
        "Dip": (0, 90),
        "Trend": (0, 360),
        "Plunge": (0, 90),
        "Pitch": (0, 90),
        "Rx": (0.1, 10),
        "Ry": (0.1, 10),
        "Rz": (0.1, 10),
    },
    calc_module=None,
)

# ── Economic Geology ─────────────────────────────────────────────────────────
ECONOMIC = DomainSpec(
    name="economic",
    table_headers={
        "Sample": ["sample", "sample id", "drill hole", "interval"],
        "Depth_from": ["from", "depth from", "top"],
        "Depth_to": ["to", "depth to", "bottom"],
        "Cu_pct": ["cu", "cu %", "copper", "cu ppm", "copper grade"],
        "Au_ppm": ["au", "au g/t", "au ppm", "gold", "gold grade"],
        "Ag_ppm": ["ag", "ag g/t", "silver"],
        "Zn_pct": ["zn", "zn %", "zinc"],
        "Pb_pct": ["pb", "pb %", "lead"],
        "Ni_pct": ["ni", "ni %", "nickel"],
        "Co_pct": ["co", "co %", "cobalt"],
        "Mo_pct": ["mo", "mo %", "molybdenum"],
        "Fe_pct": ["fe", "fe %", "iron grade"],
        "Tonnage_Mt": ["tonnage", "resource", "mt", "million tonnes"],
        "Category": ["category", "resource category", "classification"],
    },
    valid_ranges={
        "Cu_pct": (0, 100),
        "Au_ppm": (0, 10000),
        "Zn_pct": (0, 100),
        "Tonnage_Mt": (0, 10000),
    },
)

# ── Geophysics ───────────────────────────────────────────────────────────────
GEOPHYSICS = DomainSpec(
    name="geophysics",
    table_headers={
        "Depth_km": ["depth", "depth (km)", "elevation"],
        "Vp": ["vp", "p-wave velocity", "compressional velocity"],
        "Vs": ["vs", "s-wave velocity", "shear velocity"],
        "Density": ["density", "ρ", "bulk density"],
        "Gravity_mGal": ["gravity", "anomaly", "bouguer", "free-air"],
        "Magnetic_nT": ["magnetic", "magnetic anomaly", "total field"],
        "Station": ["station", "point", "profile station"],
        "X": ["x", "easting", "longitude"],
        "Y": ["y", "northing", "latitude"],
    },
    valid_ranges={
        "Vp": (1.5, 9.0),
        "Vs": (0, 5.0),
        "Density": (1.0, 4.0),
        "Depth_km": (0, 700),
    },
)

# ── Sedimentology ────────────────────────────────────────────────────────────
SEDIMENTOLOGY = DomainSpec(
    name="sedimentology",
    table_headers={
        "Sample": ["sample", "sample id", "specimen"],
        "GrainSize_phi": ["grain size", "φ", "phi", "mean grain size"],
        "Sorting": ["sorting", "standard deviation"],
        "Skewness": ["skewness", "skew"],
        "Kurtosis": ["kurtosis"],
        "Porosity_pct": ["porosity", "φ", "porosity %"],
        "Permeability_mD": ["permeability", "k", "md", "millidarcy"],
        "Facies": ["facies", "lithofacies", "depositional facies"],
        "Quartz_pct": ["quartz", "q", "qtz"],
        "Feldspar_pct": ["feldspar", "f", "fsp"],
        "Lithics_pct": ["lithics", "l", "rock fragments"],
    },
    valid_ranges={
        "GrainSize_phi": (-4, 14),  # boulder to clay
        "Porosity_pct": (0, 50),
        "Permeability_mD": (0, 10000),
    },
)

# Registry — domain name → spec
DOMAINS: dict[str, DomainSpec] = {
    "geochemistry": GEOCHEMISTRY,
    "igneous": GEOCHEMISTRY,  # same data type
    "volcanic": GEOCHEMISTRY,
    "metamorphic": GEOCHEMISTRY,
    "mantle": GEOCHEMISTRY,
    "geochronology": GEOCHRONOLOGY,
    "structural": STRUCTURAL,
    "ore": ECONOMIC,
    "geophysical": GEOPHYSICS,
    "sedimentary": SEDIMENTOLOGY,
}


# =============================================================================
# PDF Table Parser (shared across all domains)
# =============================================================================


def parse_pdf_tables(pdf_path: Path | str) -> list[list[list[str]]]:
    """Extract all tables from a PDF using pdfplumber.

    Returns list of tables, each table is list of rows, each row is list of cells.
    """
    try:
        import pdfplumber
    except ImportError:
        log.warning("pdfplumber not installed — cannot parse PDF tables")
        return []

    tables: list[list[list[str]]] = []
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page in pdf.pages:
                page_tables = page.extract_tables()
                for tbl in page_tables:
                    if tbl and len(tbl) > 1:  # at least header + 1 data row
                        tables.append(tbl)
    except Exception as e:
        log.debug("PDF table parsing failed for %s: %s", pdf_path, e)

    return tables


# =============================================================================
# Domain Header Detector — identify which columns matter
# =============================================================================


def detect_table_domain(
    table: list[list[str]],
    spec: DomainSpec,
) -> dict[str, int] | None:
    """Check if a table matches this domain's expected headers.

    Returns column mapping {canonical_name: column_index} or None.
    """
    if not table or not table[0]:
        return None

    header_row = [str(cell or "").strip().lower() for cell in table[0]]
    matches: dict[str, int] = {}

    for canonical, patterns in spec.table_headers.items():
        for col_idx, header in enumerate(header_row):
            if col_idx in matches.values():
                continue
            # Normalize header for comparison
            h = header.lower().strip()
            # Check if any pattern matches
            for pattern in patterns:
                p = pattern.lower().strip()
                if p == h or p in h or h in p:
                    matches[canonical] = col_idx
                    break

    # Need at least 3 matching columns to classify as this domain
    if len(matches) >= 3:
        return matches
    return None


# =============================================================================
# Value Extractor — parse numerical values with unit normalization
# =============================================================================


def parse_numeric(value: str) -> float | None:
    """Parse a numeric value from a string, handling common geo formats."""
    if not value:
        return None
    v = str(value).strip()
    if v in ("-", "—", "n/a", "N/A", "n.d.", "nd", "b.d.", "b.d.l.", ""):
        return None
    # Remove common suffixes/prefixes
    v = re.sub(r"[<>≤≥~]", "", v)
    v = re.sub(r"\s*(wt%|wt\.%|ppm|ppb|‰|%)", "", v, flags=re.I)
    v = v.replace(",", "")  # thousand separators
    try:
        return float(v)
    except ValueError:
        return None


def validate_value(
    value: float,
    canonical: str,
    spec: DomainSpec,
) -> bool:
    """Check if a value is within geologically plausible range."""
    if canonical in spec.valid_ranges:
        lo, hi = spec.valid_ranges[canonical]
        return lo <= value <= hi
    return True  # no range defined — accept


# =============================================================================
# Main extraction function
# =============================================================================


@dataclass
class ExtractedDataset:
    """Structured quantitative data extracted from papers."""

    domain: str
    tables: list[dict] = field(default_factory=list)
    # Each table dict: {"headers": [...], "rows": [...], "source": "filename"}
    n_samples: int = 0
    n_variables: int = 0
    summary: dict[str, dict[str, float]] = field(default_factory=dict)
    # summary: {"SiO2": {"min": 65.3, "max": 72.1, "mean": 69.5, "n": 15, "std": 1.7}}

    def compute_summary(self) -> None:
        """Compute min/max/mean/std for each variable across all tables."""
        import statistics

        var_values: dict[str, list[float]] = {}
        for tbl in self.tables:
            for row in tbl["rows"]:
                for key, val in row.items():
                    if isinstance(val, (int, float)) and val is not None:
                        var_values.setdefault(key, []).append(float(val))

        self.summary = {}
        for var, vals in var_values.items():
            if len(vals) >= 1:
                self.summary[var] = {
                    "min": min(vals),
                    "max": max(vals),
                    "mean": statistics.mean(vals) if vals else 0,
                    "std": statistics.stdev(vals) if len(vals) > 1 else 0,
                    "n": len(vals),
                }
        self.n_samples = max((len(v) for v in var_values.values()), default=0)
        self.n_variables = len(var_values)


def extract_domain_data(
    pdf_path: Path | str,
    domain: str = "geochemistry",
) -> ExtractedDataset | None:
    """Extract domain-specific quantitative data from a research paper PDF.

    Parameters
    ----------
    pdf_path : Path to the PDF file
    domain : Geological domain name (geochemistry, geochronology, etc.)

    Returns
    -------
    ExtractedDataset with structured data, or None if no data found.
    """
    spec = DOMAINS.get(domain)
    if not spec:
        log.warning("Unknown domain: %s — known: %s", domain, list(DOMAINS))
        return None

    # Parse tables from PDF
    raw_tables = parse_pdf_tables(pdf_path)
    if not raw_tables:
        log.debug("No tables found in %s", pdf_path)
        return None

    log.info("Found %d tables in %s — scanning for %s data", len(raw_tables), pdf_path, domain)

    dataset = ExtractedDataset(domain=domain)

    for table in raw_tables:
        # Check if this table matches our domain
        col_map = detect_table_domain(table, spec)
        if not col_map:
            continue

        # Extract data rows
        parsed_rows: list[dict[str, float | str]] = []
        n_valid = 0
        for row in table[1:]:  # skip header
            parsed: dict[str, float | str] = {}
            for canonical, col_idx in col_map.items():
                if col_idx >= len(row):
                    continue
                raw_val = row[col_idx]
                num_val = parse_numeric(raw_val)
                if num_val is not None:
                    if validate_value(num_val, canonical, spec):
                        parsed[canonical] = num_val
                        n_valid += 1
                    else:
                        log.debug(
                            "Rejected %s=%s (out of range for %s)",
                            canonical,
                            num_val,
                            domain,
                        )
                else:
                    # Keep as string (sample ID, facies name, etc.)
                    parsed[canonical] = str(raw_val or "").strip()
            if parsed:
                parsed_rows.append(parsed)

        if parsed_rows and n_valid > 0:
            dataset.tables.append(
                {
                    "headers": list(col_map.keys()),
                    "rows": parsed_rows,
                    "source": str(pdf_path),
                }
            )
            log.info(
                "Extracted %s table: %d rows × %d columns from %s",
                domain,
                len(parsed_rows),
                len(col_map),
                Path(pdf_path).name,
            )

    if not dataset.tables:
        return None

    dataset.compute_summary()
    return dataset


# =============================================================================
# GeoKit Integration — feed extracted data to calculation engines
# =============================================================================


def run_domain_calculations(
    dataset: ExtractedDataset,
) -> dict[str, Any]:
    """Run GeoKit domain-specific calculations on extracted data.

    Uses GeoKit's EXISTING modules (norms, geotherm, structural, diagrams)
    to compute derived values from the extracted raw data.
    """
    results: dict[str, Any] = {}

    if dataset.domain in ("geochemistry", "igneous", "volcanic", "metamorphic", "mantle"):
        # ── CIPW Normative Mineralogy ──
        try:
            import sys

            sys.path.insert(0, "src")
            import pandas as pd


            # Build DataFrame from extracted major elements
            for tbl in dataset.tables:
                oxide_cols = [
                    "SiO2",
                    "TiO2",
                    "Al2O3",
                    "Fe2O3",
                    "FeO",
                    "MnO",
                    "MgO",
                    "CaO",
                    "Na2O",
                    "K2O",
                    "P2O5",
                ]
                data = {}
                for col in oxide_cols:
                    values = [
                        r.get(col) for r in tbl["rows"] if isinstance(r.get(col), (int, float))
                    ]
                    if values:
                        data[col] = values

                if len(data) >= 8:  # need at least 8 oxides
                    df = pd.DataFrame(data)
                    try:
                        norm = cipw_hb(df)
                        results["cipw_norm"] = {
                            "n_samples": len(norm),
                            "quartz_mean": float(norm.get("Q", pd.Series()).mean())
                            if "Q" in norm
                            else None,
                            "orthoclase_mean": float(norm.get("Or", pd.Series()).mean())
                            if "Or" in norm
                            else None,
                            "plagioclase_mean": float(norm.get("Pl", pd.Series()).mean())
                            if "Pl" in norm
                            else None,
                        }
                        log.info("CIPW norm computed for %d samples", len(norm))
                    except Exception as e:
                        log.debug("CIPW norm failed: %s", e)

            # ── Classification (TAS, etc.) ──
            if "SiO2" in dataset.summary and "Na2O" in dataset.summary and "K2O" in dataset.summary:
                sio2 = dataset.summary["SiO2"]
                alkali = dataset.summary.get("Na2O", {}).get("mean", 0) + dataset.summary.get(
                    "K2O", {}
                ).get("mean", 0)
                results["tas_classification"] = _classify_tas(sio2["mean"], alkali)

            # ── A/CNK (Alumina Saturation) ──
            if all(k in dataset.summary for k in ("Al2O3", "CaO", "Na2O", "K2O")):
                for tbl in dataset.tables:
                    for row in tbl["rows"]:
                        a = row.get("Al2O3", 0)
                        c = row.get("CaO", 0)
                        n = row.get("Na2O", 0)
                        k = row.get("K2O", 0)
                        if all(isinstance(x, (int, float)) and x > 0 for x in (a, c, n, k)):
                            acnk = a / (c + n + k)
                            results.setdefault("acnk", []).append(acnk)
                if "acnk" in results:
                    results["acnk_mean"] = sum(results["acnk"]) / len(results["acnk"])
                    results["alumina_saturation"] = (
                        "peraluminous"
                        if results["acnk_mean"] > 1.0
                        else "metaluminous"
                        if results["acnk_mean"] > 0.9
                        else "subaluminous"
                    )

        except ImportError:
            log.debug("GeoKit norms module unavailable")

    elif dataset.domain == "structural":
        # ── Stereonet + Stress Inversion ──
        try:

            orientations = []
            for tbl in dataset.tables:
                for row in tbl["rows"]:
                    strike = row.get("Strike")
                    dip = row.get("Dip")
                    if isinstance(strike, (int, float)) and isinstance(dip, (int, float)):
                        orientations.append((strike, dip))
            if orientations:
                results["n_orientations"] = len(orientations)
                results["mean_strike"] = sum(s for s, _ in orientations) / len(orientations)
                results["mean_dip"] = sum(d for _, d in orientations) / len(orientations)
                log.info("Structural data: %d orientations", len(orientations))
        except ImportError:
            log.debug("GeoKit structural module unavailable")

    elif dataset.domain == "geochronology":
        # ── Age Statistics ──
        ages: list[float] = []
        for tbl in dataset.tables:
            for row in tbl["rows"]:
                age = row.get("Age_Ma")
                if isinstance(age, (int, float)) and age > 0:
                    ages.append(age)
        if ages:
            import statistics

            results["ages"] = {
                "n": len(ages),
                "min_ma": min(ages),
                "max_ma": max(ages),
                "weighted_mean_ma": statistics.mean(ages),
                "std_ma": statistics.stdev(ages) if len(ages) > 1 else 0,
            }
            # Geological era classification
            mean_age = statistics.mean(ages)
            if mean_age > 541:
                results["era"] = "Precambrian" if mean_age > 541 else "Phanerozoic"
                if mean_age > 2500:
                    results["eon"] = "Archean"
                elif mean_age > 541:
                    results["eon"] = "Proterozoic"

    elif dataset.domain == "ore":
        # ── Grade-Tonnage Statistics ──
        grades: dict[str, list[float]] = {}
        for tbl in dataset.tables:
            for row in tbl["rows"]:
                for metal in ("Cu_pct", "Au_ppm", "Zn_pct", "Pb_pct", "Ni_pct"):
                    val = row.get(metal)
                    if isinstance(val, (int, float)) and val > 0:
                        grades.setdefault(metal, []).append(val)
        for metal, vals in grades.items():
            import statistics

            results[f"{metal}_stats"] = {
                "n": len(vals),
                "min": min(vals),
                "max": max(vals),
                "mean": statistics.mean(vals),
                "grade_category": _classify_grade(metal, statistics.mean(vals)),
            }

    return results


def _classify_tas(sio2: float, alkali: float) -> str:
    """Classify rock using TAS diagram boundaries."""
    if sio2 < 45:
        if alkali < 3:
            return "basanite"
        return "foidite / tephrite"
    elif sio2 < 52:
        return "basalt"
    elif sio2 < 57:
        return "basaltic andesite"
    elif sio2 < 63:
        return "andesite"
    elif sio2 < 69:
        if alkali > 8:
            return "phonolite"
        if alkali > 5 and sio2 < 65:
            return "trachyandesite"
        return "dacite"
    elif sio2 < 77:
        if alkali > 8:
            return "phonolite"
        if alkali > 5:
            return "trachyte"
        return "rhyolite"
    return "high-silica rhyolite"


def _classify_grade(metal: str, mean_grade: float) -> str:
    """Classify ore grade as low/moderate/high."""
    thresholds = {
        "Cu_pct": (0.3, 1.0),
        "Au_ppm": (1.0, 5.0),
        "Zn_pct": (2.5, 10.0),
        "Pb_pct": (1.0, 5.0),
        "Ni_pct": (0.5, 1.5),
    }
    if metal in thresholds:
        low, high = thresholds[metal]
        if mean_grade < low:
            return "low grade"
        if mean_grade < high:
            return "moderate grade"
        return "high grade"
    return "unknown"
