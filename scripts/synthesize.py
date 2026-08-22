#!/usr/bin/env python3
"""Phase 5: Narrative synthesis — orchestrator + CLI.

Produces research_brief.md from extracted.json + correlation.json + meta.json.
NON-LLM DEFAULT: template stitching (reproducible, deterministic).
--use-llm OPT-IN: Ollama rewrites templates into smoother prose.

Five research types drive output structure:
  verification:     yes/no question → verdict + stance evidence summary
  survey:           broad topic → chronological narrative paragraphs
  subtopic:         specific focus → focused narrative + measurement summary
  comparative:      compare X vs Y → side-by-side grouping by approach
  data_compilation: compile values → data table + statistical summary

Usage:
  uv run python scripts/synthesize.py \\
      --query "latest research on basalt geochemistry" \\
      --extracted research_outputs/extracted.json \\
      --verified  research_outputs/verified.json \\
      --correlation research_outputs/correlation.json \\
      -o research_outputs/research_brief.md
"""

from __future__ import annotations

# --- Skill venv bootstrap (shared: see _bootstrap.py) ---
if __name__ == "__main__":
    import _bootstrap

    _bootstrap.ensure_env()
# --- End bootstrap ---


import json
import logging
import re
import sys
import time
from pathlib import Path

log = logging.getLogger("scientific_research.synthesize")

_SCRIPT_DIR = Path(__file__).parent.resolve()
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from _classifiers import detect_research_type
from _narrative import (
    build_chronological_narrative,
    build_comparison_narrative,
    build_compilation_narrative,
    build_verification_narrative,
    format_citation_list,
)

# =============================================================================
# Data loading helpers
# =============================================================================


# =============================================================================
# LLM prose smoothing (opt-in via --use-llm)
# =============================================================================


def _detect_best_model() -> str:
    """Detect smallest available Ollama model for smoothing task."""
    try:
        import urllib.request

        req = urllib.request.Request(
            "http://127.0.0.1:11434/api/tags",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())

        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size = m.get("size", 0)
            if any(
                skip in name.lower()
                for skip in (
                    "vision",
                    "embed",
                    "ocr",
                    "vl",
                    "bge",
                    "nomic",
                    "mxbai",
                    "snowflake",
                    "minilm",
                    "all-minilm",
                    "e5",
                    "gte",
                )
            ):
                continue
            models.append((name, size))

        if not models:
            return "qwen3.5:0.8b"

        models.sort(key=lambda x: x[1])
        return models[0][0]
    except Exception:
        return "qwen3.5:0.8b"

def _valid_smooth(original: str, output: str) -> bool:
    """Gate on LLM-smoothed text before it may enter a brief.

    Small local models answer polishing prompts with refusals, apologies,
    and meta-commentary ("I notice this paragraph contains...", "Please
    provide the full paragraph") — text that has shipped verbatim into
    briefs. A smoothed block is accepted ONLY when it is a real polish of
    the input: no refusal markers, all [N] citations and all numeric tokens
    preserved, sane length ratio.
    """
    if not output or len(output) < 40:
        return False
    low = output.lower()
    refusal_markers = (
        "i can't", "i cannot", "i'm sorry", "please provide", "please paste",
        "please share", "i notice", "notes on my edits", "what i'd recommend",
        "as an ai", "i'll be happy", "before polishing", "cannot responsibly",
        "cannot polish", "would you like me", "i'd recommend", "my edits",
        "redacted", "placeholder elements", "here's what i",
    )
    if any(m in low for m in refusal_markers):
        return False
    # Citations [N] must survive
    orig_refs = set(re.findall(r"\[\d+(?:,\s*\d+)*\]", original))
    if any(r not in output for r in orig_refs):
        return False
    # Numeric data must survive (token-level; formatting may vary slightly)
    orig_nums = re.findall(r"\d+\.?\d*", original)
    out_num_set = set(re.findall(r"\d+\.?\d*", output))
    missing = [n for n in orig_nums if n not in out_num_set]
    if missing and len(missing) > max(1, len(orig_nums) // 10):
        return False
    ratio = len(output) / max(len(original), 1)
    return 0.4 <= ratio <= 2.5



def smooth_theme_paragraphs(
    narrative: str,
    topic: str,
    model: str | None = None,
) -> str:
    """Smooth each theme paragraph individually via LLM.

    Preserves markdown headers (### Theme), only smooths prose paragraphs.
    Returns smoothed narrative or original on failure.
    """
    try:
        from _llm_extract import is_available
    except ImportError:
        return narrative

    if not is_available():
        return narrative

    import hashlib
    import json as _json
    import urllib.request

    if model is None:
        model = _detect_best_model()

    nl = chr(10)  # newline character
    lines = narrative.split(nl)
    smoothed_lines = []
    current_section = []

    for line in lines:
        if line.startswith(("###", "## ")):
            if current_section:
                section_text = nl.join(current_section).strip()
                if len(section_text) > 200 and not section_text.startswith("|"):
                    cache_key = hashlib.sha256(
                        f"theme_smooth|{topic}|{section_text[:500]}".encode()
                    ).hexdigest()
                    cache_dir = (
                        Path.home() / ".cache" / "scientific_research" / "theme_smooth"
                    )
                    cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_file = cache_dir / f"{cache_key}.txt"

                    if cache_file.exists():
                        cached = cache_file.read_text(encoding="utf-8")
                        # Validate cached blocks too — poisoned cache entries
                        # (refusals cached by pre-gate code) must not replay.
                        if cached and _valid_smooth(section_text, cached):
                            smoothed_lines.append(cached)
                            current_section = []
                            smoothed_lines.append(line)
                            continue

                    prompt = (
                        "Polish this geological research paragraph for publication quality. "
                        "CRITICAL RULES:\n"
                        "1. Keep ALL [N] citation numbers exactly as written.\n"
                        "2. PRESERVE all geological terminology: mineral names (garnet, "
                        "biotite, clinopyroxene), P-T values (550C, 6 kbar), "
                        "geochemical notation (SiO2, d18O, eNd, KD), and method names.\n"
                        "3. PRESERVE all quantitative data (temperatures, pressures, "
                        "ages, compositions, partition coefficients).\n"
                        "4. Do NOT add new facts, citations, or geological claims.\n"
                        "5. Improve sentence flow and reduce wordiness ONLY.\n"
                        "6. Do NOT replace technical terms with simpler alternatives.\n"
                        "7. Respond with ONLY the polished paragraph. No preamble, "
                        "no explanations, no offers to help, no questions.\n"
                        "8. NEVER refuse or comment on the text. If a part is "
                        "unclear, polish around it and keep it intact.\n\n"
                        f"{section_text}\n\nPolished:"  # full section (num_ctx sized)
                    )

                    try:
                        payload = _json.dumps(
                            {
                                "model": model,
                                "messages": [
                                    {"role": "user", "content": "/no_think\n" + prompt}
                                ],
                                "stream": False,
                                "think": False,
                                "options": {
                                    "temperature": 0.0,
                                    "top_k": 1,
                                    "num_ctx": 4096,
                                    "num_predict": 2048,
                                    "repeat_penalty": 1.1,
                                },
                            }
                        ).encode("utf-8")

                        req = urllib.request.Request(
                            "http://127.0.0.1:11434/api/chat",
                            data=payload,
                            headers={"Content-Type": "application/json"},
                        )
                        with urllib.request.urlopen(req, timeout=60) as resp:
                            result = _json.loads(resp.read())
                            msg = result.get("message", {})
                            smoothed = (msg.get("content") or "").strip()

                        if smoothed.startswith(("/no_think", "No_think")):
                            smoothed = smoothed.split(nl, 1)[-1].strip()
                        if smoothed and _valid_smooth(section_text, smoothed):
                            cache_file.write_text(smoothed, encoding="utf-8")
                            smoothed_lines.append(smoothed)
                        else:
                            log.info(
                                "Smoothing rejected (refusal/data-loss gate) "
                                "for section %.60r — keeping original",
                                section_text,
                            )
                            smoothed_lines.append(section_text)
                    except Exception as e:
                        log.debug("Theme smoothing failed: %s", e)
                        smoothed_lines.append(section_text)
                else:
                    smoothed_lines.append(section_text)
                current_section = []
            smoothed_lines.append(line)
        else:
            current_section.append(line)

    if current_section:
        smoothed_lines.append(nl.join(current_section))

    result = nl.join(smoothed_lines)
    log.info("Theme-paragraph smoothing complete")
    return result


# =============================================================================
# Main orchestrator
# =============================================================================


def _build_structural_gaps(
    papers: list[dict],
    correlation: dict,
    found_themes: list[str] | None = None,
) -> list[str]:
    """R5: gap signals from LIVE corpus + correlation structure."""
    gap_points: list[str] = []
    years = [p.get("year") for p in papers if p.get("year")]
    if years and len(years) >= 5:
        recent_cutoff = max(years) - 5
        n_recent = sum(1 for y in years if y >= recent_cutoff)
        if n_recent < len(years) * 0.3:
            gap_points.append(
                f"temporal: only {n_recent}/{len(years)} studies newer than "
                f"{recent_cutoff} — recent-method recalibration may be missing"
            )
    n_quant = sum(
        1
        for p in papers
        if p.get("measurements")
        or (p.get("effect_sizes") or {}).get("single_measurements")
    )
    if len(papers) and n_quant < len(papers) * 0.5:
        gap_points.append(
            f"quantitative coverage: only {n_quant}/{len(papers)} studies "
            f"yield extractable numeric estimates (abstract-only ceiling; "
            f"full-text extraction would widen this)"
        )
    theme_counts: dict[str, int] = {}
    for p in papers:
        pico = p.get("pico") or {}
        theme = pico.get("study_type") or pico.get("discipline") or "the corpus"
        theme_counts[theme] = theme_counts.get(theme, 0) + 1
    if len(theme_counts) >= 2:
        biggest = max(theme_counts.values())
        if biggest >= len(papers) * 0.6:
            gap_points.append(
                f"method concentration: one approach covers {biggest}/"
                f"{len(papers)} studies — cross-method validation is thin"
            )
    if correlation.get("n_papers") and not (
        correlation.get("bibliographic_coupling_edges") or []
    ):
        gap_points.append(
            "no bibliographic coupling detected — studies rarely share "
            "reference sets; the field may be fragmented"
        )
    return gap_points


def _render_pool_section(meta: dict) -> str:
    """R2: pooled-estimates markdown from meta.json unit_pools ('' if none)."""
    unit_pools = meta.get("unit_pools") or []
    if not unit_pools:
        return ""
    pool_lines = ["## Pooled estimates (random-effects, REML + HKSJ)", ""]
    for up in unit_pools:
        pr = up.get("pooled_random") or {}
        eff = pr.get("effect")
        ci = pr.get("ci_lower"), pr.get("ci_upper")
        i2 = (pr.get("heterogeneity") or {}).get("i_squared")
        pi = pr.get("prediction_interval")
        if eff is None or ci[0] is None or i2 is None:
            continue
        pi_txt = f"; 95% PI {pi[0]:.1f}–{pi[1]:.1f}" if pi else ""
        het_flag = (
            " **(high heterogeneity — treat as range, not consensus)**"
            if isinstance(i2, (int, float)) and i2 > 75
            else ""
        )
        pool_lines.append(
            f"- **{up['group']}**: {eff:.1f} "
            f"(95% CI {ci[0]:.1f}–{ci[1]:.1f}; k={up['k']}; "
            f"I²={i2:.0f}%{pi_txt}){het_flag}"
        )
    pool_lines += [
        "",
        ("_Pools are unit-commensurable groups (unit gate + phenomenon "
        "screen applied; see meta.json unit_pools for per-study rows)._"),
        "",
    ]
    return "\n".join(pool_lines)


def _build_executive_summary(
    cited: list[dict], narrative: str, topic: str, meta: dict | None = None
) -> str:
    """Build executive summary — thesis statement + key consensus + gaps.

    Extracts the most important quantitative findings from the narrative
    and synthesizes a 3-4 sentence overview at the top of the brief.
    """
    if not cited:
        return ""

    n = len(cited)
    years = [p.get("year") for p in cited if p.get("year")]
    year_range = ""
    if years:
        yr_min, yr_max = min(years), max(years)
        year_range = f"{yr_min}-{yr_max}" if yr_min != yr_max else str(yr_min)

    # Extract all VALIDATED numerical values
    all_temps = []
    all_pressures = []
    for p in cited:
        finding = p.get("key_finding") or ""
        for m in p.get("measurements", []):
            try:
                from _narrative import _normalize_measurement

                val, unit = _normalize_measurement(
                    m.get("measurement", ""), m.get("value"), m.get("unit", "")
                )
                if val is not None and unit == "C":
                    all_temps.append(val)
                elif val is not None and unit == "kbar":
                    all_pressures.append(val)
            except ImportError:
                pass
        # Also scan finding text (validated)
        try:
            from _narrative import _extract_numbers_from_text

            for val, unit in _extract_numbers_from_text(finding):
                if unit == "C":
                    all_temps.append(val)
                elif unit == "kbar":
                    all_pressures.append(val)
        except ImportError:
            pass

    # Count method themes
    try:
        from _narrative import _group_by_theme

        themes = _group_by_theme(cited)
        theme_labels = [t["label"] for t in themes[:4]]
    except ImportError:
        theme_labels = []

    parts = ["## Executive summary", ""]
    lines = []

    # Sentence 1: Scope
    scope = f"This review synthesizes {n} studies"
    if year_range:
        scope += f" ({year_range})"
    scope += f" on **{topic}**"
    if theme_labels:
        scope += f", spanning {len(themes)} methodological approaches"
    scope += "."
    lines.append(scope)

    # Sentence 2: Key quantitative findings
    import statistics as _stats

    quant_parts = []
    if len(all_temps) >= 3:
        t_median = _stats.median(all_temps)
        t_range = f"{min(all_temps):.0f}-{max(all_temps):.0f}"
        quant_parts.append(
            f"temperatures of {t_range}C (median {t_median:.0f}C, n={len(all_temps)})"
        )
    if len(all_pressures) >= 3:
        p_median = _stats.median(all_pressures)
        p_range = f"{min(all_pressures):.1f}-{max(all_pressures):.1f}"
        quant_parts.append(
            f"pressures of {p_range} kbar (median {p_median:.1f} kbar, n={len(all_pressures)})"
        )
    if quant_parts:
        lines.append(
            "Key quantitative findings include " + "; ".join(quant_parts) + "."
        )

    # Sentence 3: Main themes
    if theme_labels:
        lines.append(
            f"The dominant methodological themes are {', '.join(theme_labels[:3])}."
        )

    # Sentence 4: Consensus or gap
    if len(all_temps) >= 5:
        t_std = _stats.stdev(all_temps) if len(all_temps) >= 2 else 0
        t_median = _stats.median(all_temps)
        cv = t_std / abs(t_median) if t_median else 0
        if cv < 0.15:
            lines.append("Reported values show good consistency across studies.")
        elif cv > 0.4:
            lines.append(
                "Reported values show considerable variability across studies, "
                "reflecting differences in rock type, metamorphic grade, and methodological approach. "
                "Median values are more representative than means for this heterogeneous corpus."
            )

    # Sentence 5+: pooled estimates (when meta-analysis ran)
    if meta:
        pools = meta.get("unit_pools") or []
        top = [
            p for p in pools if (p.get("pooled_random") or {}).get("effect") is not None
        ][:3]
        if top:
            bits = []
            for p in top:
                pr = p["pooled_random"]
                i2 = (pr.get("heterogeneity") or {}).get("i_squared")
                bit = f"{p['group']} {pr['effect']:.1f} (95% CI {pr['ci_lower']:.1f}–{pr['ci_upper']:.1f}, k={p['k']}"
                bit += f", I²={i2:.0f}%)" if isinstance(i2, (int, float)) else ")"
                bits.append(bit)
            lines.append(
                "Random-effects pooling (REML, HKSJ) across commensurable unit "
                "groups gives " + "; ".join(bits) + "."
            )
            if any(
                isinstance(
                    (p.get("pooled_random") or {})
                    .get("heterogeneity", {})
                    .get("i_squared"),
                    (int, float),
                )
                and p["pooled_random"]["heterogeneity"]["i_squared"] > 75
                for p in top
            ):
                lines.append(
                    "High between-study heterogeneity (I² > 75%) means pooled "
                    "values describe a RANGE of storage conditions rather than "
                    "a single consensus estimate; decade-level decomposition is "
                    "reported per pool."
                )

    # Contradiction signal (cheap regex claims pass over cited abstracts)
    try:
        from _claims_engine import detect_contradictions, extract_claims

        claims = [c for p in cited[:60] for c in extract_claims(p)]
        cons = detect_contradictions(claims)
        if cons:
            c0 = cons[0]
            lines.append(
                f"Notable divergence: {c0.quantity} ({c0.unit}) estimates span "
                f"a {c0.ratio}x range across {len(c0.values)} reports — a "
                f"calibration/method disagreement candidates section details this."
            )
    except Exception as e:
        log.warning("Contradiction signal skipped: %s", e)

    parts.extend(lines)
    parts.append("")
    return "\n".join(parts)


def _build_method_comparison_table(cited: list[dict]) -> str:
    """Build enhanced method comparison table with P-T ranges.

    Groups papers by detected method theme, outputs:
    Method | n studies | Key contribution | P range | T range | Limitations
    """
    if not cited:
        return ""
    try:
        from _narrative import (
            _clip,
            _extract_limitations,
            _extract_numbers_from_text,
            _group_by_theme,
        )
    except ImportError:
        return ""

    themes = _group_by_theme(cited)
    if len(themes) < 2:
        return ""

    lines = [
        "## Methodological landscape",
        "",
        "| Approach | n | Key contribution | Reported range | Limitations |",
        "|---|---|---|---|---|",
    ]

    for theme in themes[:8]:
        label = theme["label"]
        papers = theme["papers"]
        n = len(papers)
        if n == 0:
            continue

        # Extract one key finding as representative
        top_finding = ""
        for p in papers[:5]:
            f = (p.get("key_finding") or "").strip()
            if f and len(f) > 40:
                top_finding = _clip(f, 300)
                break
        if not top_finding:
            # No finding — use first 100 chars of title as summary
            for p in papers[:3]:
                t = (p.get("title") or "").strip()
                if t:
                    top_finding = t[:200] + ("..." if len(t) > 200 else "")
                    break

        # Extract VALIDATED P-T ranges
        temps = []
        pressures = []
        for p in papers:
            # Check both 'measurements' and 'effect_sizes.single_measurements'
            all_measurements = list(p.get("measurements", []))
            es = p.get("effect_sizes") or {}
            if isinstance(es, dict):
                all_measurements.extend(es.get("single_measurements", []))
            for m in all_measurements:
                try:
                    from _narrative import _normalize_measurement

                    val, unit = _normalize_measurement(
                        m.get("measurement", "") or m.get("name", ""),
                        m.get("value"),
                        m.get("unit", ""),
                    )
                    if val is not None and unit == "C":
                        temps.append(val)
                    elif val is not None and unit == "kbar":
                        pressures.append(val)
                except (ImportError, TypeError, ValueError):
                    pass
            for val, unit in _extract_numbers_from_text(p.get("key_finding") or ""):
                if unit == "C":
                    temps.append(val)
                elif unit == "kbar":
                    pressures.append(val)

        range_parts = []
        if temps:
            tmin, tmax = min(temps), max(temps)
            if tmin == tmax:
                range_parts.append(f"T: {tmin:.0f}C (n={len(temps)})")
            elif len(temps) == 1:
                range_parts.append(f"T: {tmin:.0f}C")
            else:
                range_parts.append(f"T: {tmin:.0f}-{tmax:.0f}C (n={len(temps)})")
        if pressures:
            pmin, pmax = min(pressures), max(pressures)
            if pmin == pmax:
                range_parts.append(f"P: {pmin:.1f} kbar (n={len(pressures)})")
            elif len(pressures) == 1:
                range_parts.append(f"P: {pmin:.1f} kbar")
            else:
                range_parts.append(
                    f"P: {pmin:.1f}-{pmax:.1f} kbar (n={len(pressures)})"
                )
        range_str = "; ".join(range_parts) if range_parts else "-"

        # Extract limitations
        lims = _extract_limitations(papers[:8])
        lim_str = "; ".join(lim[:120] for lim in lims[:4]) if lims else "-"

        lines.append(f"| {label} | {n} | {top_finding} | {range_str} | {lim_str} |")

    lines.append("")
    return "\n".join(lines)


def _build_publication_trends(cited: list[dict]) -> str:
    """Build publication trend summary from cited papers."""
    if not cited:
        return ""
    from collections import Counter

    years = [p.get("year") for p in cited if p.get("year")]
    if len(years) < 5:
        return ""

    year_counts = Counter(years)
    min_year = min(years)
    max_year = max(years)

    lines = [
        "## Publication trends",
        "",
        f"- **Date range**: {min_year}–{max_year} ({len(years)} dated papers)",
        f"- **Peak year**: {year_counts.most_common(1)[0][0]} ({year_counts.most_common(1)[0][1]} papers)",
    ]

    # Decade breakdown
    decades = Counter()
    for y in years:
        decades[(y // 10) * 10] += 1
    decade_str = ", ".join(f"{d}s: {c}" for d, c in sorted(decades.items()))
    lines.append(f"- **By decade**: {decade_str}")

    # Recent trend
    recent = sum(1 for y in years if y >= max_year - 5)
    lines.append(
        f"- **Last 5 years** ({max_year - 5}–{max_year}): {recent} papers ({recent * 100 // len(years)}%)"
    )
    lines.append("")

    return "\n".join(lines)


def synthesize(
    query: str,
    extracted_path: Path | None,
    verified_path: Path | None,
    correlation_path: Path | None = None,
    meta_path: Path | None = None,
    output_path: Path | None = None,
    use_llm: bool = False,
) -> str:
    """Main synthesis orchestrator.

    1. Detect research type from query
    2. Load data from pipeline outputs
    3. Build narrative based on type
    4. Optionally smooth with LLM
    5. Write to output_path and return narrative text
    """
    from _artifact import (  # deduped helpers (Phase 1)
        ArtifactShapeError,
        _extract_papers,
        _load_json,
        load_extractions,
        load_verified,
    )

    try:
        extracted = load_extractions(extracted_path)
        verified = load_verified(verified_path)
    except ArtifactShapeError as e:
        raise SystemExit(f"FATAL: {e}") from e
    correlation = _load_json(correlation_path)  # optional
    meta = _load_json(meta_path)  # optional

    papers = _extract_papers(extracted, verified)

    if not papers:
        log.error("No papers found in extracted/verified data")
        msg = f"# Research Brief: {query}\n\nNo papers available for synthesis.\n"
        if output_path:
            output_path.write_text(msg, encoding="utf-8")
        return msg

    research_type = detect_research_type(query)
    log.info("Research type: %s (%d papers)", research_type, len(papers))

    # Build narrative based on type
    if research_type == "verification":
        narrative, cited = build_verification_narrative(papers, correlation, query)
    elif research_type == "comparative":
        narrative, cited = build_comparison_narrative(papers, query)
    elif research_type == "data_compilation":
        narrative, cited = build_compilation_narrative(papers, query)
    else:
        narrative, cited = build_chronological_narrative(
            papers, query, research_type, correlation
        )

    # LLM smoothing — per-theme paragraphs (preserves structure)
    if use_llm and cited:
        narrative = smooth_theme_paragraphs(narrative, query)

    # Assemble full brief
    parts: list[str] = []

    # Claim-level evidence synthesis
    try:
        from _claims import (
            extract_claims,
            render_claim_summary,
            render_contradiction_section,
        )

        claim_items = []
        for c in cited:
            ref = c.get("ref", 0)
            finding = c.get("finding", "") or c.get("summary", "")
            measurements = c.get("measurements") or {}
            if finding:
                claim_items.append(
                    {"ref": ref, "finding": finding, "measurements": measurements}
                )

        if claim_items:
            claims = extract_claims(claim_items)
            if claims:
                claim_text = render_claim_summary(claims)
                contradiction_text = render_contradiction_section(claims)
                if claim_text:
                    parts.append(claim_text)
                if contradiction_text:
                    parts.append(contradiction_text)
    except Exception as e:
        log.debug("Claim synthesis skipped: %s", e)
    parts.append(f"# Research Brief: {query}")
    parts.append("")
    parts.append(f"**Research type**: {research_type}")
    parts.append(f"**Corpus**: {len(papers)} papers")
    if cited:
        years = [p.get("year") for p in cited if p.get("year")]
        if years:
            parts.append(f"**Date range**: {min(years)}–{max(years)}")
    parts.append(f"**Generated**: {time.strftime('%Y-%m-%d %H:%M')}")
    parts.append(
        f"**Method**: {'LLM-smoothed' if use_llm else 'template-based (non-LLM)'}"
    )
    parts.append("")
    parts.append("---")
    parts.append("")

    # Executive summary
    if cited:
        exec_summary = _build_executive_summary(cited, narrative, query, meta=meta)
        if exec_summary:
            parts.append(exec_summary)

    pool_section = _render_pool_section(meta or {})
    if pool_section:
        parts.append(pool_section)

    # Method comparison table
    if cited:
        method_table = _build_method_comparison_table(cited)
        if method_table:
            parts.append(method_table)

    # Publication trends
    if cited:
        trends = _build_publication_trends(cited)
        if trends:
            parts.append(trends)

    # Geological implications — P-T interpretation (non-LLM, textbook logic)
    try:
        from _geo_enrich import interpret_pt_data

        all_temps: list[float] = []
        all_pressures: list[float] = []
        disc_counts: dict[str, int] = {}
        for p in cited or papers:
            pico = p.get("pico") or {}
            d = (pico.get("discipline") or "").lower()
            if d:
                disc_counts[d] = disc_counts.get(d, 0) + 1
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
                    all_temps.append(val)
                elif "press" in mname and 0 < val < 100:
                    all_pressures.append(val)
        dominant_disc = max(disc_counts, key=disc_counts.get) if disc_counts else ""
        pt_interp = interpret_pt_data(all_temps, all_pressures, dominant_disc)
        if pt_interp:
            parts.append("## Geological implications")
            parts.append("")
            parts.append(pt_interp)
            parts.append("")
    except ImportError:
        pass

    parts.append(narrative)
    parts.append("")

    # Convergence and controversies — statistical analysis
    try:
        from _geo_enrich import build_convergence_text

        # Group papers by detected theme for convergence analysis
        theme_groups: dict[str, list[dict]] = {}
        for p in cited or papers:
            pico = p.get("pico") or {}
            theme = pico.get("study_type") or pico.get("discipline") or "the corpus"
            theme_groups.setdefault(theme, []).append(p)
        conv_text = build_convergence_text(theme_groups)
        if conv_text and conv_text.strip():
            # Heading only when there is substance — a bare heading reads
            # as a synthesis failure to the reader.
            parts.append("## Convergence and controversies")
            parts.append("")
            parts.append(conv_text)
            parts.append("")
        else:
            log.info("Convergence section skipped: no statistical signal")
        # Claim-level value-divergence signals (Phase 3 engine, 2026-08-15):
        # clusters same-quantity numeric claims across papers and reports
        # >=5x spreads — the calibration-disagreement candidates a reviewer
        # expects in this section. Advisory; sentences quoted for traceability.
        try:
            from _claims_engine import (
                contradiction_report,
                detect_contradictions,
                extract_claims,
            )

            claims = [c for p in (cited or papers)[:80] for c in extract_claims(p)]
            cons = detect_contradictions(claims)
            if cons:
                parts.append("### Quantitative divergences (claim-level)")
                parts.append("")
                parts.append(contradiction_report(cons))
                parts.append("")
        except Exception as e:
            log.debug("contradiction pass skipped: %s", e)

        gap_points = _build_structural_gaps(
            cited or papers,
            correlation or {},
            found_themes=list(theme_groups.keys()),
        )
        parts.append("## Research gaps")
        parts.append("")
        if gap_points:
            for gp_ in gap_points:
                parts.append(f"- {gp_}")
        else:
            parts.append(
                f"The corpus provides broad coverage of {query.lower()}; "
                f"no structural gap signals fired."
            )
        parts.append("")
    except ImportError:
        pass

    # Add meta-analysis summary if available
    if meta and meta.get("pooled_effect"):
        parts.append("## Meta-analysis summary")
        parts.append("")
        pooled = meta["pooled_effect"]
        parts.append(
            f"- Pooled effect: {pooled.get('estimate', 'N/A')} "
            f"(95% CI: {pooled.get('ci_lower', '?')}–{pooled.get('ci_upper', '?')})"
        )
        if meta.get("heterogeneity"):
            parts.append(
                f"- Heterogeneity (I²): {meta['heterogeneity'].get('i_squared', 'N/A')}"
            )
        parts.append("")

    # Citation network analysis
    try:
        from _citation_graph import build_citation_graph, format_citation_graph_summary

        graph_data = build_citation_graph(cited)
        if graph_data:
            graph_md = format_citation_graph_summary(graph_data)
            if graph_md:
                parts.append(graph_md)
    except Exception as e:
        log.warning("Citation-network section skipped: %s", e)

    # References — must cover EVERY [n] used in the body. The narrative
    # builder numbers citations across ALL papers; `cited` may be a subset,
    # which previously left [6][7][8] dangling with only 5 entries listed
    # (caught by characterization test 2026-08-15).
    ref_source = cited if len(cited) >= len(papers) else papers
    parts.append("## References")
    parts.append("")
    parts.append(format_citation_list(ref_source))

    full_brief = "\n".join(parts)

    # Deduplicate the scope sentence: the chronological narrative opens
    # with the same "This review synthesizes N studies ..." sentence the
    # executive summary already carries — keep the first occurrence only.
    scope_sentences = [
        ln for ln in full_brief.splitlines()
        if ln.startswith("This review synthesizes ")
    ]
    for s in scope_sentences[1:]:
        full_brief = full_brief.replace(s + "\n", "", 1).replace(s, "", 1)

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(full_brief, encoding="utf-8")
        log.info(
            "Research brief written to %s (%d chars)", output_path, len(full_brief)
        )

    return full_brief


# =============================================================================
# CLI
# =============================================================================


def main() -> int:
    # Pre-parser self-check — bypasses required-positional validation
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: hard deps verified by bootstrap, ready")
        return 0
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        prog="synthesize",
        description="Phase 5: Narrative synthesis with chronological citations.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--self-check",
        action="store_true",
        help="verify deps + key imports, then exit 0",
    )  # SCIENTIFIC_RESEARCH_SELF_CHECK_WIRED
    p.add_argument(
        "--query",
        required=True,
        help="Original research query (drives research type detection)",
    )
    p.add_argument(
        "--extracted",
        type=Path,
        default=None,
        help="extracted.json from extract.py",
    )
    p.add_argument(
        "--verified",
        type=Path,
        default=None,
        help="verified.json from verify.py",
    )
    p.add_argument(
        "--correlation",
        type=Path,
        default=None,
        help="correlation.json from correlate.py (for verification type)",
    )
    p.add_argument(
        "--meta",
        type=Path,
        default=None,
        help="meta.json from meta_analyze.py (optional)",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path for research_brief.md",
    )
    p.add_argument(
        "--use-llm",
        action="store_true",
        help="Use Ollama LLM for prose smoothing (opt-in, default: template-based)",
    )
    p.add_argument(
        "--print",
        action="store_true",
        help="Also print narrative to stdout",
    )

    args = p.parse_args()
    brief = synthesize(
        query=args.query,
        extracted_path=args.extracted,
        verified_path=args.verified,
        correlation_path=args.correlation,
        meta_path=args.meta,
        output_path=args.output,
        use_llm=args.use_llm,
    )

    if args.print or not args.output:
        print(brief)

    return 0


if __name__ == "__main__":
    sys.exit(main())
