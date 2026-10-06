"""Research pipeline — orchestrates the 5-phase scientific research workflow.

Callable from QThread (ResearchWorker) or directly. Each phase saves
intermediate results to data/research/<query_hash>/ and emits progress
via a callback.

Phases:
1. DISCOVERY    — discover.py: web search across Crossref/OpenAlex/S2/arXiv
2. VERIFICATION — verify.py: DOI verification + retraction check
3. EXTRACTION   — extract.py: PICO + effect sizes + key findings
4. CORRELATION  — correlate.py: citation graph + stance matrix
5. SYNTHESIS    — synthesize.py: narrative brief with citations
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from _query import QueryState

    from config import ResearchConfig

import hashlib
import json
import logging
import re
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import orjson

from _localindex import (
    build_local_index,
    get_local_paper_count,
    load_fulltext,
    match_paper_to_local,
)
from config import ResearchConfig

log = logging.getLogger(__name__)

# Web-pro supplement gate (2026-09-02): a synthesis whose own confidence
# is 0.00 with zero covered facets is noise (live brief: pelagic-sediment
# and natural-gas pages answering a basin query). Below this floor the
# supplement is replaced by a one-line skip note.
_WEB_PRO_MIN_CONFIDENCE = 0.34

# Ensure scripts directory is on sys.path for imports
_SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


ProgressCallback = Callable[[int, str], None]
"""Phase callback: (phase_number 1-5, status_message)."""


class PipelineTimeoutError(Exception):
    """Raised when the pipeline exceeds its wall-clock budget.

    Caught by ``run_pipeline`` to build a partial result from whatever
    phases completed. Carries the phase number and elapsed seconds so the
    caller can report which phase was interrupted.
    """

    def __init__(self, phase: int, elapsed: float, budget: float) -> None:
        self.phase = phase
        self.elapsed = elapsed
        self.budget = budget
        super().__init__(f"Pipeline budget exhausted at phase {phase}: {elapsed:.1f}s > {budget:.1f}s budget")


def _check_budget(t0: float, budget_s: float | None, phase: int) -> None:
    """Raise PipelineTimeout if the wall-clock budget is exceeded.

    Called between phases so a slow Phase 1/2 cannot starve later phases
    and a slow network cannot hang an interactive (chat-tool) caller.
    """
    if budget_s is None:
        return
    elapsed = time.time() - t0
    if elapsed > budget_s:
        raise PipelineTimeoutError(phase, elapsed, budget_s)


def _check_embeddings_available() -> bool:
    """Check if BGE semantic embeddings are available without loading model."""
    try:
        from _embeddings import is_available

        return is_available()
    except ImportError:
        return False


_pipeline_lock = threading.Lock()


# Synthesis version stamp — bumped when narrative builders change in a way
# that affects output (citation numbering, structure, etc.). Persisted to
# meta.json so stale briefs can be detected without re-running synthesis.
# History:
#   2026-07-14 — unified citation index (body [N] aligns with References [N])
SYNTHESIS_VERSION = "2026-07-14"


class PipelineBusyError(Exception):
    """Raised when a pipeline is already running (concurrent invocation blocked)."""


def run_pipeline(
    config: ResearchConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Run the full 5-phase research pipeline.

    Parameters
    ----------
    config : ResearchConfig
        Pipeline configuration (query, sources, max papers, LLM, etc.)
    progress : callable, optional
        Called after each phase: progress(phase_num, message).

    Returns
    -------
    dict
        Results dict with keys:
        - query, research_type, n_papers, n_fulltext_matched
        - paths: {corpus, verified, extracted, correlation, brief}
        - brief_text: the full research brief markdown
        - elapsed: total time in seconds

    Raises
    ------
    PipelineBusyError
        If another pipeline run is already in progress (concurrent
        invocation blocked to prevent API rate-limit cascades).
    """
    if not _pipeline_lock.acquire(blocking=False):
        raise PipelineBusyError(
            "A research pipeline is already running. Wait for it to complete "
            "before starting another to prevent API rate-limit cascades."
        )
    try:
        return _run_pipeline_impl(config, progress)
    finally:
        _pipeline_lock.release()


# ── Research completion callbacks ─────────────────────────────────
# Loose-coupled notification: when any pipeline completes (from dock or chat),
# registered callbacks fire. ResearchDock uses this to display results from
# chat-triggered research without tight coupling between components.
_research_complete_callbacks: list[Callable[[dict], None]] = []

# ── Paper discovery callbacks (streaming) ─────────────────────────
# Fired per-paper during Phase 1 (Discovery) so the GUI can show a live
# feed instead of waiting for the full pipeline to complete.
# Callback signature: (paper: dict) -> None
_paper_callbacks: list[Callable[[dict], None]] = []


def register_research_complete_callback(cb: Callable[[dict], None]) -> None:
    """Register a callback fired when any pipeline run completes.

    The callback receives the full results dict from run_pipeline().
    Thread-safety: callbacks fire from the calling thread (chat worker or
    QThread). Qt consumers MUST defer GUI updates to the GUI thread via
    QTimer.singleShot(0, ...) or signals.
    """
    _research_complete_callbacks.append(cb)


def register_paper_callback(cb: Callable[[dict], None]) -> None:
    """Register a callback fired for each paper discovered during Phase 1.

    Enables streaming discovery — the GUI shows papers as they're found
    instead of waiting for the full pipeline. Thread-safety: same as
    ``register_research_complete_callback``.
    """
    _paper_callbacks.append(cb)


def _notify_research_complete(results: dict) -> None:
    """Fire all registered callbacks (internal, called after pipeline completes)."""
    for cb in _research_complete_callbacks:
        try:
            cb(results)
        except Exception:
            log.debug("Research complete callback failed", exc_info=True)


def _phase_correlation(
    config: ResearchConfig,
    extractions: dict,
    results_dir: Path,
    verified_path: Path,
    emit: Callable[[int, str], None],
    t0: float,
) -> Path:
    """Phase 4: Build citation graph + stance matrix."""
    _check_budget(t0, config.wall_clock_budget_s, 4)
    correlation_path = results_dir / "correlation.json"
    if config.skip_correlate or len(extractions) < 3:
        correlation_path.write_text("{}", encoding="utf-8")
        emit(4, "Correlation skipped (insufficient papers)")
    else:
        emit(4, "Building citation graph + stance matrix...")
        try:
            from _sources import load_corpus
            from correlate import correlate_corpus

            verified_records = load_corpus(verified_path)
            correlation_data = correlate_corpus(
                verified_records,
                no_llm_stance=not config.use_llm,
            )
            correlation_path.write_text(
                orjson.dumps(correlation_data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"),
                encoding="utf-8",
            )
            emit(4, "Correlation complete")
        except Exception as e:
            log.warning("Correlation failed: %s", e)
            correlation_path.write_text("{}", encoding="utf-8")
            emit(4, f"Correlation failed: {e}")
    return correlation_path


def _phase_synthesis(
    config: ResearchConfig,
    results_dir: Path,
    extracted_path: Path,
    verified_path: Path,
    correlation_path: Path,
    meta_analysis_path: Path,
    emit: Callable[[int, str], None],
    t0: float,
    display_query: str = "",
    quant_data_path: Path | None = None,
    research_ctx=None,
    audit_query: str | None = None,
    query_corrections: list[str] | None = None,
) -> tuple[str, Path, Path]:
    """Phase 5: Synthesize research brief + export citations.

    Uses PaperQA2 ``ask()`` (cited, evidence-grounded) when LLM is enabled
    and Ollama is available. Falls back to ``synthesize.py`` otherwise.
    """
    _check_budget(t0, config.wall_clock_budget_s, 5)
    emit(5, "Synthesizing research brief...")
    brief_path = results_dir / "research_brief.md"

    # ── Tier 1: Ollama direct synthesis (PRIMARY) ──
    # Generates brief from structured extraction data — faster + higher quality
    # than PaperQA2 RAG for batch generation because data is already structured.
    # Tier honesty (2026-08-21): "auto" means the documented chain
    # Ollama → PaperQA2 → template. It no longer requires use_llm=True —
    # a live Ollama probe gates it instead. use_llm stays the switch for
    # per-paper LLM extraction; synthesis is only LLM-gated by tier.
    # Explicit tiers fail loud when their engine is unreachable (H1).
    brief_text = None
    _ollama_reachable = False
    if config.synthesis_tier in ("auto", "ollama"):
        try:
            import urllib.request as _ur_t1

            with _ur_t1.urlopen(_ur_t1.Request("http://127.0.0.1:11434/api/tags"), timeout=3):
                _ollama_reachable = True
        except Exception:
            _ollama_reachable = False
        if config.synthesis_tier == "ollama" and not _ollama_reachable:
            raise RuntimeError(
                "synthesis_tier='ollama' but Ollama is unreachable at "
                "http://127.0.0.1:11434 — start Ollama or pick another tier"
            )

    if _ollama_reachable and config.synthesis_tier in ("auto", "ollama"):
        try:
            from _ollama_extract import generate_brief_ollama

            if extracted_path.exists():
                import json as _json_t1

                ext_data = _json_t1.loads(extracted_path.read_text())
                if isinstance(ext_data, dict):
                    ext_data = ext_data.get("extractions", [])

                # ── Enrich extractions with author names from verified papers ──
                # Extraction data lacks author info; PaperRecord has it.
                # Match by DOI to produce clean (Author, Year) citations.
                if isinstance(ext_data, list) and ext_data and verified_path.exists():
                    from _sources import load_corpus as _load_for_authors

                    vp = _load_for_authors(verified_path)
                    author_map: dict[str, str] = {}
                    year_map: dict[str, int | None] = {}
                    doi_title_map: dict[str, str] = {}
                    n_crossref_lookup = 0
                    n_crossref_failed = 0
                    for p in vp:
                        if not p.doi:
                            continue
                        names = [a.get("name", "") if isinstance(a, dict) else str(a) for a in (p.authors or [])]
                        # Fallback: Crossref API lookup when authors missing
                        if not names and p.doi:
                            try:
                                import json as _cj
                                import urllib.request as _cu

                                _cr_url = f"https://api.crossref.org/works/{p.doi}"
                                _cr_req = _cu.Request(
                                    _cr_url,
                                    headers={"User-Agent": "scientific-research-skill/1.0"},
                                )
                                _cr_resp = _cj.loads(_cu.urlopen(_cr_req, timeout=5).read())
                                _msg = _cr_resp.get("message", {})
                                _cr_authors = _msg.get("author", [])
                                names = [f"{a.get('given', '')} {a.get('family', '')}".strip() for a in _cr_authors]
                                if not p.year:
                                    _dp = _msg.get("published", {}).get("date-parts", [[None]])
                                    if _dp and _dp[0] and _dp[0][0]:
                                        year_map[p.doi] = _dp[0][0]
                                if not p.title:
                                    _titles = _msg.get("title", [])
                                    if _titles:
                                        p.title = _titles[0]
                                n_crossref_lookup += 1
                            except Exception:
                                n_crossref_failed = n_crossref_failed + 1
                                log.debug("crossref lookup failed", exc_info=True)
                        if names:
                            surname = names[0].split()[-1] if names[0].strip() else "Unknown"
                            author_map[p.doi] = surname
                        if p.year:
                            year_map[p.doi] = p.year
                        if p.title:
                            doi_title_map[p.doi] = p.title
                    if n_crossref_lookup:
                        log.info("Author enrichment: %d papers looked up via Crossref", n_crossref_lookup)
                    if n_crossref_failed:
                        log.warning("Author enrichment: %d Crossref lookups failed", n_crossref_failed)
                    # Merge into extraction dicts
                    for ext in ext_data:
                        if isinstance(ext, dict):
                            doi = ext.get("doi", "")
                            if doi in author_map:
                                ext["first_author"] = author_map[doi]
                            if doi in year_map:
                                ext["year"] = year_map[doi]
                            # Also try matching by paper_id or title
                            if not ext.get("first_author") or ext.get("first_author") == "Unknown":
                                pid = ext.get("paper_id", "")
                                if pid in author_map:
                                    ext["first_author"] = author_map[pid]
                                    if pid in year_map:
                                        ext["year"] = year_map[pid]

                if isinstance(ext_data, list) and ext_data:
                    emit(5, "Ollama direct synthesis (structured data)...")
                    t_brief = time.time()

                    # Load quantitative data from PDF tables (Phase 3.5)
                    quant_summary = ""
                    if quant_data_path and quant_data_path.exists():
                        try:
                            import json as _json

                            quant_list = _json.loads(quant_data_path.read_text())
                            if quant_list:
                                quant_lines = []
                                for qd in quant_list:
                                    for var, stats in (qd.get("summary") or {}).items():
                                        quant_lines.append(
                                            f"  {var}: {stats.get('min', '?')}-{stats.get('max', '?')} "
                                            f"(mean={stats.get('mean', '?'):.2f}, n={stats.get('n', 0)})"
                                            if isinstance(stats.get("mean"), (int, float))
                                            else f"  {var}: range data"
                                        )
                                    for calc_name, calc_val in (qd.get("calculations") or {}).items():
                                        if isinstance(calc_val, dict):
                                            quant_lines.append(f"  {calc_name}: {calc_val}")
                                        else:
                                            quant_lines.append(f"  {calc_name} = {calc_val}")
                                if quant_lines:
                                    quant_summary = (
                                        "\n\n## QUANTITATIVE DATA FROM PAPER TABLES\n"
                                        "Use these ACTUAL measured values in your synthesis. "
                                        "Cite them with the paper's [ref_id]:\n" + "\n".join(quant_lines) + "\n"
                                    )
                                    log.info(
                                        "Enriching synthesis with %d quantitative data points",
                                        len(quant_lines),
                                    )
                        except Exception as e:
                            log.debug("Quantitative data load failed: %s", e)

                    brief_text = generate_brief_ollama(
                        ext_data,
                        config.query,
                        timeout=max(120, config.paperqa_timeout_s),
                        quality=config.synthesis_quality,
                        extra_context=quant_summary,
                    )
                    if brief_text and len(brief_text) > 500:
                        # Header (title/corpus/method) is prepended by
                        # generate_brief_ollama — it knows the quality-gated
                        # papers_data count; len(ext_data) overcounted vs
                        # the reference list (2026-09-03).
                        brief_path.write_text(brief_text, encoding="utf-8")
                        elapsed_brief = time.time() - t_brief
                        log.info(
                            "Ollama direct synthesis: %d chars in %.1fs",
                            len(brief_text),
                            elapsed_brief,
                        )
                        emit(5, f"Ollama synthesis complete ({len(brief_text)} chars)")
                    else:
                        log.warning("Ollama direct synthesis too short — trying PaperQA2")
                        brief_text = None
        except ImportError:
            log.debug("_ollama_extract not available for direct synthesis")
        except Exception as e:
            log.warning("Ollama direct synthesis failed: %s — trying PaperQA2", e)
            brief_text = None

    # ── Tier 2: PaperQA2 RAG synthesis (SECONDARY) ──
    if brief_text is None and config.use_llm and config.synthesis_tier in ("auto", "paperqa"):
        # PaperQA2 artifacts persist under results_dir/paperqa_artifacts
        # (retention directive 2026-09-03) — no tempdir, no cleanup.
        try:
            from _paperqa import is_available

            if is_available():
                emit(5, "PaperQA2 synthesis (evidence-grounded)...")
                import json as _json
                import tempfile

                from _sources import load_corpus as _load_for_pqa

                verified_papers = _load_for_pqa(verified_path)

                # ── Pre-filter corpus by Ollama mineral classification ──
                # PaperQA2's vector retrieval can surface wrong papers when the
                # corpus has mixed mineral systems (e.g., garnet-clinopyroxene
                # alongside garnet-biotite). Use Ollama classification from
                # Phase 3 to keep only papers whose minerals match the query.
                extracted_path_pqa = results_dir / "extracted.json"
                pqa_papers = verified_papers
                if extracted_path_pqa.exists():
                    try:
                        ext_data = _json.loads(extracted_path_pqa.read_text())
                        if isinstance(ext_data, list):
                            # Build DOI → minerals map from extraction data
                            min_map: dict[str, list[str]] = {}
                            for ext in ext_data:
                                if isinstance(ext, dict):
                                    doi = ext.get("doi", "")
                                    minerals = ext.get("pico", {}).get("minerals", [])
                                    if doi and minerals:
                                        min_map[doi] = [m.lower() for m in minerals]

                            # Extract key minerals from the query
                            query_lower = config.query.lower()
                            query_minerals: list[str] = []
                            for mineral in (
                                "garnet",
                                "biotite",
                                "clinopyroxene",
                                "orthopyroxene",
                                "amphibole",
                                "olivine",
                                "feldspar",
                                "quartz",
                                "muscovite",
                                "chlorite",
                                "staurolite",
                                "cordierite",
                                "sillimanite",
                                "kyanite",
                                "andalusite",
                            ):
                                if mineral in query_lower:
                                    query_minerals.append(mineral)

                            if query_minerals and min_map:
                                # Keep papers that mention ALL query minerals
                                filtered = [
                                    p
                                    for p in verified_papers
                                    if not p.doi
                                    or p.doi
                                    not in min_map  # unclassified = keep (2026-09-01: reflect-added papers carry no pico)
                                    or all(any(qm in pm for pm in min_map[p.doi]) for qm in query_minerals)
                                ]
                                # Safety: don't filter below 3 papers
                                if len(filtered) >= 3:
                                    pqa_papers = filtered
                                    emit(
                                        5,
                                        f"Pre-filtered corpus: {len(pqa_papers)}/{len(verified_papers)} "
                                        f"papers (minerals: {query_minerals})",
                                    )
                                    log.info(
                                        "PaperQA2 corpus pre-filtered: %d/%d papers (query minerals: %s)",
                                        len(pqa_papers),
                                        len(verified_papers),
                                        query_minerals,
                                    )
                    except Exception as e:
                        log.debug("Corpus pre-filter skipped: %s", e)
                # RETENTION (Master directive 2026-09-03, mirrors chat
                # tool-call data handling): PaperQA2 artifacts persist under
                # the run dir for future reuse — no auto-deleted tempdirs.
                tmp_pqa = results_dir / "paperqa_artifacts"
                tmp_pqa.mkdir(parents=True, exist_ok=True)

                # ── FIX 1: Download OA PDFs BEFORE indexing ──
                # Gives PaperQA2 10x more evidence (full-text sections vs abstracts)
                n_fulltext_pqa = 0
                try:
                    from _fulltext import enrich_papers

                    paper_dicts = [
                        {
                            "doi": p.doi,
                            "oa_pdf_url": getattr(p, "oa_pdf_url", None) or "",
                            "title": p.title or "",
                        }
                        for p in verified_papers
                        if p.doi
                    ]
                    if paper_dicts:
                        emit(5, f"PaperQA2: downloading {len(paper_dicts)} OA PDFs...")
                        n_ft, _n_fail = enrich_papers(
                            paper_dicts,
                            max_downloads=config.max_pdf_downloads,
                            timeout_per_download=15.0,
                        )
                        n_fulltext_pqa = n_ft
                        enriched_map = {pd["doi"]: pd for pd in paper_dicts if pd.get("has_fulltext")}
                        log.info("PaperQA2: %d/%d papers with full-text", n_ft, len(paper_dicts))
                except Exception as e:
                    log.warning("PaperQA2 full-text download failed (abstracts only): %s", e)
                    enriched_map = {}

                # ── Write papers to temp dir (abstracts + full-text sections) ──
                for i, p in enumerate(pqa_papers):
                    text = f"# {p.title or 'Untitled'}\n\n"
                    if p.doi:
                        text += f"DOI: {p.doi}\n"
                    if p.year:
                        text += f"Year: {p.year}\n"
                    if p.venue:
                        text += f"Journal: {p.venue}\n"
                    text += f"\n{(p.abstract or 'No abstract available')}\n"

                    # Phase 1.6 enrichment (from _enrich_fulltext)
                    for attr in (
                        "_methods_text",
                        "_results_text",
                        "_discussion_text",
                        "_conclusions_text",
                    ):
                        section = getattr(p, attr, None)
                        if section:
                            sname = attr.replace("_text", "").strip("_").capitalize()
                            text += f"\n\n{sname}: {section}\n"

                    # FIX 1: PaperQA2-specific enrichment (just downloaded)
                    ed = enriched_map.get(p.doi, {}) if p.doi else {}
                    for skey, sname in [
                        ("methods_text", "Methods"),
                        ("results_text", "Results"),
                        ("discussion_text", "Discussion"),
                        ("conclusions_text", "Conclusions"),
                    ]:
                        section = ed.get(skey)
                        if section:
                            text += f"\n\n{sname}: {section}\n"

                    if getattr(p, "full_text", None):
                        text += f"\n{p.full_text}\n"
                    (tmp_pqa / f"paper_{i + 1:02d}.txt").write_text(text, encoding="utf-8")

                # ── FIX 2: Focused multi-query synthesis ──
                # Questions explicitly name the mineral system from the query
                # to prevent PaperQA2 from retrieving evidence about different
                # mineral systems in the corpus.
                query_topic = config.research_type_override or config.query
                n_papers = len(pqa_papers)
                # Adaptive questions (2026-09-01): LLM designs lenses from the
                # actual corpus instead of the fixed 3 — falls back to the
                # domain-generic templates below on any generation failure.
                adaptive_qs = None
                if getattr(config, "adaptive_questions", False) and _ollama_reachable:
                    try:
                        from _reflect import generate_adaptive_questions

                        # Whole abstracts — generate_adaptive_questions
                        # assembles them via the non-truncating budget layer.
                        _abstracts = "\n\n".join(
                            f"({i + 1}) {(pp.title or '')}: {(pp.abstract or '')}"
                            for i, pp in enumerate(pqa_papers[:12])
                        )
                        adaptive_qs = generate_adaptive_questions(query_topic, _abstracts)
                    except Exception as exc:
                        emit(5, f"Adaptive questions failed ({type(exc).__name__}) — template lenses")
                if adaptive_qs:
                    questions = [(f"Adaptive {i + 1}", q) for i, q in enumerate(adaptive_qs)]
                    emit(
                        5,
                        f"PaperQA2: {len(adaptive_qs)} adaptive questions (LLM-designed from corpus)",
                    )
                else:
                    # Domain-generic questions — the previous hardcoded P-T /
                    # mineral-pair framing asked metamorphic-petrology questions
                    # of EVERY corpus (dyke geochronology got asked about
                    # "Ferry & Spear calibrations").
                    questions = [
                        (
                            "Methods and Data",
                            f"What analytical methods, instruments, and calibration or "
                            f"measurement approaches are used to study {query_topic} in "
                            f"these papers? Name specific techniques, standards, and "
                            f"reported uncertainties. Describe limitations the authors "
                            f"acknowledge. If a paper does not address the topic, say so "
                            f"— do not extrapolate. Cite papers as (Author, Year).",
                        ),
                        (
                            "Quantitative Results",
                            f"What quantitative results (ages, temperatures, pressures, "
                            f"compositions, velocities, or whatever quantities this "
                            f"corpus reports) are given for {query_topic}? Include "
                            f"specific values WITH their units and contexts (rock type, "
                            f"location, sample, setting). Only report numbers that "
                            f"appear in the papers. Cite papers as (Author, Year).",
                        ),
                        (
                            "Interpretations and Disagreements",
                            f"How do the authors interpret their results for "
                            f"{query_topic}, and where do studies disagree? Compare "
                            f"conclusions across papers, note whether disagreements are "
                            f"methodological or geological, and flag any question the "
                            f"corpus cannot answer. Cite papers as (Author, Year).",
                        ),
                    ]

                # ── Run focused queries with single-index multi-query ──
                # Indexes papers ONCE, then queries N times (avoids 3x re-indexing overhead)
                try:
                    from _paperqa import docs_query_multi
                except ImportError:
                    docs_query_multi = None

                question_texts = [q for _, q in questions]
                emit(5, f"PaperQA2: {len(questions)} focused queries (single index)...")

                if docs_query_multi:
                    pqa_results = docs_query_multi(
                        question_texts,
                        paper_dir=str(tmp_pqa),
                        max_sources=n_papers,
                        answer_length="about 600 words",
                        timeout=config.paperqa_timeout_s,
                        evidence_k=6,
                        chunk_chars=2000,
                    )
                else:
                    from _paperqa import docs_query as _dq

                    per_q_to = max(120, config.paperqa_timeout_s // len(questions))
                    pqa_results = []
                    for qt in question_texts:
                        r = _dq(
                            qt,
                            paper_dir=str(tmp_pqa),
                            max_sources=n_papers,
                            answer_length="about 600 words",
                            timeout=per_q_to,
                            evidence_k=6,
                            chunk_chars=2000,
                        )
                        r.setdefault("error", None)
                        pqa_results.append(r)

                # Collect successful sections
                sections: list[str] = []
                for i, (section_title, _) in enumerate(questions):
                    pqa_r = pqa_results[i] if i < len(pqa_results) else {}
                    answer = pqa_r.get("answer", "")
                    if (
                        answer
                        and isinstance(answer, str)
                        and len(answer) > 100
                        and not answer.startswith("PaperQA2 timed out")
                        and not answer.startswith("Q&A failed:")
                        and "I cannot answer" not in answer[:100]
                    ):
                        sections.append(f"## {section_title}\n\n{answer}")
                        log.info("PaperQA2 '%s': %d chars", section_title, len(answer))
                    else:
                        err = pqa_r.get("error", "unknown")
                        log.warning(
                            "PaperQA2 query '%s' failed (%s): %s",
                            section_title,
                            err,
                            (answer or "empty")[:120],
                        )

                # FIX 5: Combine answers into structured brief
                if sections:
                    import time as _time

                    brief_text = (
                        f"# Research Brief: {query_topic}\n\n"
                        f"**Corpus**: {n_papers} papers "
                        f"({n_fulltext_pqa} full-text)\n"
                        f"**Generated**: {_time.strftime('%Y-%m-%d %H:%M')}\n"
                        f"**Method**: PaperQA2 (evidence-grounded RAG)\n\n"
                        f"---\n\n"
                    )
                    brief_text += "\n\n---\n\n".join(sections)
                    brief_text += "\n"
                    brief_path.write_text(brief_text, encoding="utf-8")
                    log.info(
                        "PaperQA2 synthesis: %d sections, %d chars total, %d full-text",
                        len(sections),
                        len(brief_text),
                        n_fulltext_pqa,
                    )
                    emit(
                        5,
                        f"PaperQA2 synthesis complete ({len(sections)}/{len(questions)} sections)",
                    )
                else:
                    log.warning("All PaperQA2 queries failed — falling back to template synthesis")
                    brief_text = None
        except Exception as e:
            log.warning("PaperQA2 synthesis failed, falling back: %s", e)
            brief_text = None
        # PaperQA2 artifacts persist under results_dir/paperqa_artifacts —
        # no cleanup (retention directive 2026-09-03).
    if brief_text is None and config.synthesis_tier in ("auto", "template"):
        from synthesize import synthesize as run_synthesize

        meta_for_synth = meta_analysis_path if meta_analysis_path.exists() else None
        brief_text = run_synthesize(
            query=display_query or config.research_type_override or config.query,
            extracted_path=extracted_path,
            verified_path=verified_path,
            correlation_path=correlation_path,
            meta_path=meta_for_synth,
            output_path=brief_path,
            use_llm=config.use_llm,
            query_corrections=query_corrections,
        )
    # Guarantee brief_text is always a string (never None)
    if not brief_text:
        brief_text = "*Synthesis produced no output. Check logs for errors.*"
        brief_path.write_text(brief_text, encoding="utf-8")
    emit(5, "Synthesis complete")

    # ── Reproducibility statement (retention directive 2026-09-03):
    # every brief advertises its own full audit trail.
    brief_text += (
        "\n\n---\n\n## Data and Code Availability\n\n"
        "All inputs are retained in this run directory for reproduction and "
        "audit: full LLM call journal (prompts and responses, "
        "llm_calls.jsonl), screened corpus (corpus.json), extractions "
        "(extracted.json), screening decisions, and per-artifact SHA-256 "
        "hashes with the complete configuration snapshot (meta.json).\n"
    )
    brief_path.write_text(brief_text, encoding="utf-8")

    # ── Web synthesis supplement (Perplexity-style, native) ──
    # Advisory section appended after the verified corpus synthesis.
    # Cites URLs (not DOIs) — clearly labeled, never mixed into the
    # H20-verified reference list.
    if config.use_web_pro:
        emit(5, "Web synthesis supplement (native web search)...")
        try:
            from _websearch import run_pro_synthesis

            pro = run_pro_synthesis(audit_query or config.query)
            if pro is None:
                log.warning("Web synthesis supplement returned nothing — brief unchanged")
            else:
                _pro_syn = pro.get("synthesis") or ""
                _pro_conf = float(pro.get("confidence", 0.0) or 0.0)
                _pro_facets = pro.get("facets") or []
                _pro_cov = pro.get("coverage") or {}
                _pro_covered = sum(1 for f in _pro_facets if _pro_cov.get(f))
                if _pro_syn and _pro_conf >= _WEB_PRO_MIN_CONFIDENCE and _pro_covered >= 1:
                    brief_text += "\n\n---\n\n" + _pro_syn + "\n"
                    brief_path.write_text(brief_text, encoding="utf-8")
                    log.info(
                        "Web synthesis supplement: %d chars, confidence %.2f, %d sources",
                        len(_pro_syn),
                        _pro_conf,
                        len(pro.get("sources", [])),
                    )
                else:
                    brief_text += (
                        "\n\n---\n\n*Web synthesis skipped: "
                        f"{_pro_covered}/{len(_pro_facets)} facets covered "
                        f"(confidence {_pro_conf:.2f} < {_WEB_PRO_MIN_CONFIDENCE:.2f}) "
                        "— web results did not address the query.*\n"
                    )
                    brief_path.write_text(brief_text, encoding="utf-8")
                    log.warning(
                        "Web synthesis supplement DROPPED: confidence %.2f, %d/%d facets covered",
                        _pro_conf,
                        _pro_covered,
                        len(_pro_facets),
                    )
                    emit(
                        5,
                        f"Web synthesis dropped (confidence {_pro_conf:.2f}, {_pro_covered}/{len(_pro_facets)} facets)",
                    )
        except Exception as e:
            log.warning("Web synthesis supplement failed: %s", e)

    # ── Grounding audit: entities asserted in brief but unsupported ─────
    # Deterministic post-check: if the synthesis mentions a distinctive
    # query term that ZERO verified papers discuss, append an explicit
    # warning — the claim came from the query string, not the literature.
    # Audits the NORMALIZED query (audit_query) — the raw config.query may
    # carry the very typos the gates would then disown (split-brain fix,
    # 2026-09-02). Idempotent: strip-and-replace via append_grounding_warning.
    try:
        from _honesty import (
            append_grounding_warning,
            corpus_coverage,
            extract_query_entities,
            grounding_audit,
        )
        from _sources import load_corpus as _load_for_audit

        _audit_q = audit_query or config.query
        _audit_ents = extract_query_entities(_audit_q)
        _audit_cov = corpus_coverage(_load_for_audit(verified_path), _audit_ents)
        _unsupported = grounding_audit(brief_text, _audit_ents, _audit_cov)
        _brief_before = brief_text
        brief_text = append_grounding_warning(brief_text, _unsupported, _audit_cov)
        if brief_text != _brief_before:
            brief_path.write_text(brief_text, encoding="utf-8")
        if _unsupported:
            log.warning(
                "Grounding audit: brief asserts entities unsupported by corpus: %s",
                _unsupported,
            )
            emit(5, f"Grounding warning appended: {_unsupported}")
    except Exception as e:
        log.warning("Grounding audit skipped: %s", e)

    # Citation Export
    citations_bib_path = results_dir / "citations.bib"
    try:
        from _sources import load_corpus
        from export_citations import export_bibtex

        cited_records = load_corpus(verified_path)
        bib_text = export_bibtex(cited_records)
        citations_bib_path.write_text(bib_text, encoding="utf-8")
        log.info("Citations exported: %d entries (BibTeX)", len(cited_records))
    except Exception as e:
        log.warning("Citation export failed: %s", e)

    return brief_text, brief_path, citations_bib_path


def _enrich_fulltext(
    papers: list,
    config: ResearchConfig,
    emit: Callable[[int, str], None],
) -> int:
    """Phase 1.6: Download OA PDFs for top papers, extract sections.

    Captures P-T values, mineral chemistry, and calibration details
    that abstracts miss — the single biggest improvement for depth.
    """
    n_enriched = 0
    try:
        from _fulltext import enrich_papers

        oa_papers = [
            p
            for p in papers[: min(config.max_papers, 10)]
            if getattr(p, "oa_pdf_url", None) and getattr(p, "doi", None)
        ]
        if oa_papers:
            paper_dicts = [{"doi": p.doi, "oa_pdf_url": p.oa_pdf_url, "title": p.title} for p in oa_papers]
            emit(1, f"Full-text enrichment: {len(paper_dicts)} OA papers...")
            t_ft = time.time()
            n_enriched, _n_ft_failed = enrich_papers(
                paper_dicts,
                max_downloads=config.max_pdf_downloads,
                timeout_per_download=15.0,
            )
            enriched_by_doi = {pd["doi"]: pd for pd in paper_dicts if pd.get("has_fulltext")}
            for p in papers:
                if p.doi and p.doi in enriched_by_doi:
                    ed = enriched_by_doi[p.doi]
                    p._methods_text = ed.get("methods_text", "")
                    p._results_text = ed.get("results_text", "")
                    p._discussion_text = ed.get("discussion_text", "")
                    p._conclusions_text = ed.get("conclusions_text", "")
            log.info(
                "Full-text enrichment: %d/%d papers enriched in %.1fs",
                n_enriched,
                len(oa_papers),
                time.time() - t_ft,
            )
    except Exception as e:
        log.warning("Full-text enrichment skipped (PDF sections not extracted): %s", e)
    return n_enriched


def _phase_quantitative_extraction(
    config: ResearchConfig,
    results_dir: Path,
    extractions: list[dict],
    emit: Callable[[int, str], None],
    t0: float,
) -> Path | None:
    """Phase 3.5: Extract quantitative data from downloaded PDF tables.

    AUTOMATIC: runs when PDFs are available from Phase 1.6 download.
    GUARDED: any error → log + skip, never crashes pipeline.
    ADDITIVE: adds quantitative data file for synthesis, doesn't change
    existing extraction results.

    Returns path to quantitative_data.json, or None if no data extracted.
    """
    _check_budget(t0, config.wall_clock_budget_s, 3)
    output_path = results_dir / "quantitative_data.json"

    try:
        from _data_extractor import (
            DOMAINS,
            ExtractedDataset,
            extract_domain_data,
        )
    except ImportError:
        log.debug("Data extractor module unavailable — skipping Phase 3.5")
        return None

    # Detect domain from query
    try:
        from _semantic_context import classify_context_semantic

        domain = classify_context_semantic(config.query) or "geochemistry"
    except Exception:
        domain = "geochemistry"

    if domain not in DOMAINS:
        domain = "geochemistry"  # fallback

    # Find downloaded PDFs in the fulltext cache
    from pathlib import Path as _Path

    cache_dir = _Path.home() / ".cache" / "scientific-research" / "fulltext"
    pdf_files = list(cache_dir.glob("*.pdf")) if cache_dir.exists() else []

    if not pdf_files:
        log.debug("No cached PDFs found — skipping quantitative extraction")
        return None

    emit(3, f"Extracting {domain} data from {len(pdf_files)} PDFs...")
    log.info("Phase 3.5: quantitative data extraction (%s, %d PDFs)", domain, len(pdf_files))

    all_datasets: list[dict] = []
    for pdf_path in pdf_files[:10]:  # cap at 10 PDFs for speed
        try:
            dataset = extract_domain_data(pdf_path, domain=domain)
            if dataset and dataset.tables:
                # Run GeoKit calculations
                calculations = run_domain_calculations(dataset)
                all_datasets.append(
                    {
                        "source": pdf_path.name,
                        "domain": domain,
                        "n_samples": dataset.n_samples,
                        "n_variables": dataset.n_variables,
                        "summary": dataset.summary,
                        "calculations": calculations,
                        "tables": [
                            {
                                "headers": tbl["headers"],
                                "n_rows": len(tbl["rows"]),
                            }
                            for tbl in dataset.tables
                        ],
                    }
                )
                log.info(
                    "  %s: %d samples, %d variables",
                    pdf_path.name,
                    dataset.n_samples,
                    dataset.n_variables,
                )
        except Exception as e:
            log.debug("Data extraction failed for %s: %s", pdf_path.name, e)

    if not all_datasets:
        log.info("Phase 3.5: no quantitative tables found in PDFs")
        return None

    # Save results
    import json

    output_path.write_text(
        json.dumps(all_datasets, indent=2, default=str),
        encoding="utf-8",
    )
    total_samples = sum(d["n_samples"] for d in all_datasets)
    emit(3, f"Extracted {total_samples} samples from {len(all_datasets)} papers")
    log.info(
        "Phase 3.5 complete: %d datasets, %d total samples → %s",
        len(all_datasets),
        total_samples,
        output_path.name,
    )
    return output_path


def _enrich_abstracts(
    papers: list,
    config: ResearchConfig,
    emit: Callable[[int, str], None],
    corpus_path: Path,
) -> None:
    """Phase 1.5: Parallel abstract enrichment via S2 + OpenAlex.

    Mutates ``papers`` in-place — fills missing abstracts.
    """
    abstractless = [p for p in papers if not (p.abstract or "").strip() and p.doi]
    if not abstractless:
        return

    oa_targets = [p for p in abstractless if "openalex" not in (p.sources_seen or [])]
    t_enrich = time.time()
    emit(1, f"Enriching {len(abstractless)} abstracts (S2 + OpenAlex sequential)...")

    def _enrich_s2(targets: list) -> int:
        count = 0
        if not targets:
            return 0
        try:
            from _sources import _s2_available, s2_get_papers_batch

            if not _s2_available():
                log.info("S2 circuit open — skipping S2 enrichment")
                return 0
            identifiers = [f"DOI:{p.doi}" for p in targets if p.doi]
            if not identifiers:
                return 0
            batch = s2_get_papers_batch(identifiers)
            for p in targets:
                if not p.doi:
                    continue
                rec = batch.get(f"DOI:{p.doi}")
                if rec and (rec.abstract or "").strip() and not (p.abstract or "").strip():
                    p.abstract = rec.abstract
                    if "s2" not in p.sources_seen:
                        p.sources_seen.append("s2")
                    count += 1
        except Exception as e:
            log.warning("S2 enrichment error: %s", e)
        return count

    def _enrich_openalex(targets: list) -> int:
        count = 0
        try:
            from _sources import openalex_get_by_doi

            for i, p in enumerate(targets):
                rec = openalex_get_by_doi(p.doi)
                if rec and (rec.abstract or "").strip() and not (p.abstract or "").strip():
                    p.abstract = rec.abstract
                    if "openalex" not in p.sources_seen:
                        p.sources_seen.append("openalex")
                    count += 1
                if config.api_rate_limit_s > 0 and i < len(targets) - 1:
                    time.sleep(config.api_rate_limit_s)
        except Exception as e:
            log.warning("OpenAlex enrichment error: %s", e)
        return count

    s2_count = 0
    if "s2" in config.sources:
        s2_count = _enrich_s2(abstractless)
    oa_count = _enrich_openalex(oa_targets)

    total_enriched = s2_count + oa_count
    log.info(
        "Abstract enrichment: S2=%d OpenAlex=%d total=%d/%d in %.1fs",
        s2_count,
        oa_count,
        total_enriched,
        len(abstractless),
        time.time() - t_enrich,
    )
    if total_enriched:
        from _sources import save_corpus

        save_corpus(
            papers,
            corpus_path,
            meta={
                "query": config.query,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "s2_enriched": s2_count,
                "oa_enriched": oa_count,
            },
        )


def _phase_verification(
    config: ResearchConfig,
    results_dir: Path,
    corpus_path: Path,
    papers: list,
    emit: Callable[[int, str], None],
    t0: float,
) -> tuple[Path, int, int]:
    """Phase 2: DOI/arXiv verification + retraction check.

    Returns (verified_path, n_verified, n_failed_verify).
    """
    _check_budget(t0, config.wall_clock_budget_s, 2)
    t_phase = time.time()
    verified_path = results_dir / "verified.json"
    if config.skip_verify:
        log.warning(
            "VERIFICATION SKIPPED — %d papers included WITHOUT DOI/arXiv "
            "verification (§5 H20 bypass). Results may contain invalid "
            "identifiers or retracted papers.",
            len(papers),
        )
        verified_path.write_text(corpus_path.read_text(encoding="utf-8"), encoding="utf-8")
        n_verified = len(papers)
        n_failed_verify = 0
        emit(2, f"Verification skipped: {n_verified} papers (UNVERIFIED)")
        return verified_path, n_verified, n_failed_verify

    emit(2, f"Verifying {len(papers)} papers...")
    from _sources import load_corpus, save_corpus
    from verify import verify_paper

    corpus_papers = load_corpus(corpus_path)
    verified_records: list = []
    failed = 0
    rate_delay = config.api_rate_limit_s
    dns_failures_consecutive = 0
    DNS_FAILURE_THRESHOLD = 5
    dns_tripped = False

    for i, paper in enumerate(corpus_papers):
        if dns_tripped:
            verified_records.append(paper)
            continue
        # Skip verification for papers without DOI or arXiv — can't verify
        if not paper.doi and not paper.arxiv_id:
            verified_records.append(paper)
            continue
        try:
            merged, result = verify_paper(paper)
            if result.resolved:
                verified_records.append(merged)
                dns_failures_consecutive = 0
            else:
                failed += 1
                reasons = "; ".join(result.resolvers_failed) or "no resolvers tried"
                if any(
                    "name or service not known" in r.lower()
                    or "name resolution" in r.lower()
                    or "gaierror" in r.lower()
                    for r in result.resolvers_failed
                ):
                    dns_failures_consecutive += 1
                else:
                    dns_failures_consecutive = 0
                if dns_failures_consecutive >= DNS_FAILURE_THRESHOLD:
                    dns_tripped = True
                    remaining = len(corpus_papers) - i - 1
                    log.error(
                        "DNS circuit breaker tripped after %d consecutive DNS "
                        "failures — skipping verification of remaining %d papers. "
                        "Check network connectivity. Papers will be kept unverified.",
                        dns_failures_consecutive,
                        remaining,
                    )
                    emit(2, f"DNS down — skipping {remaining} remaining verifications")
                else:
                    log.warning(
                        "  Verification FAILED for %s: %s",
                        paper.primary_id,
                        reasons,
                    )
        except Exception as e:
            failed += 1
            err_str = str(e).lower()
            if "name or service not known" in err_str or "name resolution" in err_str or "gaierror" in err_str:
                dns_failures_consecutive += 1
                if dns_failures_consecutive >= DNS_FAILURE_THRESHOLD:
                    dns_tripped = True
                    remaining = len(corpus_papers) - i - 1
                    log.error(
                        "DNS circuit breaker tripped — skipping %d remaining papers. Check network connectivity.",
                        remaining,
                    )
                    emit(2, f"DNS down — skipping {remaining} remaining verifications")
            else:
                dns_failures_consecutive = 0
                log.warning(
                    "  Verification FAILED for %s: %s",
                    paper.primary_id,
                    e,
                )
        if rate_delay > 0 and i < len(corpus_papers) - 1:
            time.sleep(rate_delay)
        if (i + 1) % 5 == 0:
            emit(2, f"Verifying... {i + 1}/{len(corpus_papers)}")

    n_failed_verify = failed
    if failed:
        failure_rate = failed / len(corpus_papers) * 100 if corpus_papers else 0
        log.warning(
            "  %d/%d papers failed verification (%.0f%%) — dropped from results",
            failed,
            len(corpus_papers),
            failure_rate,
        )
        if failure_rate > 20:
            log.warning(
                "  HIGH verification failure rate (>20%%) — consider checking network connectivity or API rate limits"
            )

    save_corpus(
        verified_records,
        verified_path,
        meta={
            "query": config.query,
            "total_in": len(corpus_papers),
            "verified": len(verified_records),
            "failed_verification": failed,
        },
    )
    n_verified = len(verified_records)
    log.info(
        "Phase 2 timing: %.1fs (%d/%d verified)",
        time.time() - t_phase,
        n_verified,
        len(corpus_papers),
    )
    emit(2, f"Verified: {n_verified} papers")
    return verified_path, n_verified, n_failed_verify


def _phase_meta_and_assessment(
    extractions: list[dict],
    verified_records: list,
    config: ResearchConfig,
    results_dir: Path,
    emit: Callable[[int, str], None],
) -> tuple[Path, Path]:
    """Phase 3b/3c: Meta-analysis pooling + risk-of-bias assessment.

    Returns (meta_analysis_path, assessment_path). Both may not exist
    if insufficient data or modules unavailable.
    """
    meta_analysis_path = results_dir / "meta_analysis.json"
    try:
        from meta_analyze import collect_continuous_effects, pool_effects

        continuous_studies = collect_continuous_effects(extractions, verified_records)
        if continuous_studies:
            pooled = pool_effects(continuous_studies, model="random")
            meta_analysis_path.write_text(
                orjson.dumps(
                    {
                        "n_studies": len(continuous_studies),
                        "model": "random",
                        "pooled_effect": pooled,
                    },
                    option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY,
                ).decode("utf-8"),
                encoding="utf-8",
            )
            log.info("Meta-analysis: %d studies pooled", len(continuous_studies))
            emit(3, f"Meta-analysis: {len(continuous_studies)} studies pooled")
        else:
            log.info("Meta-analysis: no continuous effect sizes found")
    except Exception as e:
        log.warning("Meta-analysis failed: %s", e)

    assessment_path = results_dir / "assessment.json"
    try:
        from assess import build_assessment

        assessments = []
        for record in verified_records:
            try:
                assessment = build_assessment(
                    record,
                    study_design_hint="",
                    abstract=record.abstract or "",
                    topic=config.query,
                )
                assessments.append(assessment)
            except Exception as e:
                log.debug("Assessment build failed for %s: %s", getattr(record, "doi", "?"), e)
        if assessments:
            assessment_path.write_text(
                orjson.dumps(assessments, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"),
                encoding="utf-8",
            )
            log.info("Assessment: %d papers assessed", len(assessments))
    except Exception as e:
        log.warning("Assessment failed: %s", e)

    return meta_analysis_path, assessment_path


def _kb_merge_and_enrich(
    papers: list,
    config: ResearchConfig,
    emit: Callable[[int, str], None],
) -> None:
    """Phase 1b: Merge local knowledge-base papers + enrich metadata.

    Searches the unified KB for papers matching the query, merges with
    API results (dedup by DOI), applies BGE cosine relevance filter,
    then enriches KB papers missing structured metadata (authors, year)
    via OpenAlex DOI lookup. Mutates ``papers`` in-place.
    """
    kb_new = 0
    try:
        get_kb = None  # KB integration not part of the standalone skill

        kb = get_kb()
        if kb.size > 0:
            kb_results = kb.search(config.query, max_results=10)
            existing_dois = {p.doi.lower() for p in papers if p.doi}

            kb_texts = []
            for kr in kb_results:
                if kr.get("doi") and kr["doi"].lower() in existing_dois:
                    continue
                text = f"{kr.get('title', '')} {kr.get('snippet', '')}"
                kb_texts.append((kr, text))

            if kb_texts:
                try:
                    import numpy as np
                    from _embeddings import embed_texts, is_available

                    if is_available():
                        query_emb = embed_texts([config.query])
                        paper_embs = embed_texts([t for _, t in kb_texts])
                        if query_emb is not None and paper_embs is not None:
                            q_norm = query_emb[0] / (np.linalg.norm(query_emb[0]) + 1e-8)
                            p_norms = paper_embs / (np.linalg.norm(paper_embs, axis=1, keepdims=True) + 1e-8)
                            cosines = p_norms @ q_norm
                            RELEVANCE_THRESHOLD = 0.70
                            from _sources import PaperRecord

                            for i, (kr, _) in enumerate(kb_texts):
                                if cosines[i] < RELEVANCE_THRESHOLD:
                                    log.debug(
                                        "KB filter: %s cosine=%.3f < %.2f — filtered",
                                        kr.get("doc", "?")[:30],
                                        cosines[i],
                                        RELEVANCE_THRESHOLD,
                                    )
                                    continue
                                doi = kr.get("doi", "")
                                title = kr.get("title", kr.get("doc", ""))
                                text = kb.get_text(kr.get("doc", ""))
                                rec = PaperRecord(
                                    doi=doi,
                                    title=title,
                                    abstract=text if text else "",
                                    authors=[],
                                    year=kr.get("year"),
                                    venue="",
                                    type="",
                                    source="knowledge_base",
                                    sources_seen=["knowledge_base"],
                                    discovery_provenance="kb:local",
                                )
                                papers.append(rec)
                                existing_dois.add(doi.lower())
                                kb_new += 1
                        else:
                            raise ImportError("embeddings unavailable")
                    else:
                        raise ImportError("embeddings not available")
                except ImportError:
                    log.warning(
                        "EMBEDDINGS UNAVAILABLE — knowledge-base results using "
                        "BM25F instead of BGE semantic similarity (lower quality). "
                        "Install: uv sync --extra rag"
                    )
                    from _sources import PaperRecord

                    # No semantic filter available — require lexical overlap
                    # with the query (plural-stripped) so place-name-only
                    # matches (malaria surveys naming a district) cannot enter.
                    _q_tokens = {
                        t[:-1] if t.endswith("s") and len(t) > 4 else t
                        for t in re.findall(r"[a-z]{4,}", config.query.lower())
                    }

                    def _has_overlap(kr: dict) -> bool:
                        hay = f"{kr.get('title', '')} {kr.get('snippet', '')}".lower()
                        toks = {t[:-1] if t.endswith("s") and len(t) > 4 else t for t in re.findall(r"[a-z]{4,}", hay)}
                        return bool(_q_tokens & toks)

                    for kr, _ in kb_texts:
                        if not kr.get("snippet"):
                            continue
                        if not _has_overlap(kr):
                            log.debug(
                                "KB fallback filter: no query-term overlap — %s",
                                kr.get("doc", "?")[:40],
                            )
                            continue
                        doi = kr.get("doi", "")
                        title = kr.get("title", kr.get("doc", ""))
                        text = kb.get_text(kr.get("doc", ""))
                        rec = PaperRecord(
                            doi=doi,
                            title=title,
                            abstract=text if text else "",
                            authors=[],
                            year=kr.get("year"),
                            venue="",
                            type="",
                            source="knowledge_base",
                            sources_seen=["knowledge_base"],
                        )
                        papers.append(rec)
                        existing_dois.add(doi.lower())
                        kb_new += 1

            if kb_new:
                log.info("KB: merged %d papers from local knowledge base (BGE-filtered)", kb_new)
    except Exception as e:
        log.warning("KB search skipped (local papers not included): %s", e)

    # KB metadata enrichment
    kb_metadata_missing = [
        p
        for p in papers
        if (p.abstract or "").strip()
        and p.doi
        and (not (p.authors or []) or not p.year)
        and "openalex" not in (p.sources_seen or [])
    ]
    if kb_metadata_missing:
        t_meta = time.time()
        emit(1, f"Enriching {len(kb_metadata_missing)} KB papers with OpenAlex metadata...")
        try:
            from _sources import openalex_get_by_doi

            meta_count = 0
            for i, p in enumerate(kb_metadata_missing):
                try:
                    record = openalex_get_by_doi(p.doi)
                    if record:
                        if not p.authors and record.authors:
                            p.authors = record.authors
                            meta_count += 1
                        if not p.year and record.year:
                            p.year = record.year
                            meta_count += 1
                except Exception as e:
                    log.debug("KB metadata enrichment failed for %s: %s", p.doi, e)
                if config.api_rate_limit_s > 0 and i < len(kb_metadata_missing) - 1:
                    time.sleep(config.api_rate_limit_s)
            log.info(
                "KB metadata enrichment: %d fields filled in %d/%d papers (%.1fs)",
                meta_count,
                sum(1 for p in kb_metadata_missing if p.authors or p.year),
                len(kb_metadata_missing),
                time.time() - t_meta,
            )
        except ImportError:
            log.debug("KB metadata enrichment skipped (_sources.openalex_get_by_doi unavailable)")


def _phase_extraction(
    config: ResearchConfig,
    results_dir: Path,
    verified_path: Path,
    emit: Callable[[int, str], None],
    t0: float,
) -> tuple[list[dict], Path, int, list]:
    """Phase 3: PICO + effect-size extraction + P-T plausibility filter.

    Returns (extractions, extracted_path, n_fulltext, verified_records).
    """
    _check_budget(t0, config.wall_clock_budget_s, 3)
    emit(3, "Extracting PICO + effect sizes...")
    t_phase = time.time()
    extracted_path = results_dir / "extracted.json"

    local_index: list[dict] = []
    n_fulltext = 0
    if config.match_local_pdfs:
        # Full-text candidate dirs (2026-09-01): the configured dir PLUS the
        # repo's curated references/papers library when present (dev
        # checkout) — 200+ domain-filed PDFs that were sitting unused while
        # every brief synthesized from abstracts alone.
        _ft_dirs = []
        if config.fulltext_dir:
            _ft_dirs.append(Path(config.fulltext_dir))
        _ref_lib = Path(__file__).resolve().parents[3] / "references" / "papers"
        if _ref_lib.is_dir():
            _ft_dirs.append(_ref_lib)
        local_index = []
        for _d in _ft_dirs:
            try:
                local_index.extend(build_local_index(_d, ocr_model=config.ocr_model))
            except Exception as exc:
                log.warning("local fulltext index failed for %s: %s", _d, exc)
    else:
        local_index = []

    from _sources import PaperRecord, load_corpus
    from extract import extract_from_paper

    verified_records = load_corpus(verified_path)
    extractions: list[dict] = []
    n_total = len(verified_records)

    # Phase 3a: Prepare records (full-text matching + abstract capping).
    prepared: list = []
    for record in verified_records:
        if local_index:
            local_path = match_paper_to_local(
                paper_doi=record.doi,
                paper_title=record.title,
                local_index=local_index,
            )
            if local_path:
                full_text = load_fulltext(local_path)
                if full_text and len(full_text) > len(record.abstract or ""):
                    record = PaperRecord.from_dict(record.to_dict())
                    record.full_text = full_text[: config.fulltext_cap]
                    n_fulltext += 1

        methods = getattr(record, "_methods_text", "")
        results_txt = getattr(record, "_results_text", "")
        discussion = getattr(record, "_discussion_text", "")
        conclusions = getattr(record, "_conclusions_text", "")
        if methods or results_txt or discussion or conclusions:
            ft_parts = [record.abstract or ""]
            if methods:
                ft_parts.append(f"\n\nMethods: {methods}")
            if results_txt:
                ft_parts.append(f"\n\nResults: {results_txt}")
            if discussion:
                ft_parts.append(f"\n\nDiscussion: {discussion}")
            if conclusions:
                ft_parts.append(f"\n\nConclusions: {conclusions}")
            record = PaperRecord.from_dict(record.to_dict())
            record.full_text = "".join(ft_parts)[: config.fulltext_cap]
            n_fulltext += 1

        if config.abstract_cap and record.abstract and len(record.abstract) > config.abstract_cap:
            record = PaperRecord.from_dict(record.to_dict())
            record.abstract = record.abstract[: config.abstract_cap]
        prepared.append(record)

    # ── Inject unmatched local PDFs as new corpus entries ──
    # If user has PDFs in fulltext_dir that DON'T match any discovered paper,
    # add them as new papers. This lets users provide their own corpus.
    if local_index:
        matched_dois = {r.doi.lower() for r in prepared if r.doi}
        matched_titles = {(r.title or "").lower()[:50] for r in prepared}
        n_injected = 0
        for entry in local_index:
            if entry.get("is_matched"):
                continue
            entry_doi = (entry.get("doi") or "").lower()
            entry_title = (entry.get("title") or "").lower()[:50]
            if entry_doi and entry_doi in matched_dois:
                continue  # already matched
            if entry_title and entry_title in matched_titles:
                continue
            # Unmatched local PDF/MD → inject as new paper, but ONLY if it
            # is a plausible on-topic paper. Unfiltered injection shipped
            # reference-library OCR dumps ("dare2014", journal header pages)
            # into every corpus regardless of query.
            ft = load_fulltext(entry["path"])
            # Strip HTML comments and metadata headers from text
            import re as _re_strip

            ft = _re_strip.sub(r"<!--.*?-->", "", ft, flags=_re_strip.DOTALL).strip()
            if ft and len(ft) > 200 and not ft.startswith("Quaternary"):
                # Skip non-paper files (journal volumes, metadata)
                title = entry.get("title") or entry["filename"]
                if title.startswith("<!--") or len(title) < 5:
                    title = entry["filename"]
                _title_lower = title.lower()
                _header_garbage = _re_strip.match(
                    r"^(?:vol\.|volume\b|pages?\b|no\.\s*\d|american "
                    r"mineralogist|economic geology|journal of|transactions)",
                    _title_lower,
                )
                _filename_stem = len(_re_strip.findall(r"\w+", title)) < 3 and not entry.get("doi")
                if _header_garbage or _filename_stem:
                    log.info(
                        "Local paper rejected (header/filename title): %s",
                        entry["filename"],
                    )
                    continue
                # Query relevance: title + opening text must match at least
                # one query content term (stem/synonym aware).
                try:
                    from _honesty import _term_matches, query_content_terms

                    _terms = query_content_terms(config.query)
                except ImportError:
                    _terms = []
                if _terms:
                    _hay = f"{title} {ft[:2000]}".lower()
                    if not any(_term_matches(t, _hay) for t in _terms):
                        log.info(
                            "Local paper rejected (no query-term match): %s",
                            entry["filename"],
                        )
                        continue
                injected = PaperRecord(
                    doi=entry.get("doi"),
                    title=title,
                    abstract=ft,
                    full_text=ft[: config.fulltext_cap],
                )
                prepared.append(injected)
                n_injected += 1
                log.info("Injected local paper: %s (%d chars)", entry["filename"], len(ft))
        if n_injected:
            emit(3, f"Injected {n_injected} unmatched local PDFs as corpus entries")
        n_total = len(prepared)

    # KB full-text enrichment — check KB for existing full text, download OA PDFs.
    ft_count = 0
    try:
        get_kb = None  # KB integration not part of the standalone skill

        kb = get_kb()

        for idx in range(min(config.max_pdf_downloads, len(prepared))):
            record = prepared[idx]
            if not record.doi:
                continue
            existing = kb.get_by_doi(record.doi)
            if existing and existing.has_fulltext:
                kb_text = kb.get_text(existing.doc_name)
                if kb_text and len(kb_text) > len(record.abstract or ""):
                    new_record = PaperRecord.from_dict(record.to_dict())
                    new_record.full_text = kb_text[: config.fulltext_cap]
                    prepared[idx] = new_record
                    ft_count += 1
                    log.info("KB: reused full text for %s (%d chars)", record.doi, len(kb_text))

        from _fulltext import enrich_paper

        for idx in range(min(config.max_pdf_downloads, len(prepared))):
            if ft_count >= config.max_pdf_downloads:
                break
            record = prepared[idx]
            if not record.doi:
                continue
            if kb.has_paper(doi=record.doi):
                continue

            paper_dict = {
                "doi": record.doi,
                "oa_pdf_url": getattr(record, "oa_pdf_url", None) or "",
                "title": record.title or "",
            }
            enrich_paper(paper_dict)
            if paper_dict.get("has_fulltext"):
                pdf_cache = Path.home() / ".cache" / "scientific_research" / "pdfs"
                from _fulltext import _doi_hash

                pdf_path = pdf_cache / f"{_doi_hash(record.doi)}.pdf"
                if pdf_path.exists():
                    kb.add_pdf(
                        pdf_path,
                        doi=record.doi,
                        title=record.title,
                        source="oa_download",
                    )

                ft_parts = [record.abstract or ""]
                if paper_dict.get("methods_text"):
                    ft_parts.append(f"\n\nMethods: {paper_dict['methods_text']}")
                if paper_dict.get("results_text"):
                    ft_parts.append(f"\n\nResults: {paper_dict['results_text']}")
                if paper_dict.get("discussion_text"):
                    ft_parts.append(f"\n\nDiscussion: {paper_dict['discussion_text']}")
                if paper_dict.get("conclusions_text"):
                    ft_parts.append(f"\n\nConclusions: {paper_dict['conclusions_text']}")
                new_record = PaperRecord.from_dict(record.to_dict())
                new_record.full_text = "".join(ft_parts)[: config.fulltext_cap]
                prepared[idx] = new_record
                ft_count += 1
                log.info(
                    "Full-text enriched: %s (%d chars full_text)",
                    record.doi,
                    len(new_record.full_text),
                )

        for record in prepared:
            if record.doi and not kb.has_paper(doi=record.doi):
                kb.add_abstract(
                    doi=record.doi,
                    title=record.title or "",
                    abstract=record.abstract or "",
                    source="api_discovery",
                )
    except Exception as e:
        log.warning("KB full-text enrichment failed (KB PDFs not processed): %s", e)

    # Phase 3b: Batch LLM extraction.
    llm_elapsed = 0.0
    if config.use_llm and config.llm_batch_size > 0 and prepared:
        try:
            from _llm_extract import extract_papers_batch

            papers_input = [{"abstract": r.abstract or "", "title": r.title or ""} for r in prepared]
            emit(
                3,
                f"Batch extracting {len(papers_input)} papers (batch_size={config.llm_batch_size})…",
            )
            t_batch = time.time()
            extract_papers_batch(
                papers_input,
                topic=config.query,
                batch_size=config.llm_batch_size,
            )
            llm_elapsed = time.time() - t_batch
            log.info(
                "Phase 3 batch pre-population: %d papers in %.1fs (%.1fs/batch of %d)",
                len(papers_input),
                llm_elapsed,
                llm_elapsed,
                config.llm_batch_size,
            )
        except Exception as e:
            log.warning("Batch extraction pre-population failed: %s — per-paper fallback", e)

    # Phase 3c: Ollama-backed extraction (preferred when LLM enabled).
    # Runs when synthesis will use Ollama (Tier 1) — needs structured data.
    # NOT gated on config.use_llm because PaperQA2 detection (line 1812)
    # temporarily sets use_llm=False. Gate on synthesis_tier instead.
    ollama_extractions: dict[int, dict] = {}
    if config.synthesis_tier in ("auto", "ollama") or config.use_llm:
        try:
            from _ollama_extract import extract_paper_ollama, to_extraction_result

            emit(3, f"Ollama zero-shot extraction: {n_total} papers...")

            # Parallel extraction — Ollama handles concurrent HTTP requests.
            # max_workers=3 balances throughput vs Ollama's internal queueing.
            import concurrent.futures

            def _extract_one(idx: int, rec) -> tuple[int, dict | None]:
                paper_dict = {
                    "title": rec.title or "",
                    "abstract": rec.abstract or rec.full_text or "",
                    "doi": rec.doi or "",
                }
                try:
                    result = extract_paper_ollama(paper_dict, query=config.query)
                    if result.get("classification") or result.get("measurements"):
                        return idx, to_extraction_result(paper_dict, result)
                except Exception as e:
                    log.debug("  [%d/%d] Ollama failed: %s", idx + 1, n_total, e)
                return idx, None

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                futures = {pool.submit(_extract_one, i, rec): i for i, rec in enumerate(prepared)}
                done_count = 0
                for future in concurrent.futures.as_completed(futures):
                    idx, extraction = future.result()
                    done_count += 1
                    if extraction is not None:
                        ollama_extractions[idx] = extraction
                    emit(3, f"Ollama extracted {done_count}/{n_total}...")
        except ImportError:
            log.debug("_ollama_extract not available — using custom extraction")
        except Exception as e:
            log.warning("Ollama extraction batch failed: %s — falling back to custom", e)

    # Phase 3d: Per-paper extraction loop (fallback for papers not extracted via Ollama).
    llm_budget = config.llm_time_budget_s
    llm_budget_exhausted = False
    if llm_elapsed and llm_budget is not None and llm_elapsed >= llm_budget:
        llm_budget_exhausted = True
        log.warning(
            "LLM budget exhausted during batch pre-population (%.1fs/%.1fs) — per-paper loop uses non-LLM path",
            llm_elapsed,
            llm_budget,
        )
    for i, record in enumerate(prepared):
        # Skip papers already extracted via Ollama
        if i in ollama_extractions:
            extractions.append(ollama_extractions[i])
            continue

        effective_use_llm = config.use_llm and not llm_budget_exhausted

        try:
            if effective_use_llm:
                emit(3, f"Extracting {i + 1}/{n_total} (LLM)…")
            else:
                emit(3, f"Extracting {i + 1}/{n_total}…")
            t_one = time.time()
            result = extract_from_paper(record, topic=config.query, use_llm=effective_use_llm)
            if effective_use_llm:
                if config.ollama_delay_s > 0 and i < n_total - 1:
                    time.sleep(config.ollama_delay_s)
                if not llm_elapsed:
                    llm_elapsed += time.time() - t_one
            if effective_use_llm and llm_budget is not None and llm_elapsed >= llm_budget:
                llm_budget_exhausted = True
                log.warning(
                    "LLM budget exhausted after %d/%d papers (%.1fs/%.1fs) — remaining papers use non-LLM path",
                    i + 1,
                    n_total,
                    llm_elapsed,
                    llm_budget,
                )
                emit(
                    3,
                    f"LLM budget exhausted ({llm_elapsed:.0f}s) — non-LLM for remaining {n_total - i - 1} papers",
                )
            extractions.append(result.to_dict())
            log.debug(
                "  [%d/%d] extracted: %s (disc=%s)",
                i + 1,
                len(prepared),
                record.title[:40] if record.title else "?",
                result.pico.get("discipline", "?"),
            )
        except Exception as e:
            log.warning(
                "  [%d/%d] extraction FAILED for '%s': %s",
                i + 1,
                len(prepared),
                (record.title or "?")[:40],
                e,
            )

    # P-T plausibility filter — flag implausible values before synthesis.
    try:
        from _geo_enrich import filter_pt_values

        disc_counts: dict[str, int] = {}
        for extr in extractions:
            pico = extr.get("pico") or {}
            d = (pico.get("discipline") or "").lower()
            if d:
                disc_counts[d] = disc_counts.get(d, 0) + 1
        dominant_disc = max(disc_counts, key=disc_counts.get) if disc_counts else ""
        extractions, pt_warnings = filter_pt_values(extractions, dominant_disc)
        for w in pt_warnings:
            log.warning("P-T filter: %s", w)
    except ImportError:
        log.debug("Optional module unavailable", exc_info=True)

    extracted_data = {
        "meta": {
            "query": config.query,
            "n_papers": len(extractions),
            "n_fulltext": n_fulltext,
            "use_llm": config.use_llm,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
        "extractions": extractions,
    }
    extracted_path.write_text(
        orjson.dumps(extracted_data, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"),
        encoding="utf-8",
    )
    log.info(
        "Phase 3 timing: %.1fs (%d extracted, %d full-text, %d failed%s)",
        time.time() - t_phase,
        len(extractions),
        n_fulltext,
        len(verified_records) - len(extractions),
        f", llm={llm_elapsed:.1f}s" + (" [budget exhausted]" if llm_budget_exhausted else "") if config.use_llm else "",
    )
    emit(3, f"Extraction complete: {len(extractions)} papers ({n_fulltext} full-text)")
    return extractions, extracted_path, n_fulltext, verified_records


def _zero_match_repair(
    papers: list,
    qs: QueryState,
    *,
    config: ResearchConfig,
    raw_query: str,
    strategy_queries: list[str],
    effective_sources: list[str],
    search_filters: dict,
    over_fetch: int,
    research_ctx,
    emit,
) -> tuple[list, QueryState]:
    """Kept-pool repair contract: (papers, qs) -> (papers, qs).

    A distinctive query term (>=8 chars) matching NO paper in the KEPT
    pool triggers web-vocabulary typo repair + one corrected
    rediscovery round whose results pass the SAME term-coverage gate.
    Extracted 2026-09-03 — operates ONLY on the post-ranking pool
    (raw-pool mentions the ranker is about to drop must not certify
    the corpus).
    """
    # ── Zero-match term repair (2026-09-03, dictionary-free) ──────────
    # A distinctive query term (>=8 chars) matching NO paper in the KEPT
    # (post-ranking) pool, even fuzzily, is either out-of-corpus or
    # TYPO'D — and no offline dictionary can repair a typo it has never
    # seen (live case: 'prahnita godawari sedimentary chandrapur' —
    # geodict + 231 local-library titles all lack the basin name, so
    # every API searched the typo verbatim and returned off-topic
    # candidates). The live web IS the dictionary: search engines
    # fuzzy-correct typos in their RESULTS, so one text search on the
    # raw query yields a vocabulary ('Pranhita-Godavari Basin') from
    # which Damerau-Levenshtein<=1 unique-candidate repair recovers the
    # intended term. One bounded retry round then re-searches with the
    # corrected tokens and passes them through the SAME term-coverage
    # gate — no quality loosening. Runs on the KEPT pool, not the raw
    # dedup pool: passing mentions in abstract-less papers the ranker
    # is about to drop must not certify the corpus (live repro: gate
    # stayed silent on a 49-paper raw pool whose only prahnita mention
    # lived in a paper the ranker discarded; screening then saw 20 with
    # zero pranhita). Fail-open everywhere; every applied fix is loud.
    try:
        from _honesty import (
            _term_matches,
        )
        from _honesty import (
            filter_by_term_coverage as _ftc,
        )
        from _honesty import (
            query_content_terms as _qct,
        )
        from _query import spellcorrect_query
        from _sources import dedup_papers
        from discover import search_multi_source

        _audit_terms = _qct(qs.audit)
        if papers and _audit_terms:
            _hays = [f"{getattr(p, 'title', '') or ''} {getattr(p, 'abstract', '') or ''}".lower() for p in papers]
            _zero = [t for t in _audit_terms if len(t) >= 8 and not any(_term_matches(t, h) for h in _hays)]
            if _zero:
                emit(1, f"Term(s) match no candidate: {', '.join(_zero)} — live-web typo repair...")
                log.info("Zero-match distinctive terms %s — web-vocab repair", _zero)
                _web_vocab: list[str] = []
                try:
                    from _websearch import search_text as _engine_text

                    _rc, _data = _engine_text(
                        raw_query,
                        max_results=15,
                        n_extract=0,
                        call_budget=40,
                        quiet=True,
                    )
                    if _rc == 0:
                        _web_vocab = [f"{_r.get('title', '')} {_r.get('body', '')}" for _r in _data.get("results", [])]
                except Exception as _ex:
                    log.warning("Web-vocab repair probe failed (fail-open): %s", _ex)
                _fixes: dict[str, str] = {}
                if _web_vocab:
                    _, _rep_notes = spellcorrect_query(qs.audit, extra_vocab=_web_vocab)
                    for _note in _rep_notes:
                        _orig, _, _fixed = _note.partition(" -> ")
                        if _orig in _zero:
                            _fixes[_orig] = _fixed
                if _fixes:
                    _fix_txt = "; ".join(f"{o} → {f}" for o, f in sorted(_fixes.items()))
                    emit(1, f"Query typo repaired via web vocabulary: {_fix_txt}")
                    log.info("Web-vocab typo repair applied: %s", _fix_txt)

                    def _sub_tokens(qstr: str) -> str:
                        out = qstr
                        for _o, _f in _fixes.items():
                            out = re.sub(rf"\b{re.escape(_o)}\b", _f, out, flags=re.IGNORECASE)
                        return out

                    _qs_notes = [f"{o} → {f} (web vocab)" for o, f in sorted(_fixes.items())]
                    qs = qs.with_token_fixes(_fixes, notes=_qs_notes)
                    search_filters["query"] = qs.search
                    _round2_raw: list = []
                    for sq in dict.fromkeys([_sub_tokens(s) for s in strategy_queries] + [qs.audit]):
                        _round2_raw.extend(
                            _r2
                            for _r2 in search_multi_source(
                                sq,
                                max_per_source=max(3, over_fetch // 3 + 2),
                                sources=effective_sources,
                                per_source_timeout=config.search_timeout,
                                filters=search_filters or None,
                                original_query=qs.audit,
                                domain_hint=getattr(research_ctx, "domain", "") or "",
                                use_web_search=(config.use_web_search or "web_search" in effective_sources),
                                use_web_search_agentic=config.use_web_search_agentic,
                            )
                        )
                        for _r2 in _round2_raw:
                            _r2.discovery_provenance = "discovery:typo-repair-rediscovery"
                    _kept2, _ = _ftc(
                        _round2_raw,
                        qs.audit,
                        alias_map=getattr(research_ctx, "alias_map", None),
                    )
                    if _kept2:
                        papers = dedup_papers(list(papers) + _kept2)
                        log.info(
                            "Corrected rediscovery: +%d on-topic papers (kept pool now %d)",
                            len(_kept2),
                            len(papers),
                        )
                else:
                    # RECALL ROUND (2026-09-03, live gadchiroli case): the
                    # zero-match term is CORRECTLY spelled (web vocab found
                    # no fix) — the literature may exist yet never have been
                    # retrieved (Wairagarh/Gondpipri dyke papers live in
                    # OpenAlex but missed the candidate pool for a narrow
                    # district query). One domain-anchored search per
                    # unmatched term, through the SAME term gate.
                    _recall_raw: list = []
                    # Query construction (live-tested): zero term + the
                    # query's SUBJECT term + the mapped domain anchor —
                    # 'gadchiroli dyke geochemistry' surfaces the Wairagarh
                    # dykes; domain-only ('gadchiroli igneous') pulls district
                    # junk (police-stress, avifaunal studies).
                    try:
                        from _context import _DOMAIN_TERMS as _DT
                    except ImportError:
                        _DT = {}
                    _dom_anchor = (
                        _DT.get(getattr(research_ctx, "domain", "") or "", "")
                        or getattr(research_ctx, "domain", "")
                        or ""
                    )
                    _subject = next((u for u in _audit_terms if u not in _zero), "")
                    for t in _zero[:3]:
                        _rq = " ".join(x for x in (t, _subject, _dom_anchor) if x)
                        _recall_raw.extend(
                            _rr
                            for _rr in search_multi_source(
                                _rq,
                                max_per_source=max(3, over_fetch // 4 + 2),
                                sources=effective_sources,
                                per_source_timeout=config.search_timeout,
                                filters=search_filters or None,
                                original_query=qs.audit,
                                domain_hint=getattr(research_ctx, "domain", "") or "",
                                use_web_search=(config.use_web_search or "web_search" in effective_sources),
                                use_web_search_agentic=config.use_web_search_agentic,
                            )
                        )
                    for _rr in _recall_raw:
                        _rr.discovery_provenance = "discovery:zero-term-recall"
                    # Recall gate: the zero terms + subject pair, NOT the
                    # full audit query — we are recalling papers ABOUT the
                    # unmatched term; generic sibling-subject papers that
                    # match only the subject stay out.
                    _recall_query = " ".join(x for x in [*_zero[:3], _subject] if x)
                    _kept3, _ = _ftc(
                        _recall_raw,
                        _recall_query,
                        alias_map=getattr(research_ctx, "alias_map", None),
                    )
                    if _kept3:
                        papers = dedup_papers(list(papers) + _kept3)
                        emit(
                            1,
                            f"Zero-term recall: +{len(_kept3)} paper(s) for {', '.join(_zero[:3])}",
                        )
                        log.info(
                            "Zero-term recall round: +%d papers (kept pool now %d)",
                            len(_kept3),
                            len(papers),
                        )
    except Exception as e:
        log.warning("Zero-match term repair skipped (fail-open): %s", e)
    return papers, qs


def _rank_and_trim(deduped: list, search_str: str, max_papers: int) -> list:
    """Kept-pool contract: enrich abstracts, BGE-rerank, trim to max.

    Defines THE candidate pool every downstream consumer (screening,
    zero-match repair, honesty gates) sees. Abstract-less papers are
    discarded ONLY when enough abstracted papers exist.
    """
    # ── Abstract enrichment BEFORE ranking ──
    # Crossref rarely returns abstracts; OpenAlex/S2 must fill them.
    # Must run before ranking so the abstract-based filter works correctly.
    abstractless_pre = [p for p in deduped if not (p.abstract or "").strip() and p.doi]
    if abstractless_pre:
        log.info("Pre-ranking: enriching %d abstracts from OpenAlex...", len(abstractless_pre))
        try:
            from _sources import openalex_get_by_doi

            n_enriched = 0
            for p in abstractless_pre:
                try:
                    rec = openalex_get_by_doi(p.doi)
                    if rec and (rec.abstract or "").strip():
                        p.abstract = rec.abstract
                        n_enriched += 1
                except Exception:
                    log.debug("openalex enrichment failed for %s", p.doi, exc_info=True)
            if n_enriched:
                log.info(
                    "Pre-ranking: filled %d/%d abstracts from OpenAlex",
                    n_enriched,
                    len(abstractless_pre),
                )
        except ImportError:
            log.debug("openalex_get_by_doi unavailable for pre-ranking enrichment")

    # ── BGE semantic reranking ──
    ranking_method = "tf-idf"
    try:
        from _embeddings import is_available as embeddings_available

        if embeddings_available():
            ranking_method = "BGE"
    except ImportError:
        log.debug("Optional module unavailable", exc_info=True)

    try:
        from _ranking import semantic_relevance_scores

        scores = semantic_relevance_scores(search_str, deduped)
        ranked = sorted(zip(deduped, scores), key=lambda x: -x[1])

        with_abstract = [p for p, _ in ranked if (p.abstract or "").strip()]
        without_abstract = [p for p, _ in ranked if not (p.abstract or "").strip()]

        if len(with_abstract) >= max_papers:
            papers = with_abstract[:max_papers]
            log.info(
                "Discarded %d papers without abstracts, kept %d with abstracts",
                len(without_abstract),
                len(papers),
            )
        elif len(with_abstract) > 0:
            papers = with_abstract + without_abstract[: max_papers - len(with_abstract)]
            log.info(
                "Using %d papers with abstracts + %d without (fill gap)",
                len(with_abstract),
                len(papers) - len(with_abstract),
            )
        else:
            papers = [p for p, _ in ranked[:max_papers]]
            log.warning("No papers have abstracts — using title-based fallback for all")

        log.info(
            "Ranked %d papers by %s semantic relevance (top score=%.3f)",
            len(papers),
            ranking_method.upper(),
            scores.max() if len(scores) > 0 else 0,
        )
    except Exception as e:
        log.warning("Ranking failed (%s), using arbitrary order", e)
        papers = deduped[:max_papers]

    return papers


def _phase_discovery(
    config: ResearchConfig,
    results_dir: Path,
    emit: Callable[[int, str], None],
) -> tuple[list, str, str]:
    """Phase 1: Multi-source discovery + query normalization + screening + ranking.

    Returns (papers, search_query, rtype).
    """
    emit(1, f"Searching: {config.query[:60]}...")

    # Detect research type — needed by LLM screening prompt selection.
    try:
        from _classifiers import detect_research_type

        rtype = config.research_type_override or detect_research_type(config.query)
    except Exception:
        rtype = config.research_type_override or "survey"

    from _sources import dedup_papers
    from discover import search_multi_source

    # ── Query normalization ──
    # search_query: retrieval string (synonym-expanded — max recall).
    # audit_query: the USER's terms, typo-repaired only. Honesty gates
    # (sufficiency veto, coverage, grounding) audit audit_query — auditing
    # the expanded string demanded corpus support for normalizer-INJECTED
    # synonyms ('Sedimentary rock' → 'rock', BGE '+geochemistry') that the
    # user never asked about and refused honest corpora (live 2026-09-02).
    search_query = config.query
    audit_query = config.query
    spell_fix_notes: list[str] = []
    try:
        from _query import normalize_query

        search_query, corrections, ambiguities = normalize_query(
            config.query,
            use_ollama=config.use_llm,
            ollama_model=config.llm_model or "",
        )
        if corrections:
            emit(1, f"Query corrected: {'; '.join(corrections)}")
        if ambiguities:
            emit(
                1,
                f"Note: ambiguous term(s): {', '.join(t for t, _ in ambiguities)}",
            )
    except ImportError:
        log.warning("Query normalization module not available — no spelling correction or abbreviation expansion")

    if config.use_llm:
        try:
            from _llm_extract import llm_normalize_query

            llm_corrected = llm_normalize_query(search_query, model=config.llm_model or None)
            if llm_corrected != search_query:
                emit(1, f"LLM query fix: {search_query!r} → {llm_corrected!r}")
                search_query = llm_corrected
        except Exception as e:
            log.debug("LLM query normalization skipped: %s", e)

    # Offline spell-correct (2026-09-02): Damerau-Levenshtein <=1 against a
    # dynamic vocabulary (geodict + curated canonical terms). Covers the
    # LLM-down path where typos previously reached retrieval AND the honesty
    # gates verbatim (live case: prahnita/godawari/geologyl). Applied to the
    # ORIGINAL (→ audit_query, user-facing notes) and to the expanded
    # retrieval string (→ search_query).
    try:
        from _query import QueryState, spellcorrect_query

        # UNIVERSAL TYPO CORRECTION, stage 1: the project references index
        # (200+ domain papers) carries the proper nouns geodict lacks —
        # basin/district names. Absent file degrades loudly to geodict-only.
        _lib_vocab: list[str] = []
        try:
            import json as _json_lib

            _idx = Path(__file__).resolve().parents[3] / "references" / "INDEX.json"
            if _idx.exists():

                def _titles(node):
                    if isinstance(node, dict):
                        for k, v in node.items():
                            if k.lower() == "title" and isinstance(v, str):
                                yield v
                            else:
                                yield from _titles(v)
                    elif isinstance(node, list):
                        for it in node:
                            yield from _titles(it)

                _lib_vocab = list(_titles(_json_lib.loads(_idx.read_text(encoding="utf-8"))))
                log.info("Spell-correct vocab: %d reference titles", len(_lib_vocab))
        except Exception as e:
            log.warning("references INDEX vocab unavailable: %s", e)

        audit_query, spell_fix_notes = spellcorrect_query(config.query, extra_vocab=_lib_vocab)
        search_query, _search_spell_notes = spellcorrect_query(search_query)
        if spell_fix_notes:
            emit(1, f"Query spell-corrected: {'; '.join(spell_fix_notes)}")
    except Exception as e:
        log.warning("Offline spell-correct unavailable: %s", e)

    # ── QueryState: THE query object from here on (2026-09-03) ─────────
    # One frozen object crossing every phase boundary; no layer ever
    # re-reads config.query for matching. audit/search locals above were
    # its construction inputs only.
    qs = QueryState(
        raw=config.query,
        audit=audit_query,
        search=search_query,
        fixes=tuple(spell_fix_notes),
    )

    # ── Phase 0: research context builder ─────────────────────────────
    # Decompose the query BEFORE searching: entities + aliases + intent +
    # niche detection drive per-entity search strategies (no dilution),
    # entity-aware screening, and hybrid web search for locality-anchored
    # queries. Built from the TYPO-REPAIRED qs.audit (2026-09-03) —
    # building from raw config.query sent every search strategy and alias
    # lookup out with the typos the normalizer had just fixed.
    # Fail-open: morphology-only context when offline.
    research_ctx = None
    try:
        from _context import build_research_context

        research_ctx = build_research_context(qs.audit)
        emit(
            1,
            f"Context: {research_ctx.intent}"
            + (f"/{research_ctx.domain}" if research_ctx.domain else "")
            + (" [niche: hybrid web]" if research_ctx.niche else "")
            + f" — {len(research_ctx.entities)} entities, "
            + f"{len(research_ctx.search_strategies)} search strategies",
        )
        # Persist the context for auditability (step audit contract)
        try:
            (results_dir / "research_context.json").write_text(
                orjson.dumps(research_ctx.to_dict(), option=orjson.OPT_INDENT_2).decode(),
                encoding="utf-8",
            )
        except Exception:
            log.debug("research_context.json write failed", exc_info=True)
    except Exception as e:
        log.warning("Context builder unavailable (fail-open): %s", e)
    t_phase = time.time()

    # ── Multi-source search ──
    search_filters: dict = {}
    if config.year_from or config.year_to:
        search_filters["year_from"] = config.year_from
        search_filters["year_to"] = config.year_to
    if config.open_access_only:
        search_filters["open_access_only"] = True
    if config.publication_type:
        search_filters["publication_type"] = config.publication_type
    search_filters["query"] = qs.search
    over_fetch = max(5, int(config.max_papers * 3 // len(config.sources)))
    # Context-driven search: ORIGINAL query first (normalization sibling
    # expansion dilutes primary subjects — pyrite evidence), then per-entity
    # strategies for coordinated/niche queries. Niche (locality-anchored)
    # queries additionally get web search UPFRONT — the literature lives in
    # regional journals the academic APIs' keyword relevance misses.
    effective_sources = list(config.sources)
    if (
        research_ctx is not None
        and research_ctx.niche
        and not config.use_web_search
        and "web_search" not in effective_sources
    ):
        effective_sources.append("web_search")
        emit(1, "Niche query detected — web search enabled upfront")

    raw_papers: list = []
    strategy_queries = (
        [research_ctx.search_strategies[0]]
        if research_ctx is not None and research_ctx.search_strategies
        else [qs.search]
    )
    if research_ctx is not None:
        strategy_queries = research_ctx.search_strategies or [qs.search]
    for i, sq in enumerate(strategy_queries):
        # All strategies keep the full source set incl. web_search: the
        # per-entity strategies target literature the academic APIs miss
        # (Gondpipri/Wairagarh papers resolved only via ddgs).
        srcs = effective_sources
        got = search_multi_source(
            sq,
            max_per_source=max(3, over_fetch // len(strategy_queries) + 2),
            sources=srcs,
            per_source_timeout=config.search_timeout,
            filters=search_filters or None,
            original_query=config.query,
            # Dynamic scholarly anchor for web queries (2026-09-03): the
            # detected research domain stops place-name collisions from
            # pulling off-domain papers into the candidate pool.
            domain_hint=getattr(research_ctx, "domain", "") or "",
            # 2026-09-01 FIX: the GUI source list never contains "web_search"
            # (it rides the use_web_search checkbox flag) — deriving the flag
            # from srcs alone silently disabled websearch for EVERY narrowed
            # or GUI-built run. Respect the config flag; srcs still force-add.
            use_web_search=(config.use_web_search or "web_search" in srcs),
            use_web_search_agentic=config.use_web_search_agentic,
        )
        for _r in got:
            _r.discovery_provenance = _r.discovery_provenance or "discovery:primary"
        raw_papers.extend(got)
        if len(raw_papers) >= config.max_papers * 3:
            break
    deduped = dedup_papers(raw_papers)
    n_after_dedup = len(deduped)

    # ── Auto web-search supplement (sparse-corpus safety net) ──
    if (
        config.auto_web_search_threshold > 0
        and not config.use_web_search
        and not config.use_web_search_agentic
        and n_after_dedup < config.auto_web_search_threshold
    ):
        emit(1, f"Sparse corpus ({n_after_dedup}) — supplementing via web search...")
        log.info(
            "Sparse post-dedup (%d < threshold %d) — auto-supplementing via web_search",
            n_after_dedup,
            config.auto_web_search_threshold,
        )
        try:
            from discover import web_search_paper_discovery

            supplement = web_search_paper_discovery(
                qs.audit,
                max_results=max(10, config.auto_web_search_threshold * 2),
            )
            if supplement:
                for _r in supplement:
                    _r.discovery_provenance = "supplement:web-sparse"
                log.info("Auto-supplement: +%d papers from web_search", len(supplement))
                deduped = dedup_papers(deduped + supplement)
        except Exception as ex:
            log.warning("Auto-supplement via web_search failed: %s", ex)
        n_after_dedup = len(deduped)

    # ── Agentic escalation (one-shot): text-mode web search came up empty ──
    # When websearch ran but contributed ZERO papers and the corpus is still
    # sparse, the literature likely lives in JS-rendered/repository pages the
    # text tier misses — escalate once to the adaptive crawler (30-120s).
    if (
        config.use_web_search
        and not config.use_web_search_agentic
        and n_after_dedup < config.max_papers
        and sum(1 for p in deduped if getattr(p, "source", "") == "web_search") < 3
    ):
        emit(1, "Web text search found few papers — escalating to deep crawl...")
        log.info("web_search text tier yielded 0 papers — one-shot agentic escalation")
        try:
            from discover import web_search_agentic_discovery

            agentic_papers = web_search_agentic_discovery(qs.audit, max_results=min(config.max_papers, 10))
            if agentic_papers:
                for _r in agentic_papers:
                    _r.discovery_provenance = "escalation:agentic-crawl"
                log.info("Agentic escalation: +%d papers", len(agentic_papers))
                deduped = dedup_papers(deduped + agentic_papers)
                n_after_dedup = len(deduped)
        except Exception as ex:
            log.warning("Agentic escalation failed: %s", ex)

    # ── Intent-aware filter ──
    try:
        from _intent import intent_filter, parse_intent

        intent = parse_intent(config.query)
        if intent.has_template:
            before_intent = len(deduped)
            deduped = intent_filter(deduped, intent)
            if before_intent != len(deduped):
                log.info(
                    "Intent filter: %d -> %d (removed %d peripheral)",
                    before_intent,
                    len(deduped),
                    before_intent - len(deduped),
                )
    except Exception as e:
        log.warning("Intent filter skipped (quality reduced): %s", e)

    # ── Domain relevance screening lives in _run_pipeline_impl (after KB
    # merge) so local-knowledge-base papers pass the SAME gate as API
    # papers. Paper callbacks fire there too — only screened papers reach
    # the GUI live feed.

    # ── Post-discovery filters (year/OA/type enforcement) ──
    if config.year_from or config.year_to:
        before = len(deduped)
        deduped = [
            p
            for p in deduped
            if (not config.year_from or (p.year or 0) >= config.year_from)
            and (not config.year_to or (p.year or 9999) <= config.year_to)
        ]
        if before != len(deduped):
            log.info("Year filter: %d → %d papers", before, len(deduped))
    if config.open_access_only:
        deduped = [p for p in deduped if p.is_open_access]
    if config.publication_type:
        deduped = [p for p in deduped if config.publication_type in (p.type or "")]

    papers = _rank_and_trim(deduped, qs.search, config.max_papers)

    log.info(
        "Phase 1 timing: %.1fs (raw=%d, deduped=%d, kept=%d)",
        time.time() - t_phase,
        len(raw_papers),
        len(deduped),
        len(papers),
    )
    log.info(
        "Discovery funnel: raw=%d → dedup=%d → filtered=%d → ranked=%d (%d with abstracts)",
        len(raw_papers),
        n_after_dedup,
        len(deduped),
        len(papers),
        sum(1 for p in papers if (p.abstract or "").strip()),
    )

    papers, qs = _zero_match_repair(
        papers,
        qs,
        config=config,
        raw_query=config.query,
        strategy_queries=strategy_queries,
        effective_sources=effective_sources,
        search_filters=search_filters,
        over_fetch=over_fetch,
        research_ctx=research_ctx,
        emit=emit,
    )

    return papers, qs, rtype, research_ctx, spell_fix_notes


def _run_pipeline_impl(
    config: ResearchConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    """Pipeline implementation (called under _pipeline_lock)."""
    t0 = time.time()
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)
    # Retention (2026-09-03): per-run FULL LLM call journal — every prompt
    # and response preserved with metadata for future replay/audit.
    try:
        from _llm_extract import set_llm_journal

        set_llm_journal(results_dir / "llm_calls.jsonl")
    except Exception:
        log.debug("LLM journal unavailable", exc_info=True)

    log.info(
        "=== PIPELINE START === query='%s' sources=%s max=%d llm=%s local_pdfs=%s skip_verify=%s skip_correlate=%s",
        config.query[:80],
        config.sources,
        config.max_papers,
        config.use_llm,
        config.match_local_pdfs,
        config.skip_verify,
        config.skip_correlate,
    )

    # Apply LLM model + parameter overrides
    if config.llm_model:
        from _llm_extract import set_model_override

        set_model_override(config.llm_model)
    # Always set LLM params (takes effect when use_llm=True)
    try:
        from _llm_extract import set_llm_params, set_quality_policy

        set_llm_params(config.llm_num_ctx, config.llm_num_predict)
        set_quality_policy(config.llm_quality)
    except ImportError:
        log.debug("Optional module unavailable", exc_info=True)

    def _emit(phase: int, msg: str) -> None:
        log.info("[Phase %d/5] %s", phase, msg)
        if progress:
            progress(phase, msg)

    # ── Resolve LLM model from config (auto-detect if empty) ──
    if config.use_llm and not config.llm_model:
        try:
            import json as _mj
            import urllib.request as _mu

            _tags = _mj.loads(_mu.urlopen("http://127.0.0.1:11434/api/tags", timeout=3).read())
            _names = [m.get("name", "") for m in _tags.get("models", [])]
            # Exclude non-chat models (embedding, OCR)
            _chat = [n for n in _names if not any(k in n.lower() for k in ("embed", "ocr", "whisper", "tts"))]
            if _chat:
                config.llm_model = _chat[0]
                log.info("Auto-detected Ollama model: %s", config.llm_model)
        except Exception:
            log.warning("Could not auto-detect Ollama model — LLM features may fail")

    # Skip S2 entirely if user didn't select it as a source
    if "s2" not in config.sources:
        try:
            from _sources import force_skip_s2

            force_skip_s2()
            log.info("S2 skipped — not in user-selected sources")
        except ImportError:
            log.debug("Optional module unavailable", exc_info=True)

    # Fresh rate-limit registry for this pipeline run — circuit breakers
    # start closed, pacing counters at zero. A transient failure in a
    # prior topic CANNOT cascade (run-scoped state, not module globals).
    from _ratelimits import RateLimitRegistry, set_registry

    set_registry(RateLimitRegistry())

    from _sources import save_corpus

    # ── Phase 1: DISCOVERY ──────────────────────────────────────────────
    papers, qs, rtype, research_ctx, spell_fix_notes = _phase_discovery(config, results_dir, _emit)

    # ── Phase 1b: Local Knowledge Base search + metadata enrichment ─────
    _kb_merge_and_enrich(papers, config, _emit)

    n_discovered = len(papers)
    screening_log: list[dict] = []

    # ── Phase 1c-i: Deterministic term-coverage pre-filter ──────────────
    # General recall gate (no LLM): a paper must match >=2 DISTINCT query
    # content terms. Kills place-name-only matches — a malaria survey that
    # merely names the district is excluded; a dyke study OF that district
    # matches two terms and survives. Applies to every source uniformly.
    try:
        from _honesty import filter_by_term_coverage

        # Alias map comes from Phase 0 context (built once, auditable in
        # research_context.json). No per-run re-fetch here.
        alias_map: dict[str, list[str]] = getattr(research_ctx, "alias_map", {}) or {}
        if alias_map:
            log.info("Context alias map: %d terms with aliases", len(alias_map))

        # UNIVERSAL TYPO CORRECTION, stage 2 (Master directive 2026-09-03):
        # correct the screening query against the DISCOVERED CANDIDATES —
        # the most query-relevant vocabulary that exists. A typo'd proper
        # noun ('prahnita' -> 'pranhita' inside a candidate title) is
        # repaired BEFORE it can veto the corpus; every fix is loud.
        screen_query = config.query
        try:
            from _query import spellcorrect_query

            _cand_vocab = [f"{getattr(p, 'title', '') or ''}" for p in papers]
            screen_query, _screen_fixes = spellcorrect_query(config.query, extra_vocab=_cand_vocab)
            if _screen_fixes:
                _emit(
                    1,
                    "Screening query typo-fixed vs candidates: " + "; ".join(_screen_fixes),
                )
        except Exception as e:
            log.warning("Candidate-vocab spell-correct unavailable: %s", e)
        before_terms = len(papers)
        papers, term_excluded = filter_by_term_coverage(papers, screen_query, alias_map=alias_map)
        if term_excluded:
            log.info(
                "Term-coverage filter: %d → %d papers (excluded %d single-term matches)",
                before_terms,
                len(papers),
                len(term_excluded),
            )
            screening_log.extend(term_excluded)

        # ── Relevance-emptiness web rescue ──────────────────────────────
        # Academic APIs can return ONLY off-topic candidates for niche/
        # regional queries (Indian district dyke studies live in GSI/
        # Springer journals the top-N API results miss). When the relevance
        # gate zeroes the corpus and web search hasn't run yet, escalate:
        # fetch web-discovered papers and pass them through the SAME
        # term-coverage gate — they earn their place, no quality loosening.
        if not papers and not config.use_web_search and not config.use_web_search_agentic:
            _emit(1, "Academic sources off-target — escalating to web search...")
            log.info("Relevance-empty corpus — web-search rescue for %r", qs.audit)
            try:
                from _honesty import filter_by_term_coverage as _ftc
                from _sources import dedup_papers
                from discover import web_search_paper_discovery

                # Raw query on ddgs returns query-echo junk; use the
                # context's domain-enriched per-entity strategies.
                rescue_queries = (
                    research_ctx.search_strategies
                    if research_ctx is not None and research_ctx.search_strategies
                    else [qs.audit]
                )
                rescue_raw: list = []
                for rq in rescue_queries[:4]:
                    got_r = web_search_paper_discovery(rq, max_results=max(6, config.max_papers))
                    rescue_raw.extend(got_r)
                    if len(rescue_raw) >= config.max_papers * 2:
                        break
                rescue = dedup_papers(rescue_raw)
                if rescue:
                    log.info("Web rescue: %d raw candidates", len(rescue))
                    rescued_kept, rescued_excluded = _ftc(rescue, qs.audit, alias_map=alias_map)
                    screening_log.extend(rescued_excluded)
                    if rescued_kept:
                        for _r in rescued_kept:
                            _r.discovery_provenance = "rescue:web-relevance"
                        papers = dedup_papers(list(papers) + rescued_kept)
                        log.info(
                            "Web rescue passed relevance gate: %d papers",
                            len(papers),
                        )
                    else:
                        log.info("Web rescue candidates also failed relevance gate")
            except Exception as ex:
                log.warning("Web-search rescue failed: %s", ex)
    except Exception as e:
        log.warning("Term-coverage filter skipped: %s", e)

    # ── Phase 1c-ii: LLM/content screening (single choke point) ─────────
    # Runs AFTER KB merge so local-knowledge-base papers pass the SAME
    # screen_paper gate as API/web papers (previously KB papers bypassed
    # screening entirely — place-name junk entered the corpus).
    try:
        from _screening import screen_paper

        before_domain = len(papers)
        kept: list = []
        for p in papers:
            decision, stage, reason = screen_paper(
                p,
                config.query,
                use_llm=config.use_llm,
                llm_model=config.llm_model or None,
                research_type=rtype,
            )
            if not decision:
                screening_log.append(
                    {
                        "paper_id": getattr(p, "primary_id", ""),
                        "doi": getattr(p, "doi", "") or "",
                        "title": getattr(p, "title", "") or "",
                        "decision": "exclude",
                        "stage": stage,
                        "reason": reason,
                    }
                )
            else:
                kept.append(p)
        papers = kept
        try:
            import json as _json

            (results_dir / "screening_decisions.json").write_text(
                _json.dumps(screening_log, indent=2, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
        except Exception:
            log.debug("Could not write screening_decisions.json", exc_info=True)
        if before_domain != len(papers):
            stage_counts: dict[str, int] = {}
            for entry in screening_log:
                stage_counts[entry["stage"]] = stage_counts.get(entry["stage"], 0) + 1
            breakdown = ", ".join(f"{s}={n}" for s, n in sorted(stage_counts.items()))
            log.info(
                "Screening: %d → %d papers (excluded %d: %s)",
                before_domain,
                len(papers),
                before_domain - len(papers),
                breakdown or "none",
            )
    except Exception as e:
        log.warning("Domain filter skipped (non-geo papers may appear): %s", e)

    # ── Fire streaming paper callbacks (screened corpus only) ───────────
    # Enables live paper feed in the GUI — only screened papers appear.
    if _paper_callbacks:
        for paper in papers:
            for cb in _paper_callbacks:
                try:
                    cb(paper)
                except Exception:
                    log.debug("paper callback failed", exc_info=True)

    # ── Honesty gate inputs (deterministic, no LLM) ─────────────────────
    from _honesty import (
        corpus_coverage,
        extract_query_entities,
        sufficiency_verdict,
    )

    query_entities = extract_query_entities(qs.audit)
    entity_coverage = corpus_coverage(papers, query_entities)
    insufficient, insufficiency_why = sufficiency_verdict(
        query_entities, entity_coverage, n_discovered, len(screening_log), n_final=len(papers)
    )

    corpus_path = results_dir / "corpus.json"
    save_corpus(
        papers,
        corpus_path,
        meta={"query": config.query, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    _emit(1, f"Discovery complete: {len(papers)} papers")

    if not papers:
        # Zero-corpus case: emit the AUDITABLE insufficient brief (entity
        # coverage table + rejection reasons), never a bare "No papers
        # found." that hides why discovery failed.
        brief_text = f"# Research Brief: {config.query}\n\nNo papers found.\n"
        try:
            from _honesty import render_insufficient_brief

            brief_text = render_insufficient_brief(
                config.query, query_entities, entity_coverage, screening_log, n_discovered
            )
        except Exception:
            log.debug("insufficient-brief render failed; plain fallback", exc_info=True)
        brief_path = results_dir / "research_brief.md"
        brief_path.write_text(brief_text, encoding="utf-8")
        return {
            "query": config.query,
            "research_type": rtype,
            "n_papers": 0,
            "n_fulltext_matched": 0,
            "paths": {
                "corpus": str(corpus_path),
                "brief": str(brief_path),
                "screening": str(results_dir / "screening_decisions.json"),
            },
            "brief_text": brief_text,
            "elapsed": time.time() - t0,
            "error": "No papers discovered",
        }

    # ── Honesty gate: refuse synthesis on an off-topic corpus ───────────
    # Fires when screening rejected >80% of discoveries AND zero corpus
    # papers mention any distinctive query term. Synthesizing anyway would
    # fabricate regional/comparative claims from unrelated papers (H7).
    if insufficient:
        from _honesty import render_insufficient_brief

        brief_text = render_insufficient_brief(
            config.query, query_entities, entity_coverage, screening_log, n_discovered
        )
        brief_path = results_dir / "research_brief.md"
        brief_path.write_text(brief_text, encoding="utf-8")
        log.warning("Sufficiency gate fired: %s — synthesis refused", insufficiency_why)
        _emit(1, f"Insufficient evidence: {insufficiency_why}")
        return {
            "query": config.query,
            "research_type": rtype,
            "n_papers": len(papers),
            "n_fulltext_matched": 0,
            "paths": {
                "corpus": str(corpus_path),
                "brief": str(brief_path),
                "screening": str(results_dir / "screening_decisions.json"),
            },
            "brief_text": brief_text,
            "elapsed": time.time() - t0,
            "error": f"Insufficient evidence: {insufficiency_why}",
        }

    # Re-enable S2 for abstract enrichment — even if user deselected S2
    # for discovery search, S2 provides abstracts Crossref/OpenAlex lack.
    try:
        from _sources import enable_s2_for_enrichment

        enable_s2_for_enrichment()
    except ImportError:
        log.debug("Optional module unavailable", exc_info=True)

    # ── Phase 1.5: ABSTRACT ENRICHMENT ─────────────────────────────────
    _enrich_abstracts(papers, config, _emit, corpus_path)

    # ── Phase 1.6: FULL-TEXT ENRICHMENT ─────────────────────────────────
    n_fulltext_enriched = _enrich_fulltext(papers, config, _emit)

    # ── Phase 2: VERIFICATION ───────────────────────────────────────────
    verified_path, n_verified, n_failed_verify = _phase_verification(
        config,
        results_dir,
        corpus_path,
        papers,
        _emit,
        t0,
    )

    # ── Phase 3: EXTRACTION ─────────────────────────────────────────────
    # When PaperQA2 is available + use_llm=True, use FAST regex extraction
    # (PaperQA2 does its own evidence gathering in Phase 5 — no need for
    # slow per-paper LLM extraction that competes for Ollama GPU time).
    _pqa_available = False
    if config.use_llm:
        try:
            from _paperqa import is_available

            _pqa_available = is_available()
        except ImportError:
            log.debug("paperqa not installed — LLM extraction disabled")
    _effective_use_llm = config.use_llm and not _pqa_available
    if _pqa_available and config.use_llm:
        log.info("PaperQA2 detected — using fast regex extraction (LLM synthesis via PaperQA2)")
        # Temporarily disable LLM extraction — PaperQA2 does its own evidence
        # gathering in Phase 5. Saves 5-8 min of Ollama GPU time.
        config.use_llm = False
    extractions, extracted_path, n_fulltext, verified_records = _phase_extraction(
        config,
        results_dir,
        verified_path,
        _emit,
        t0,
    )
    # Restore use_llm for Phase 5 synthesis decision
    if _pqa_available:
        config.use_llm = True

    # ── Phase 3.5: QUANTITATIVE DATA EXTRACTION (auto, guarded) ─────────
    # Extracts actual geochemical/geophysical numerical data from downloaded
    # PDF tables. Runs ONLY when PDFs are available. Completely additive —
    # never changes existing pipeline behavior, never crashes on failure.
    quant_data_path = _phase_quantitative_extraction(
        config,
        results_dir,
        extractions,
        _emit,
        t0,
    )

    # ── Phase 3b/3c: META-ANALYSIS + RISK-OF-BIAS ──────────────────────
    meta_analysis_path, assessment_path = _phase_meta_and_assessment(
        extractions,
        verified_records,
        config,
        results_dir,
        _emit,
    )

    # S2 circuit NOT reset mid-pipeline — if S2 was rate-limited during
    # verification, it will still be rate-limited here (sliding window).
    # Resetting just causes fresh 429 cascades. OpenAlex cited_by is the
    # primary citation source (generous rate limit).

    # ── Phase 4: CORRELATION ────────────────────────────────────────────
    # When PaperQA2 is available, skip LLM stance detection in correlation —
    # PaperQA2 does its own evidence gathering in Phase 5. This prevents
    # Ollama circuit-breaker cascades when processing many papers.
    if _pqa_available:
        config.use_llm = False
    correlation_path = _phase_correlation(
        config,
        extractions,
        results_dir,
        verified_path,
        _emit,
        t0,
    )
    if _pqa_available:
        config.use_llm = True

    # ── Phase 5: SYNTHESIS + CITATION EXPORT ────────────────────────────
    brief_text, brief_path, citations_bib_path = _phase_synthesis(
        config,
        results_dir,
        extracted_path,
        verified_path,
        correlation_path,
        meta_analysis_path,
        _emit,
        t0,
        display_query=qs.raw,  # original user query — qs.search is synonym-expanded and made garbage titles
        audit_query=qs.audit,
        query_corrections=spell_fix_notes,
        quant_data_path=quant_data_path,
        research_ctx=research_ctx,
    )

    # ── Phase 6: AGENTIC REFLECT LOOP (critique → follow-up search →
    # re-synthesis). The 2026 deep-research loop: an LLM critic judges
    # whether corpus + brief answer the question; "insufficient" triggers
    # bounded follow-up discovery and one re-synthesis per iteration.
    # Bounded by config.reflect_iterations; every exit is loud.
    if getattr(config, "reflect_iterations", 0) > 0:
        try:
            import urllib.request as _ur_reflect

            with _ur_reflect.urlopen(_ur_reflect.Request("http://127.0.0.1:11434/api/tags"), timeout=3):
                _reflect_llm_ok = True
        except Exception:
            _reflect_llm_ok = False
        if not _reflect_llm_ok:
            _emit(6, "Reflect skipped — Ollama unreachable (brief kept as synthesized)")
        else:
            from _claims_engine import (
                contradiction_report,
                detect_contradictions,
                extract_claims,
            )
            from _reflect import critique_coverage
            from _sources import dedup_papers, load_corpus, save_corpus
            from discover import search_multi_source

            # Wire contradictions (advisory) into critic input + brief
            # appendix. Scoped to the SAME corpus the brief synthesizes
            # (_extract_papers over the artifacts) — the raw extractions
            # list mixed in off-topic papers the brief never cites, and
            # duplicated sources (2026-09-02). Full text joined back from
            # the extraction rows for corpus papers only.
            _contradiction_text = ""
            try:
                from _artifact import _extract_papers, load_extractions, load_verified

                _ext_art = load_extractions(extracted_path)
                _scan_papers = _extract_papers(_ext_art, load_verified(verified_path))
                _fulltexts = {
                    str(e.get("doi") or ""): e.get("full_text") or "" for e in (_ext_art.get("extractions") or [])
                }
                _claims = []
                for _p in _scan_papers:
                    _claims.extend(extract_claims(_p, _fulltexts.get(str(_p.get("doi") or ""), "")))
                _cons = detect_contradictions(_claims)
                if _cons:
                    _contradiction_text = contradiction_report(_cons)
                    log.info("Reflect: %d numeric contradiction(s) detected", len(_cons))
            except Exception as exc:
                log.warning("Contradiction scan failed (brief continues): %s", exc)

            def _resynthesize():
                return _phase_synthesis(
                    config,
                    results_dir,
                    extracted_path,
                    verified_path,
                    correlation_path,
                    meta_analysis_path,
                    _emit,
                    t0,
                    display_query=config.query,
                    audit_query=qs.audit,
                    query_corrections=spell_fix_notes,
                    quant_data_path=quant_data_path,
                    research_ctx=research_ctx,
                )

            _reflect_notes: list[str] = []
            _n_reflect_added = 0
            _round = 0
            while _round < config.reflect_iterations:
                _critique = critique_coverage(
                    config.query,
                    brief_text,
                    _contradiction_text,
                    {
                        "n_papers": len(extractions),
                        "n_verified": n_verified,
                        "sources": config.sources,
                        "research_type": rtype,
                    },
                )
                if _critique is None:
                    _emit(6, "Reflect: critic unavailable (LLM error) — keeping current brief")
                    break
                if _critique["verdict"] == "sufficient":
                    _emit(6, "Reflect critique: SUFFICIENT — brief stands")
                    _reflect_notes.append(f"- Critique verdict: **sufficient** — {_critique.get('reasoning', '')}")
                    break
                _queries = _critique["follow_up_queries"]
                _emit(
                    6,
                    f"Reflect critique: INSUFFICIENT — {len(_critique['gaps'])} gap(s), "
                    f"{len(_queries)} follow-up quer{'y' if len(_queries) == 1 else 'ies'}",
                )
                if not _queries:
                    _emit(6, "Reflect: no actionable follow-up queries — stopping")
                    break
                _new: list = []
                for _q in _queries:
                    try:
                        _new.extend(
                            search_multi_source(
                                _q,
                                max_per_source=5,
                                sources=config.sources,
                                use_web_search=config.use_web_search,
                                per_source_timeout=config.search_timeout,
                                original_query=config.query,
                            )
                        )
                    except Exception as exc:
                        log.warning("Follow-up search failed for %r: %s", _q, exc)
                _current = load_corpus(verified_path)
                _existing_dois = {str(pp.doi or "").lower() for pp in _current}
                _existing_titles = {(pp.title or "").lower()[:80] for pp in _current}
                _fresh = [
                    pp
                    for pp in dedup_papers(_new)
                    if str(pp.doi or "").lower() not in _existing_dois
                    and (pp.title or "").lower()[:80] not in _existing_titles
                ]
                if not _fresh:
                    _emit(
                        6,
                        f"Reflect: {len(_queries)} follow-up queries added 0 new papers — stopping",
                    )
                    _reflect_notes.append(
                        f"- Critique: insufficient "
                        f"({'; '.join(str(g) for g in _critique['gaps'][:3])}) but "
                        f"follow-up searches added no new papers."
                    )
                    break
                _round += 1
                _n_reflect_added += len(_fresh)
                _emit(6, f"Reflect round {_round}: +{len(_fresh)} papers — regenerating brief")
                _reflect_notes.append(
                    f"- Round {_round}: critique found gaps "
                    f"({'; '.join(str(g) for g in _critique['gaps'][:3])}); "
                    f"{len(_queries)} follow-up queries → {len(_fresh)} new papers."
                )
                save_corpus(_current + _fresh, verified_path)
                extractions = extractions + [
                    {
                        "doi": pp.doi,
                        "title": pp.title,
                        "abstract": pp.abstract,
                        "source": pp.source,
                        "reflect_added": True,
                    }
                    for pp in _fresh
                ]
                brief_text, brief_path, citations_bib_path = _resynthesize()

            # Transparency appendix: contradictions + reflection provenance
            _appendix = []
            if _contradiction_text:
                _appendix.append("## Numeric Contradictions (advisory — analyst adjudicates)\n\n" + _contradiction_text)
            if _reflect_notes:
                _appendix.append("## Reflective Research Notes\n\n" + "\n".join(_reflect_notes))
            if _appendix:
                brief_text = brief_text + "\n\n---\n\n" + "\n\n".join(_appendix)
                brief_path.write_text(brief_text, encoding="utf-8")

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1fs", elapsed)

    # Save pipeline metadata (+ retention manifest 2026-09-03: config
    # snapshot, query provenance, and per-artifact SHA256 so any run dir is
    # self-verifying and replayable; mirrors chat tool-call data handling)
    _artifact_hashes = {}
    for _f in sorted(results_dir.iterdir()):
        if _f.is_file():
            _artifact_hashes[_f.name] = hashlib.sha256(_f.read_bytes()).hexdigest()
    meta = {
        "query": config.query,
        "search_query": qs.search,
        "audit_query": qs.audit,
        "query_fixes": list(qs.fixes),
        "spell_fix_notes": spell_fix_notes,
        "model": config.llm_model or "auto",
        "config": {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()},
        "artifact_sha256": _artifact_hashes,
        "research_type": rtype,
        "n_papers": len(extractions),
        "n_verified": n_verified,
        "n_fulltext_matched": n_fulltext,
        "n_local_papers": get_local_paper_count(config.fulltext_dir) if config.fulltext_dir else 0,
        "use_llm": config.use_llm,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": config.sources,
        "synthesis_version": SYNTHESIS_VERSION,
    }
    (results_dir / "meta.json").write_text(
        orjson.dumps(meta, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"),
        encoding="utf-8",
    )

    # Update research index
    _update_research_index(results_dir, meta)

    results = {
        "query": config.query,
        "research_type": rtype,
        "n_papers": len(extractions),
        "n_verified": n_verified,
        "n_failed_verification": n_failed_verify,
        "verification_skipped": config.skip_verify,
        "n_fulltext_matched": n_fulltext,
        "paths": {
            "corpus": str(corpus_path),
            "verified": str(verified_path),
            "extracted": str(extracted_path),
            "correlation": str(correlation_path),
            "brief": str(brief_path),
            "meta": str(results_dir / "meta.json"),
            "meta_analysis": str(meta_analysis_path) if meta_analysis_path.exists() else None,
            "assessment": str(assessment_path) if assessment_path.exists() else None,
            "citations": str(citations_bib_path) if citations_bib_path.exists() else None,
            "screening": str(results_dir / "screening_decisions.json")
            if (results_dir / "screening_decisions.json").exists()
            else None,
        },
        "brief_text": brief_text,
        "elapsed": elapsed,
        "results_dir": str(results_dir),
        "quality": {
            "embeddings_available": _check_embeddings_available(),
            "llm_used": config.use_llm,
            "fulltext_pdfs_downloaded": n_fulltext_enriched if "n_fulltext_enriched" in dir() else 0,
            "sources_used": config.sources,
        },
    }

    _notify_research_complete(results)
    return results


def _update_research_index(results_dir: Path, meta: dict) -> None:
    """Update data/research/_index.json with this run's metadata."""
    index_path = results_dir.parent / "_index.json"
    try:
        index: dict = {}
        if index_path.exists():
            index = json.loads(index_path.read_text(encoding="utf-8"))
        index[meta["query"][:80]] = {
            "query": meta["query"],
            "type": meta["research_type"],
            "n_papers": meta["n_papers"],
            "timestamp": meta["timestamp"],
            "dir": str(results_dir),
        }
        index_path.write_text(
            orjson.dumps(index, option=orjson.OPT_INDENT_2 | orjson.OPT_SERIALIZE_NUMPY).decode("utf-8"),
            encoding="utf-8",
        )
    except Exception as e:
        log.warning("Failed to update research index: %s", e)


# ── CLI entry (2026-10-06): make the pipeline reachable without the GUI ──
import argparse
from datetime import datetime, timezone


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "research"


def _default_output_dir(query: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return Path("research_outputs") / f"pipeline-{ts}-{_slugify(query)}"


def build_config(args: argparse.Namespace) -> ResearchConfig:
    return ResearchConfig(
        query=args.query,
        sources=[s.strip() for s in args.sources.split(",") if s.strip()],
        max_papers=args.max,
        use_llm=args.use_llm,
        llm_model=args.llm_model,
        llm_quality=args.llm_quality,
        research_type_override=args.research_type,
        skip_verify=args.skip_verify,
        skip_correlate=args.skip_correlate,
        match_local_pdfs=not args.no_local_pdfs,
        output_dir=Path(args.out_dir) if args.out_dir else _default_output_dir(args.query),
        fulltext_dir=Path(args.fulltext_dir) if args.fulltext_dir else None,
        reflect_iterations=0 if args.no_reflect else args.reflect,
        use_web_search=not args.no_web_search,
        use_web_search_agentic=args.agentic_web,
        use_web_pro=not args.no_web_pro,
        year_from=args.from_year,
        year_to=args.to_year,
        open_access_only=args.open_access_only,
        publication_type=args.type,
        max_pdf_downloads=args.max_pdf,
        wall_clock_budget_s=args.budget,
    )


def _progress_to_stderr(phase_num: int, message: str) -> None:
    print(f"[phase {phase_num}] {message}", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    import _bootstrap

    _bootstrap.ensure_env()

    parser = argparse.ArgumentParser(
        prog="pipeline.py",
        description="5-phase scientific research pipeline (discovery -> verification -> "
        "extraction -> correlation -> synthesis). Writes brief + corpus + verified "
        "DOIs + extraction JSON under the output dir.",
    )
    parser.add_argument("query", help="Research question (quoted string)")
    parser.add_argument("--max", type=int, default=30, help="Max papers to keep (default 30)")
    parser.add_argument(
        "--sources",
        default="web_search,crossref,openalex,s2,eartharxiv,usgs",
        help="Comma-separated discovery sources",
    )
    parser.add_argument("--use-llm", action="store_true", help="Enable Ollama LLM extraction")
    parser.add_argument("--llm-model", default="", help="Ollama model (empty = auto-detect smallest)")
    parser.add_argument(
        "--llm-quality",
        choices=["fast", "balanced", "quality"],
        default="balanced",
        help="Model selection policy",
    )
    parser.add_argument("--from-year", type=int, default=None, help="Earliest publication year")
    parser.add_argument("--to-year", type=int, default=None, help="Latest publication year")
    parser.add_argument("--open-access-only", action="store_true", help="Only open-access papers")
    parser.add_argument("--type", default="", help="Publication type filter (e.g. journal-article)")
    parser.add_argument("--reflect", type=int, default=1, help="LLM critic re-search rounds (default 1)")
    parser.add_argument("--no-reflect", action="store_true", help="Disable the reflect loop")
    parser.add_argument("--no-web-pro", action="store_true", help="Skip the Perplexity-style web synthesis section")
    parser.add_argument("--no-web-search", action="store_true", help="Disable web_search discovery source")
    parser.add_argument("--agentic-web", action="store_true", help="Deep web discovery (trafilatura + DOI mining)")
    parser.add_argument("--skip-verify", action="store_true", help="Skip Phase 2 DOI verification")
    parser.add_argument("--skip-correlate", action="store_true", help="Skip Phase 4 cross-correlation")
    parser.add_argument("--no-local-pdfs", action="store_true", help="Do not match local PDF library")
    parser.add_argument("--max-pdf", type=int, default=10, help="Max OA PDFs to download + extract")
    parser.add_argument("--budget", type=float, default=None, help="Wall-clock budget in seconds")
    parser.add_argument("--out-dir", default=None, help="Output dir (default research_outputs/pipeline-<ts>-<slug>)")
    parser.add_argument("--fulltext-dir", default=None, help="Local PDF library dir to match")
    parser.add_argument("--research-type", default=None, help="Override auto-detected research type")
    parser.add_argument("--json", action="store_true", help="Print results as JSON to stdout")
    parser.add_argument(
        "--with-brief",
        action="store_true",
        help="With --json: include full brief_text in the JSON output",
    )
    args = parser.parse_args(argv)

    config = build_config(args)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        results = run_pipeline(config, progress=_progress_to_stderr)
    except PipelineBusyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        payload = {k: v for k, v in results.items() if (args.with_brief or k != "brief_text")}
        print(json.dumps(payload, default=str))
    else:
        print(f"Research type : {results.get('research_type', '?')}")
        print(
            f"Papers        : {results.get('n_papers', '?')} (fulltext matched: {results.get('n_fulltext_matched', '?')})"
        )
        print(f"Elapsed       : {results.get('elapsed', 0):.1f}s")
        for name, p in (results.get("paths") or {}).items():
            print(f"{name:<13} : {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
