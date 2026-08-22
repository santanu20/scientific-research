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

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))
from config import ResearchConfig

# Local-PDF matching is a geokit-environment feature (out of this skill's
# web-only scope). Lazy + loud: absent geokit → warning + skip that
# enhancement; the web pipeline itself must run anywhere.
try:
    from geokit.research._fulltext import (  # pyright: ignore[reportMissingImports]
        build_local_index,
        get_local_paper_count,
        load_fulltext,
        match_paper_to_local,
    )

    _GEOKIT_FULLTEXT = True
except ImportError:
    _GEOKIT_FULLTEXT = False

    def _geokit_unavailable(*_args, **_kwargs):
        raise RuntimeError(
            "geokit full-text matching not available in this environment"
        )

    build_local_index = get_local_paper_count = _geokit_unavailable
    load_fulltext = match_paper_to_local = _geokit_unavailable

log = logging.getLogger(__name__)

# Ensure scripts directory is on sys.path for imports
_SCRIPTS_DIR = Path(__file__).parent / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


ProgressCallback = Callable[[int, str], None]
"""Phase callback: (phase_number 1-5, status_message)."""


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
    """
    t0 = time.time()
    results_dir = config.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

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
        from _llm_extract import set_llm_params

        set_llm_params(config.llm_num_ctx, config.llm_num_predict)
    except ImportError:
        pass

    def _emit(phase: int, msg: str) -> None:
        log.info("[Phase %d/5] %s", phase, msg)
        if progress:
            progress(phase, msg)

    def _provenance(stage: str, params: dict, inputs: dict, outputs: dict) -> None:
        """Append audit record per stage (FAIR / PRISMA-S trail). Fails soft:
        provenance must never abort the research run."""
        try:
            from _provenance import record_stage

            record_stage(
                stage,
                params=params,
                inputs=inputs,
                outputs=outputs,
                results_dir=results_dir,
            )
        except Exception as e:
            log.warning("Provenance record for '%s' failed: %s", stage, e)

    # Skip S2 entirely if user didn't select it as a source
    if "s2" not in config.sources:
        try:
            from _sources import force_skip_s2

            force_skip_s2()
            log.info("S2 skipped — not in user-selected sources")
        except ImportError:
            pass

    # ── Phase 1: DISCOVERY ──────────────────────────────────────────────
    _emit(1, f"Searching: {config.query[:60]}...")
    from _sources import dedup_papers, save_corpus
    from discover import search_multi_source

    t_phase = time.time()
    # Build search filters from config
    search_filters: dict = {}
    if config.year_from or config.year_to:
        search_filters["year_from"] = config.year_from
        search_filters["year_to"] = config.year_to
    if config.open_access_only:
        search_filters["open_access_only"] = True
    if config.publication_type:
        search_filters["publication_type"] = config.publication_type
    search_filters["query"] = config.query  # for domain relevance filtering

    # Over-fetch 1.5× to compensate for papers that will be discarded
    # (no abstract = no data for extraction/synthesis)
    over_fetch = max(5, int(config.max_papers * 1.5 // len(config.sources)))

    raw_papers = search_multi_source(
        config.query,
        max_per_source=over_fetch,
        sources=config.sources,
        per_source_timeout=config.search_timeout,
        filters=search_filters or None,
        use_web_search=config.use_web_search,
        use_web_search_agentic=config.use_web_search_agentic,
    )

    # Auto-supplement with web_search when discovery is sparse.
    # Triggered when: (a) auto threshold > 0, (b) user didn't already enable
    # web_search explicitly, (c) raw count below threshold.
    # Catches niche topics where Crossref/OpenAlex/S2 miss the field entirely.
    if (
        config.auto_web_search_threshold > 0
        and not config.use_web_search
        and not config.use_web_search_agentic
        and len(raw_papers) < config.auto_web_search_threshold
    ):
        log.info(
            "Sparse discovery (%d papers < threshold %d) — auto-supplementing via web_search",
            len(raw_papers),
            config.auto_web_search_threshold,
        )
        try:
            from discover import web_search_paper_discovery

            supplement = web_search_paper_discovery(
                config.query,
                max_results=max(10, config.auto_web_search_threshold * 2),
            )
            if supplement:
                log.info("web_search auto-supplement: +%d papers", len(supplement))
                raw_papers.extend(supplement)
        except Exception as ex:
            log.warning("web_search auto-supplement failed: %s", ex)

    deduped = dedup_papers(raw_papers)
    n_after_dedup = len(deduped)

    # Domain relevance filter — remove papers from unrelated fields
    try:
        from discover import _is_domain_relevant

        before_domain = len(deduped)
        deduped = [p for p in deduped if _is_domain_relevant(p, config.query)]
        if before_domain != len(deduped):
            log.info(
                "Domain filter: %d → %d papers (removed %d off-domain)",
                before_domain,
                len(deduped),
                before_domain - len(deduped),
            )
    except Exception as e:
        log.debug("Domain filter skipped: %s", e)

    # Post-discovery filter: enforce year/OA/type constraints (KB cache may bypass API filters)
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

    # PRISMA Phase-2 triage — non-LLM keyword screen with include terms
    # derived DYNAMICALLY from the query itself (no field vocabulary).
    # Zero-query-overlap papers are excluded (citation-noise guard);
    # a sparse survivor set (<3) skips screening instead of shrinking
    # the corpus. Exclusions are logged + rendered for the PRISMA trail.
    if config.screen:
        try:
            from discover import _content_tokens

            include_terms = sorted(_content_tokens(config.query))
            if not include_terms:
                log.info("Screen skipped: query has no content tokens")
            else:
                from screen import (
                    prisma_phase2_counts,
                    render_screening_report,
                    screen_corpus,
                )

                pre_screen = len(deduped)
                scored = screen_corpus(
                    deduped,
                    include=include_terms,
                    exclude=[],
                    query=config.query,
                )
                counts = prisma_phase2_counts(scored)
                kept = [s.paper for s in scored if s.recommendation != "exclude"]
                log.info(
                    "Screen: %d -> %d papers (excluded %d zero-overlap, "
                    "%d borderline kept)",
                    pre_screen,
                    len(kept),
                    counts["n_exclude_recommended"],
                    counts["n_borderline"],
                )
                if len(kept) >= 3:
                    deduped = kept
                    try:
                        (results_dir / "screening_report.md").write_text(
                            render_screening_report(scored, counts),
                            encoding="utf-8",
                        )
                    except OSError as ex:
                        log.warning("Screening report write failed: %s", ex)
                else:
                    log.warning(
                        "Screen would shrink corpus to %d (<3) — skipped this run",
                        len(kept),
                    )
        except ImportError as e:
            log.warning("Screen stage unavailable — continuing unfiltered: %s", e)

    # Rank by semantic relevance (TF-IDF cosine) before truncation so
    # the TOP papers are kept when corpus exceeds max_papers.
    try:
        from _ranking import semantic_relevance_scores

        scores = semantic_relevance_scores(config.query, deduped)
        ranked = sorted(zip(deduped, scores), key=lambda x: -x[1])

        # Prefer papers WITH abstracts — discard abstractless papers
        # when enough abstracted papers exist to fill max_papers
        with_abstract = [p for p, _ in ranked if (p.abstract or "").strip()]
        without_abstract = [p for p, _ in ranked if not (p.abstract or "").strip()]

        if len(with_abstract) >= config.max_papers:
            # Enough abstracted papers — discard all abstractless
            papers = with_abstract[: config.max_papers]
            log.info(
                "Discarded %d papers without abstracts, kept %d with abstracts",
                len(without_abstract),
                len(papers),
            )
        elif len(with_abstract) > 0:
            # Not enough abstracted — fill remaining with best abstractless
            papers = (
                with_abstract
                + without_abstract[: config.max_papers - len(with_abstract)]
            )
            log.info(
                "Using %d papers with abstracts + %d without (fill gap)",
                len(with_abstract),
                len(papers) - len(with_abstract),
            )
        else:
            # No abstracted papers at all — use all (title fallback)
            papers = [p for p, _ in ranked[: config.max_papers]]
            log.warning("No papers have abstracts — using title-based fallback for all")

        log.info(
            "Ranked %d papers by semantic relevance (top score=%.3f)",
            len(papers),
            scores.max() if len(scores) > 0 else 0,
        )
    except Exception as e:
        log.warning("Ranking failed (%s), using arbitrary order", e)
        papers = deduped[: config.max_papers]
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

    corpus_path = results_dir / "corpus.json"
    save_corpus(
        papers,
        corpus_path,
        meta={"query": config.query, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S")},
    )
    _provenance(
        "discovery",
        params={
            "query": config.query,
            "sources": config.sources,
            "max_papers": config.max_papers,
            "year_from": config.year_from,
            "year_to": config.year_to,
            "open_access_only": config.open_access_only,
            "publication_type": config.publication_type,
        },
        inputs={},
        outputs={"corpus": corpus_path},
    )
    _emit(1, f"Discovery complete: {len(papers)} papers")

    if not papers:
        return {
            "query": config.query,
            "research_type": "survey",
            "n_papers": 0,
            "n_fulltext_matched": 0,
            "paths": {"corpus": str(corpus_path)},
            "brief_text": f"# Research Brief: {config.query}\n\nNo papers found.\n",
            "elapsed": time.time() - t0,
            "error": "No papers discovered",
        }

    # Re-enable S2 for abstract enrichment — even if user deselected S2
    # for discovery search, S2 provides abstracts Crossref/OpenAlex lack.
    # force_skip_s2() was called at line ~105 when "s2" not in sources;
    # this overrides it for enrichment (metadata fetch, not discovery).
    try:
        from _sources import enable_s2_for_enrichment

        enable_s2_for_enrichment()
    except ImportError:
        pass

    # ── Phase 1.5: PARALLEL ABSTRACT ENRICHMENT ─────────────────────
    # S2 and OpenAlex DOI lookup run concurrently in separate threads.
    # Each thread is sequential internally (rate-limit safe):
    #   S2: 1.2s/call, circuit breaker threshold=2
    #   OpenAlex: ~1s/call (polite pool), generous rate limit
    # Race condition on shared PaperRecord.abstract is benign:
    #   check-and-set pattern — if paper already enriched by the other
    #   thread, the second writer skips (both abstracts are valid anyway).
    abstractless = [p for p in papers if not (p.abstract or "").strip() and p.doi]
    if abstractless:
        # OpenAlex targets: skip papers already fetched from OpenAlex
        # (their inverted index was already parsed during search)
        oa_targets = [
            p for p in abstractless if "openalex" not in (p.sources_seen or [])
        ]
        t_enrich = time.time()
        _emit(1, f"Enriching {len(abstractless)} abstracts (S2 + OpenAlex parallel)...")

        def _enrich_s2(targets: list) -> int:
            count = 0
            try:
                from _sources import _s2_available, s2_get_paper

                for p in targets:
                    if not _s2_available():
                        log.info("S2 circuit open — skipping remaining S2")
                        break
                    rec = s2_get_paper(f"DOI:{p.doi}")
                    if (
                        rec
                        and (rec.abstract or "").strip()
                        and not (p.abstract or "").strip()
                    ):
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
                from _sources import openalex_get_by_doi, reset_openalex_circuit

                reset_openalex_circuit()
                for p in targets:
                    rec = openalex_get_by_doi(p.doi)
                    if (
                        rec
                        and (rec.abstract or "").strip()
                        and not (p.abstract or "").strip()
                    ):
                        p.abstract = rec.abstract
                        if "openalex" not in p.sources_seen:
                            p.sources_seen.append("openalex")
                        count += 1
            except Exception as e:
                log.warning("OpenAlex enrichment error: %s", e)
            return count

        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as pool:
            fut_s2 = pool.submit(_enrich_s2, abstractless)
            fut_oa = pool.submit(_enrich_openalex, oa_targets)
            s2_count = fut_s2.result()
            oa_count = fut_oa.result()

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

    # ── Phase 2: VERIFICATION ───────────────────────────────────────────
    t_phase = time.time()
    verified_path = results_dir / "verified.json"
    if config.skip_verify:
        # Skip verification — use corpus as verified
        verified_path.write_text(
            corpus_path.read_text(encoding="utf-8"), encoding="utf-8"
        )
        n_verified = len(papers)
        _emit(2, f"Verification skipped: {n_verified} papers")
    else:
        _emit(2, f"Verifying {len(papers)} papers...")
        from _sources import load_corpus, save_corpus
        from verify import load_rw_index, verify_paper

        # Retraction Watch CSV: --rw-csv arg > env > default install location
        # (auto-discovered; fail-open with one log line when absent)
        rw_index = load_rw_index()

        corpus_papers = load_corpus(corpus_path)

        # Parallel verify — verify_paper is I/O-bound (API calls), thread-safe
        from concurrent.futures import ThreadPoolExecutor, as_completed

        verified_records: list = []
        failed = 0

        def _verify_one(paper):
            try:
                merged, result = verify_paper(paper, rw_index=rw_index)
                return merged if result.resolved else None
            except Exception as e:
                log.warning("verify failed for %s: %s", paper.primary_id, e)
                return None

        max_workers = min(config.verify_workers, len(corpus_papers))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_verify_one, p): p for p in corpus_papers}
            for future in as_completed(futures):
                result_rec = future.result()
                if result_rec is not None:
                    verified_records.append(result_rec)
                else:
                    failed += 1

        if failed:
            log.warning(
                "  %d/%d papers failed verification", failed, len(corpus_papers)
            )

        save_corpus(
            verified_records,
            verified_path,
            meta={
                "query": config.query,
                "total_in": len(corpus_papers),
                "verified": len(verified_records),
            },
        )
        _provenance(
            "verification",
            params={
                "skip_verify": False,
                "workers": config.verify_workers,
                "rw_csv_env": bool(os.environ.get("SCIENTIFIC_RESEARCH_RW_CSV")),
            },
            inputs={"corpus": corpus_path},
            outputs={"verified": verified_path},
        )
        n_verified = len(verified_records)
        log.info(
            "Phase 2 timing: %.1fs (%d/%d verified)",
            time.time() - t_phase,
            n_verified,
            len(corpus_papers),
        )
        _emit(2, f"Verified: {n_verified} papers")

    # ── Phase 3: EXTRACTION ─────────────────────────────────────────────
    _emit(3, "Extracting PICO + effect sizes...")
    t_phase = time.time()
    extracted_path = results_dir / "extracted.json"

    # Build local paper index for full-text matching
    local_index: list[dict] = []
    n_fulltext = 0
    if config.match_local_pdfs and config.fulltext_dir:
        if _GEOKIT_FULLTEXT:
            local_index = build_local_index(config.fulltext_dir)
        else:
            log.warning(
                "match_local_pdfs=True but geokit not installed — "
                "local full-text matching SKIPPED (web pipeline continues)"
            )

    # Load verified papers as PaperRecord objects
    from _sources import PaperRecord, load_corpus
    from extract import extract_from_paper

    verified_records = load_corpus(verified_path)
    extractions: list[dict] = []
    n_total = len(verified_records)
    for i, record in enumerate(verified_records):
        # Check for local full-text match
        if local_index and _GEOKIT_FULLTEXT:
            local_path = match_paper_to_local(
                paper_doi=record.doi,
                paper_title=record.title,
                local_index=local_index,
            )
            if local_path:
                full_text = load_fulltext(local_path)
                if full_text and len(full_text) > len(record.abstract or ""):
                    # Replace abstract with full text (capped)
                    record = PaperRecord.from_dict(record.to_dict())
                    record.abstract = full_text[: config.fulltext_cap]
                    n_fulltext += 1

        # Cap abstract length for extraction efficiency
        if (
            config.abstract_cap
            and record.abstract
            and len(record.abstract) > config.abstract_cap
        ):
            record = PaperRecord.from_dict(record.to_dict())
            record.abstract = record.abstract[: config.abstract_cap]

        # Extract
        try:
            if config.use_llm:
                _emit(3, f"Extracting {i + 1}/{n_total} (LLM)…")
            result = extract_from_paper(
                record, topic=config.query, use_llm=config.use_llm
            )
            extractions.append(result.to_dict())
            log.debug(
                "  [%d/%d] extracted: %s (disc=%s)",
                i + 1,
                len(verified_records),
                record.title[:40] if record.title else "?",
                result.pico.get("discipline", "?"),
            )
        except Exception as e:
            log.warning(
                "  [%d/%d] extraction FAILED for '%s': %s",
                i + 1,
                len(verified_records),
                (record.title or "?")[:40],
                e,
            )

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
        json.dumps(extracted_data, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _provenance(
        "extraction",
        params={
            "use_llm": config.use_llm,
            "llm_model": config.llm_model,
            "match_local_pdfs": config.match_local_pdfs and _GEOKIT_FULLTEXT,
        },
        inputs={"verified": verified_path},
        outputs={"extracted": extracted_path},
    )
    log.info(
        "Phase 3 timing: %.1fs (%d extracted, %d full-text, %d failed)",
        time.time() - t_phase,
        len(extractions),
        n_fulltext,
        len(verified_records) - len(extractions),
    )
    _emit(3, f"Extraction complete: {len(extractions)} papers ({n_fulltext} full-text)")

    # ── Phase 3b: META-ANALYSIS (optional) ─────────────────────────────
    meta_analysis_path = results_dir / "meta_analysis.json"
    meta_analysis_data = None
    try:
        from meta_analyze import collect_continuous_effects, pool_effects

        continuous_studies = collect_continuous_effects(extractions, verified_records)
        if continuous_studies:
            pooled = pool_effects(continuous_studies, model="random")
            meta_analysis_data = {
                "n_studies": len(continuous_studies),
                "model": "random",
                "pooled_effect": pooled,
            }
            meta_analysis_path.write_text(
                json.dumps(
                    meta_analysis_data, indent=2, default=str, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            log.info("Meta-analysis: %d studies pooled", len(continuous_studies))
            _emit(3, f"Meta-analysis: {len(continuous_studies)} studies pooled")
        else:
            log.info("Meta-analysis: no continuous effect sizes found")
    except Exception as e:
        log.warning("Meta-analysis failed: %s", e)

    # ── Phase 3c: RISK-OF-BIAS ASSESSMENT (optional) ────────────────────
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
            except Exception:
                pass
        if assessments:
            assessment_path.write_text(
                json.dumps(assessments, indent=2, default=str, ensure_ascii=False),
                encoding="utf-8",
            )
            log.info("Assessment: %d papers assessed", len(assessments))
    except Exception as e:
        log.warning("Assessment failed: %s", e)

    # Reset S2 circuit for correlation (citation lookups)
    try:
        from _sources import reset_s2_circuit

        reset_s2_circuit()
    except ImportError:
        pass

    # ── Phase 4: CORRELATION ────────────────────────────────────────────
    correlation_path = results_dir / "correlation.json"
    if config.skip_correlate or len(extractions) < 3:
        correlation_path.write_text("{}", encoding="utf-8")
        _emit(4, "Correlation skipped (insufficient papers)")
    else:
        _emit(4, "Building citation graph + stance matrix...")
        try:
            from _sources import load_corpus
            from correlate import correlate_corpus

            verified_records = load_corpus(verified_path)
            correlation_data = correlate_corpus(
                verified_records,
                no_llm_stance=not config.use_llm,
            )
            correlation_path.write_text(
                json.dumps(correlation_data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            _provenance(
                "correlation",
                params={"use_llm_stance": config.use_llm, "skip_correlate": False},
                inputs={"verified": verified_path, "extracted": extracted_path},
                outputs={"correlation": correlation_path},
            )
            _emit(4, "Correlation complete")
        except Exception as e:
            log.warning("Correlation failed: %s", e)
            correlation_path.write_text("{}", encoding="utf-8")
            _emit(4, f"Correlation failed: {e}")

    # ── Phase 5: SYNTHESIS ──────────────────────────────────────────────
    _emit(5, "Synthesizing research brief...")
    from synthesize import synthesize as run_synthesize

    brief_path = results_dir / "research_brief.md"
    meta_for_synth = meta_analysis_path if meta_analysis_path.exists() else None
    brief_text = run_synthesize(
        query=config.research_type_override or config.query,
        extracted_path=extracted_path,
        verified_path=verified_path,
        correlation_path=correlation_path,
        meta_path=meta_for_synth,
        output_path=brief_path,
        use_llm=config.use_llm,
    )
    _emit(5, "Synthesis complete")

    # Determine research type
    from _classifiers import detect_research_type

    rtype = config.research_type_override or detect_research_type(config.query)

    # ── Citation Export ──────────────────────────────────────────────────
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

    _provenance(
        "synthesis",
        params={
            "use_llm_smoothing": config.use_llm,
            "research_type_override": config.research_type_override,
        },
        inputs={
            "verified": verified_path,
            "extracted": extracted_path,
            "correlation": correlation_path,
        },
        outputs={"brief": brief_path, "citations": citations_bib_path},
    )

    elapsed = time.time() - t0
    log.info("Pipeline complete in %.1fs", elapsed)

    # Save pipeline metadata
    meta = {
        "query": config.query,
        "research_type": rtype,
        "n_papers": len(extractions),
        "n_verified": n_verified,
        "n_fulltext_matched": n_fulltext,
        "n_local_papers": get_local_paper_count(config.fulltext_dir)
        if (config.fulltext_dir and _GEOKIT_FULLTEXT)
        else 0,
        "use_llm": config.use_llm,
        "elapsed_seconds": elapsed,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "sources": config.sources,
    }
    (results_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Update research index
    _update_research_index(results_dir, meta)

    return {
        "query": config.query,
        "research_type": rtype,
        "n_papers": len(extractions),
        "n_verified": n_verified,
        "n_fulltext_matched": n_fulltext,
        "paths": {
            "corpus": str(corpus_path),
            "verified": str(verified_path),
            "extracted": str(extracted_path),
            "correlation": str(correlation_path),
            "brief": str(brief_path),
            "meta": str(results_dir / "meta.json"),
            "meta_analysis": str(meta_analysis_path)
            if meta_analysis_path.exists()
            else None,
            "assessment": str(assessment_path) if assessment_path.exists() else None,
            "citations": str(citations_bib_path)
            if citations_bib_path.exists()
            else None,
        },
        "brief_text": brief_text,
        "elapsed": elapsed,
        "results_dir": str(results_dir),
    }


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
            json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        log.warning("Failed to update research index: %s", e)
