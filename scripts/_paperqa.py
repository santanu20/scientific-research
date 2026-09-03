"""PaperQA2 comprehensive configuration for the research skill.

Maximizes PaperQA2's internal features — no custom search/evidence code.
All behavior is controlled through Settings configuration.

Features utilized:
    - LiteLLM rate limiting (replaces custom circuit breakers for LLM calls)
    - Persistent Tantivy index (build once, reuse across queries)
    - Configurable chunking (tuned for geological papers with P-T tables)
    - Parallel evidence gathering (max_concurrent_requests)
    - Geological domain system prompt
    - BibTeX export from cited papers
    - Index reuse for multiple queries

Presets:
    get_settings()         — balanced (default)
    get_research_settings() — comprehensive (Phase 5 synthesis: long briefs)
    get_chat_settings()     — quick (chat Q&A: short cited answers)

Usage:
    from _paperqa import ask, get_research_settings
    result = ask("What P-T conditions?", settings=get_research_settings())
    print(result.session.answer)    # cited answer
    print(result.bibtex)            # BibTeX dict
    print(result.session.contexts)  # evidence passages

Reference: PaperQA2 v2026.3 (Apache-2.0, Future-House)
    Docs: https://github.com/future-house/paper-qa
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

from _timeouts import TIMEOUTS  # noqa: E402

log = logging.getLogger(__name__)

# Suppress LiteLLM cost-map warnings at MODULE LEVEL — fires on import,
# before any function call. Ollama models aren't in LiteLLM's cost DB,
# so the remote fetch + "not in built-in cost map" warnings are noise.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Proxy").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

# Set GLOBAL LiteLLM timeout — per-model request_timeout in router config
# doesn't reliably override the 60s default for Ollama. This ensures all
# LiteLLM calls (including PaperQA2's internal evidence gathering) get
# the full timeout. Configurable per-run via _build_settings(timeout=).
try:
    import litellm

    litellm.request_timeout = TIMEOUTS.ollama_synthesis * 3  # paperqa is multi-call
except ImportError:
    log.debug("Optional feature failed in _paperqa.py", exc_info=True)

# ── Model defaults (overridable via environment) ──────────────────────
_DEFAULT_LLM = os.environ.get("GEOKIT_LLM_MODEL", "")
_DEFAULT_EMBEDDING = os.environ.get("GEOKIT_EMBEDDING_MODEL", "ollama/nomic-embed-text")
_KB_DIR = Path.home() / ".local" / "share" / "scientific-research" / "papers"

# ── Geological domain system prompt ───────────────────────────────────
_GEO_SYSTEM_PROMPT = (
    "You are a research assistant specializing in geology, geochemistry, "
    "petrology, and structural geology. Provide precise, quantitative "
    "answers with pressure-temperature conditions, mineral assemblages, "
    "geochemical data, and methodological details. Always cite sources "
    "using the provided citation format."
)


# Module-level regex + robust score extractor (used by patch function).
_MODULE_THINK_RE = re.compile(
    r"<(?:think|thinking|reasoning)\b[^>]*>.*?</(?:think|thinking|reasoning)>",
    re.DOTALL | re.IGNORECASE,
)
_MODULE_SCORE_RE = re.compile(r"(?:^|\n)\s*(\d{1,2})\s*(?:$|\n)", re.MULTILINE)


def _extract_score_robust(text: str) -> int:
    """Robust score extraction replacing PaperQA2's extract_score.

    Handles thinking models that produce thousands of chars of reasoning
    before the actual score. Finds the LAST standalone integer 1-10.
    """
    if not text:
        raise ValueError("Empty text")
    clean = _MODULE_THINK_RE.sub("", text)
    matches = list(_MODULE_SCORE_RE.finditer(clean))
    if not matches:
        nums = re.findall(r"\b(\d{1,2})\b", clean)
        for n in reversed(nums):
            val = int(n)
            if 1 <= val <= 10:
                return val
        raise ValueError(f"No score (1-10) found in text ({len(clean)} chars)")
    score = int(matches[-1].group(1))
    if not (1 <= score <= 10):
        raise ValueError(f"Score {score} out of range 1-10")
    return score


def _patch_paperqa_thinking():
    """Monkey-patch PaperQA2 + LiteLLM to strip <think> blocks from model responses.

    qwythos3.5 and similar community GGUF models embed <think>...</think>
    blocks directly in the content field, ignoring Ollama's ``think=False``
    parameter (see _thinking_filter.py docs). PaperQA2's internal
    ``extract_score()`` crashes when trying to parse a numerical score
    from thinking-contaminated text.

    Two-layer defense:
    1. Patch ``litellm.acompletion`` — strip thinking from raw LLM response
       BEFORE PaperQA2 processes it. Fixes both score extraction AND
       evidence summary quality.
    2. Patch ``paperqa.utils.extract_score`` — strip thinking as a fallback
       safety net in case layer 1 misses something.

    Must be called BEFORE any PaperQA2 query — applied at module import.
    """
    try:

        _THINK_RE = _MODULE_THINK_RE  # use module-level compiled regex

        # ── Layer 1: Patch litellm.acompletion ──
        import litellm

        if not getattr(litellm, "_thinking_stripped", False):
            _orig_acompletion = litellm.acompletion

            async def _thinking_safe_acompletion(*args, **kwargs):
                resp = await _orig_acompletion(*args, **kwargs)
                try:
                    for choice in getattr(resp, "choices", []) or []:
                        msg = getattr(choice, "message", None)
                        if msg and getattr(msg, "content", None):
                            msg.content = _THINK_RE.sub("", msg.content).strip()
                except Exception:
                    log.debug("Optional feature failed in _paperqa.py", exc_info=True)
                return resp

            litellm.acompletion = _thinking_safe_acompletion
            litellm._thinking_stripped = True
            log.info("LiteLLM acompletion patched: <think> blocks will be stripped")

        # ── Layer 2: Replace extract_score entirely ──
        # PaperQA2's original extract_score expects a clean summary + score
        # format. Our thinking models (qwythos3.5) produce THOUSANDS of chars
        # of reasoning before the actual summary+score. Whether wrapped in
        # <think> tags or not (retry path omits tags), the original parser
        # fails. Our replacement:
        #   1. Strip <think>/<thinking> tags if present
        #   2. Find the LAST standalone integer 1-10 in the text
        #   3. Use everything before that integer as the summary context
        # This works for ALL model output formats: clean, <think>-wrapped,
        # or raw reasoning without tags.
        import paperqa.core as _pqa_core
        import paperqa.utils as _pqa_utils

        for _mod in (_pqa_utils, _pqa_core):
            if not getattr(_mod, "_extract_replaced", False):
                _mod.extract_score = _extract_score_robust
                _mod._extract_replaced = True
        log.info("PaperQA2 extract_score replaced with robust parser (utils + core)")
    except Exception as e:
        log.warning("Failed to patch paperqa/litellm for thinking: %s", e)


# Apply patch at import — before any PaperQA2 query runs.
_patch_paperqa_thinking()


# =============================================================================
# Settings builder
# =============================================================================


def _build_settings(
    *,
    paper_dir: Path | str = _KB_DIR,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    temperature: float = 0.0,
    max_sources: int = 10,
    evidence_k: int = 15,
    answer_length: str = "about 200 words, but can be longer",
    max_concurrent: int = 4,
    chunk_chars: int = 5000,
    overlap: int = 500,
    timeout: int = 600,
):
    """Build PaperQA2 Settings with comprehensive configuration.

    All PaperQA2 internal features are configured here — no custom code
    needed outside this function.
    """
    from paperqa import Settings
    from paperqa.settings import AgentSettings

    # Set global LiteLLM timeout for this run (per-model request_timeout
    # in router config isn't reliably picked up by fhlmi/paperqa internals)
    try:
        import litellm

        litellm.request_timeout = timeout
    except Exception:
        log.debug("Optional feature failed in _paperqa.py", exc_info=True)

    # Pass agent explicitly — paperqa.Settings extends pydantic_settings.
    # BaseSettings, which has no env_prefix, so it reads ALL field names
    # from the shell environment. On systems with AGENT set (gpg-agent,
    # ssh-agent, etc.), the unprefixed `agent` field picks up AGENT=1
    # and fails pydantic validation. Explicit init_kwargs take precedence
    # over env sources in pydantic-settings v2, so this neutralizes the
    # collision. Other 14 fields were audited and have no env collisions.
    s = Settings(
        llm=llm,
        summary_llm=llm,
        embedding=embedding,
        temperature=temperature,
        batch_size=1,
        verbosity=0,
        agent=AgentSettings(),
    )

    # ── LiteLLM rate limiting ──
    # Ollama handles its own request queueing — no external rate limit needed.
    # Setting high limits to prevent LiteLLM's GLOBAL_LIMITER from timing out
    # during bulk indexing (275+ papers = 275+ metadata inference calls).
    # CRITICAL: pass think=False via extra_body to disable thinking output
    # (same approach as chat_dock _chat_worker.py line 246). Without this,
    # thinking models (qwen3, qwythos) output <think> blocks that break
    # PaperQA2's JSON score extraction.
    s.llm_config = {
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {
                    "model": llm,
                    "api_base": "http://127.0.0.1:11434",
                    "temperature": temperature,
                    "extra_body": {"think": False, "num_ctx": 32768},
                    "request_timeout": timeout,
                },
            }
        ],
    }
    s.summary_llm_config = {}

    # ── Embedding config (no rate limit — Ollama queues internally) ──
    s.embedding_config = {
        "model_list": [
            {
                "model_name": embedding,
                "litellm_params": {
                    "model": embedding,
                    "api_base": "http://127.0.0.1:11434",
                    "extra_body": {"think": False},
                },
            }
        ],
    }

    # ── Answer generation ──
    s.answer.evidence_k = evidence_k
    s.answer.evidence_retrieval = True
    s.answer.evidence_summary_length = "about 150 words"
    s.answer.answer_max_sources = max_sources
    s.answer.answer_length = answer_length
    s.answer.max_concurrent_requests = max_concurrent

    # ── Parsing — tuned for geological papers ──
    # Larger chunks (5000 chars) capture P-T tables and mineral chemistry
    # data that span multiple lines. Overlap (500) prevents splitting
    # data tables mid-row.
    s.parsing.reader_config = {"chunk_chars": chunk_chars, "overlap": overlap}
    # Skip LLM metadata inference during indexing — use filename as citation.
    # Makes indexing 10x+ faster (no per-paper LLM call). Citation details
    # are already in the .md file headers from knowledge_base.py.
    s.parsing.use_doc_details = False
    s.parsing.disable_doc_valid_check = True

    # ── Prompts — geological domain ──
    s.prompts.system = _GEO_SYSTEM_PROMPT
    s.prompts.use_json = False  # thinking models don't produce clean JSON scores

    # ── Agent — persistent index, local papers only ──
    paper_dir = Path(paper_dir)
    s.agent.index.paper_directory = paper_dir
    s.agent.index.index_directory = paper_dir / "paperqa_index"
    s.agent.index.manifest_file = paper_dir / "paperqa_manifest.json"
    s.agent.search_count = 5  # papers to retrieve from local index per query
    s.agent.timeout = timeout
    # CRITICAL: agent uses its OWN LLM for tool selection — defaults to
    # gpt-4o. Must override to Ollama or it crashes on missing API key.
    s.agent.agent_llm = llm
    s.agent.agent_llm_config = {
        "model_list": [
            {
                "model_name": llm,
                "litellm_params": {
                    "model": llm,
                    "api_base": "http://127.0.0.1:11434",
                    "extra_body": {"think": False},
                    "request_timeout": timeout,
                },
            },
        ],
    }

    return s


def get_settings(
    *,
    paper_dir: Path | str = _KB_DIR,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    max_sources: int = 10,
    answer_length: str = "about 200 words, but can be longer",
):
    """Balanced Settings — default for chat Q&A and quick lookups."""
    return _build_settings(
        paper_dir=paper_dir,
        llm=llm,
        embedding=embedding,
        max_sources=max_sources,
        answer_length=answer_length,
    )


def get_research_settings(
    *,
    paper_dir: Path | str = _KB_DIR,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    max_sources: int = 20,
):
    """Comprehensive Settings — Phase 5 synthesis (long cited research briefs).

    More evidence pieces (25), longer answers (~2000 words), more sources,
    and larger chunks to capture full methodological details.
    """
    return _build_settings(
        paper_dir=paper_dir,
        llm=llm,
        embedding=embedding,
        max_sources=max_sources,
        evidence_k=25,
        answer_length="about 2000 words, but can be longer",
        max_concurrent=6,
        chunk_chars=8000,
        overlap=800,
        timeout=TIMEOUTS.ollama_synthesis * 3,
    )


def get_chat_settings(
    *,
    paper_dir: Path | str = _KB_DIR,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    max_sources: int = 5,
):
    """Quick Settings — chat dock Q&A (short cited answers, fast).

    Fewer evidence pieces (5), shorter answers (~300 words), fewer sources.
    Optimized for interactive responsiveness.
    """
    return _build_settings(
        paper_dir=paper_dir,
        llm=llm,
        embedding=embedding,
        max_sources=max_sources,
        evidence_k=5,
        answer_length="about 300 words",
        max_concurrent=4,
        chunk_chars=3000,
        overlap=300,
        timeout=TIMEOUTS.ollama_synthesis,
    )


def ask(question: str, settings=None, **kwargs):
    """Thin passthrough to paperqa.ask — cited answer from papers.

    Uses PaperQA2 agent pipeline (SearchPapers → GatherEvidence → GenerateAnswer).
    For simpler direct-query over known papers, use ``docs_query()`` instead.

    Returns ``AnswerResponse`` with:
        result.session.answer    — cited answer text
        result.session.contexts  — evidence passages with scores
        result.bibtex            — {key: bibtex_str} of cited papers
        result.duration          — wall clock seconds
        result.status            — AgentStatus enum
    """
    from paperqa import ask as _ask

    if settings is None:
        settings = get_settings(**kwargs)
    return _ask(question, settings=settings)


async def _docs_query_async(question: str, paper_dir: str, settings):
    """Build Docs from a directory and query directly (no agent pipeline).

    Simpler and more reliable than ``ask()`` — indexes files, gathers
    evidence, and generates answer in one step. No external API search.
    """
    from paperqa import Docs

    docs = Docs()
    pdir = Path(paper_dir)
    files = sorted(list(pdir.glob("*.txt")) + list(pdir.glob("*.md")))
    for f in files:
        try:
            await docs.aadd(str(f), settings=settings)
        except Exception as e:
            log.debug("PaperQA2: skip %s: %s", f.name, e)
    log.info("PaperQA2: indexed %d/%d files", len(docs.docs), len(files))
    return await docs.aquery(question, settings=settings)


def docs_query(
    question: str,
    paper_dir: str,
    *,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    max_sources: int = 10,
    answer_length: str = "about 200 words, but can be longer",
    timeout: int = 600,
    evidence_k: int = 10,
    chunk_chars: int = 4000,
) -> dict:
    """Direct Docs query — index papers + generate cited answer.

    Simpler than ``ask()`` — no agent, no external search. Just indexes
    the files in ``paper_dir``, gathers evidence, and generates an answer.

    Parameters
    ----------
    question
        Question to answer from the papers.
    paper_dir
        Directory containing .txt/.md/.pdf files to index.
    max_sources
        Maximum papers to cite in the answer.
    answer_length
        Prompt hint for answer length.
    timeout
        Maximum seconds for the entire operation (default 600 = 10min).
    evidence_k
        Number of evidence passages to retrieve (lower = faster, noisier).
    chunk_chars
        Characters per text chunk (lower = better for short abstracts).

    Returns dict with:
        answer    — cited answer text
        contexts  — list of evidence passages
        n_papers  — number of papers indexed
    """
    import asyncio

    settings = _build_settings(
        paper_dir=paper_dir,
        llm=llm,
        embedding=embedding,
        max_sources=max_sources,
        answer_length=answer_length,
        evidence_k=evidence_k,
        chunk_chars=chunk_chars,
    )
    try:

        async def _run_with_cleanup():
            try:
                return await asyncio.wait_for(
                    _docs_query_async(question, paper_dir, settings),
                    timeout=timeout,
                )
            finally:
                # Cancel pending aiohttp/httpx tasks before loop tears down
                tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

        result = asyncio.run(_run_with_cleanup())
    except TimeoutError:
        log.warning("PaperQA2 docs_query timed out after %ds", timeout)
        return {"answer": f"PaperQA2 timed out after {timeout}s.", "contexts": [], "n_papers": 0}
    except Exception as e:
        log.warning("PaperQA2 docs_query failed: %s", e)
        return {"answer": f"Q&A failed: {e}", "contexts": [], "n_papers": 0}
    # Safety net: strip inline <think>/<thinking>/<reasoning> blocks from
    # answer (some models embed thinking in content despite think=False).
    # Same filter as chat_dock's strip_inline_thinking().
    answer = result.answer
    try:
        import sys as _sys

        _gui_path = str(Path(__file__).resolve().parents[3] / "gui" / "docks" / "chat_tools")
        if _gui_path not in _sys.path:
            _sys.path.insert(0, _gui_path)
        from _thinking_filter import strip_inline_thinking

        answer = strip_inline_thinking(answer)
    except ImportError:
        log.debug("strip_inline_thinking unavailable — returning raw answer")
    return {
        "answer": answer,
        "contexts": [
            {
                "text": str(ctx.text)[:300],
                "score": float(ctx.score) if hasattr(ctx, "score") else 0.0,
                "doc": ctx.doc.docname if hasattr(ctx, "doc") else "",
            }
            for ctx in (result.contexts or [])
        ],
        "n_papers": len(result.contexts) if result.contexts else 0,
    }


def docs_query_multi(
    questions: list[str],
    paper_dir: str,
    *,
    llm: str = _DEFAULT_LLM,
    embedding: str = _DEFAULT_EMBEDDING,
    max_sources: int = 10,
    answer_length: str = "about 600 words",
    timeout: int = 600,
    evidence_k: int = 6,
    chunk_chars: int = 2000,
) -> list[dict]:
    """Index papers ONCE, then answer multiple focused questions.

    Avoids the 3x indexing overhead of calling docs_query() separately
    for each question. The embedding/indexing phase (~30-60s) runs only
    once; subsequent queries reuse the same Docs index.

    Returns list of dicts (same shape as docs_query), one per question,
    in the same order as ``questions``.
    """
    import asyncio

    settings = _build_settings(
        paper_dir=paper_dir,
        llm=llm,
        embedding=embedding,
        max_sources=max_sources,
        answer_length=answer_length,
        evidence_k=evidence_k,
        chunk_chars=chunk_chars,
    )

    async def _run_multi():
        from paperqa import Docs

        _t0 = __import__("time").time

        # Phase A: Indexing (embedding + vector store)
        docs = Docs()
        pdir = Path(paper_dir)
        files = sorted(list(pdir.glob("*.txt")) + list(pdir.glob("*.md")))
        _t_index = _t0()
        for f in files:
            try:
                await docs.aadd(str(f), settings=settings)
            except Exception as e:
                log.debug("PaperQA2: skip %s: %s", f.name, e)
        _index_elapsed = _t0() - _t_index
        log.info(
            "PaperQA2 multi: indexed %d/%d files in %.1fs (%.1fs/file)",
            len(docs.docs),
            len(files),
            _index_elapsed,
            _index_elapsed / max(len(files), 1),
        )

        # Phase B: Evidence gathering + answer generation (per query)
        results = []
        per_query_timeout = max(60, timeout // max(len(questions), 1))
        for i, q in enumerate(questions):
            _t_query = _t0()
            try:
                answer_result = await asyncio.wait_for(
                    docs.aquery(q, settings=settings),
                    timeout=per_query_timeout,
                )
                _query_elapsed = _t0() - _t_query
                answer_text = answer_result.answer
                n_contexts = len(answer_result.contexts) if answer_result.contexts else 0
                log.info(
                    "PaperQA2 query %d/%d: %.1fs (%d contexts, %d chars answer)",
                    i + 1,
                    len(questions),
                    _query_elapsed,
                    n_contexts,
                    len(answer_text),
                )
                # Strip inline thinking blocks
                try:
                    import sys as _sys

                    _gui_path = str(
                        Path(__file__).resolve().parents[3] / "gui" / "docks" / "chat_tools"
                    )
                    if _gui_path not in _sys.path:
                        _sys.path.insert(0, _gui_path)
                    from _thinking_filter import strip_inline_thinking

                    answer_text = strip_inline_thinking(answer_text)
                except ImportError:
                    log.debug("Optional feature failed in _paperqa.py", exc_info=True)
                results.append(
                    {
                        "answer": answer_text,
                        "contexts": [
                            {
                                "text": str(ctx.text)[:300],
                                "doc": ctx.doc.docname if hasattr(ctx, "doc") else "",
                            }
                            for ctx in (answer_result.contexts or [])
                        ],
                        "n_papers": len(answer_result.contexts) if answer_result.contexts else 0,
                        "error": None,
                    }
                )
                log.info(
                    "PaperQA2 multi query %d/%d: %d chars", i + 1, len(questions), len(answer_text)
                )
            except TimeoutError:
                log.warning(
                    "PaperQA2 multi query %d/%d timed out (%ds)",
                    i + 1,
                    len(questions),
                    per_query_timeout,
                )
                results.append(
                    {
                        "answer": "",
                        "contexts": [],
                        "n_papers": 0,
                        "error": f"timeout ({per_query_timeout}s)",
                    }
                )
            except Exception as e:
                log.warning("PaperQA2 multi query %d/%d failed: %s", i + 1, len(questions), e)
                results.append({"answer": "", "contexts": [], "n_papers": 0, "error": str(e)})
        return results

    try:
        return asyncio.run(_run_multi())
    except Exception as e:
        log.warning("PaperQA2 docs_query_multi failed: %s", e)
        return [{"answer": "", "contexts": [], "n_papers": 0, "error": str(e)} for _ in questions]


def is_available() -> bool:
    """Check PaperQA2 + Ollama availability."""
    try:
        import urllib.request

        import paperqa  # noqa: F401

        urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=TIMEOUTS.ollama_health)
        return True
    except Exception:
        log.debug("_paperqa.py:674 — feature degraded")
        return False
