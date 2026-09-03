"""Research configuration — user-selectable settings for the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ResearchConfig:
    """Configuration for a research pipeline run.

    Core settings are on the compact UI bar. Advanced settings are in
    the ⚙ Advanced Settings dialog.
    """

    # ── Core (compact UI bar) ────────────────────────────────────────
    query: str = ""
    sources: list[str] = field(
        default_factory=lambda: ["web_search", "crossref", "openalex", "s2", "eartharxiv", "usgs"]
    )
    max_papers: int = 30
    use_llm: bool = False
    match_local_pdfs: bool = True
    research_type_override: str | None = None
    skip_verify: bool = False
    skip_correlate: bool = False

    # ── Directories (⚙ Advanced) ────────────────────────────────────
    fulltext_dir: Path | None = None
    output_dir: Path | None = None

    # ── LLM (⚙ Advanced) ────────────────────────────────────────────
    llm_num_ctx: int = 4096
    """Ollama context window for extraction. Larger = better but more VRAM."""
    llm_num_predict: int = 1024
    """Max output tokens for LLM extraction."""
    llm_model: str = ""
    """Specific Ollama model to use (empty = auto-detect smallest)."""
    llm_quality: str = "balanced"
    """Model selection policy: 'fast' | 'balanced' | 'quality'.

    - fast: smallest suitable model for ALL tasks (speed priority, lower quality)
    - balanced: smallest suitable per task tier (simple→smallest, moderate→smallest-suitable)
    - quality: largest suitable for ALL tasks (quality priority, slower)
    """
    llm_batch_size: int = 0
    """Papers per LLM extraction call (0 = disabled, sequential per-paper).

    When >0 and use_llm=True, the pipeline batches papers into groups of
    this size and sends one LLM call per batch. Falls back to individual
    extraction if a batch fails to parse. Default 0 = sequential, which
    avoids overwhelming Ollama and produces more stable results.
    """

    # ── Pipeline tuning (⚙ Advanced) ────────────────────────────────
    api_rate_limit_s: float = 0.0
    """Delay between sequential API calls (S2, OpenAlex, Crossref DOI lookup).

    Prevents rate-limit 429s by spacing requests. 1.0s = safe for all APIs.
    Set to 0 for no delay (parallel mode — will trigger rate limits).
    """
    ollama_delay_s: float = 0.5
    """Delay between sequential Ollama LLM calls.

    Prevents Ollama from crashing under memory pressure when processing
    many papers. 0.5s gives Ollama time to free GPU memory between calls.
    """
    search_timeout: int = 30
    """Per-source timeout for web search (seconds)."""
    use_web_search: bool = True
    """Web search (web-search skill CLI: 8+ metasearch backends,
    bot-block bypass, rate-limit defense) as a PRIMARY discovery source
    alongside scholarly APIs. Finds papers Crossref/OpenAlex/S2 miss:
    conference papers, niche journals, preprints on repository pages."""
    use_web_search_agentic: bool = False
    """Deep discovery: additionally extract top result pages (trafilatura)
    and mine DOIs from page content. Slower but higher precision."""
    auto_web_search_threshold: int = 5
    """Auto-enable web_search supplement when discovery yields fewer than this
    many papers. Catches sparse topics where Crossref/OpenAlex/S2 miss the
    field. Set to 0 to disable auto-supplement. Default 5."""
    use_web_pro: bool = True
    reflect_iterations: int = 1
    """Agentic reflect loop (2026-09-01): after synthesis, an LLM critic
    judges coverage; "insufficient" triggers one bounded re-discovery +
    re-synthesis round per iteration (follow-up queries from the critique).
    0 disables. Skipped loudly when Ollama is unreachable."""
    adaptive_questions: bool = True
    ocr_model: str = "chandra-ocr-2"
    """Vision model for the pdf-ocr engine (full-text library parsing).
    Must match an `ollama list` name. When unavailable, the run warns
    with a pull suggestion and falls back to the first available
    vision-capable model."""
    """Generate PaperQA2 questions adaptively from the corpus (LLM) instead
    of the fixed methods/quantitative/disagreements lenses. Template lenses
    remain the explicit fallback (emitted when generation fails)."""
    """Append a Perplexity-style web synthesis (decompose → search →
    coverage → cited markdown, native ddgs+trafilatura) to the brief as a
    clearly-labeled supplementary section. URLs, not DOIs — advisory
    supplement around the verified corpus core."""
    verify_workers: int = 8
    """Thread pool size for parallel paper verification."""
    year_from: int | None = None
    """Earliest publication year (None = no limit)."""
    year_to: int | None = None
    """Latest publication year (None = no limit)."""
    open_access_only: bool = False
    """Only discover open access papers."""
    max_pdf_downloads: int = 10
    """Max OA PDFs to download + extract per pipeline run.

    Controls full-text enrichment depth. Each download takes up to 15s
    (timeout), with 0.5s pacing between downloads. KB-cached papers
    skip download entirely (0s). Higher = better extraction but slower.
    """
    publication_type: str = ""
    """Filter by type: journal-article, book-chapter, conference-paper, etc."""

    abstract_cap: int = 50000
    """Max chars of abstract to process for extraction."""
    fulltext_cap: int = 200000
    """Max chars of full-text to process for extraction."""

    # ── Time budgets (⚙ Advanced) ──────────────────────────────────
    wall_clock_budget_s: float | None = None
    """Total pipeline wall-clock budget in seconds. None = unbounded.

    Checked between phases. On expiry, raises PipelineTimeout and the
    caller receives a partial result built from whatever phases completed.
    Bounds the worst case for interactive (chat-tool) use so a flaky
    network or slow LLM cannot hang the pipeline indefinitely.
    """
    llm_time_budget_s: float | None = None
    """Per-pipeline LLM extraction budget in seconds (Phase 3 only).

    Checked inside the extraction loop. On expiry, remaining papers are
    extracted via the fast non-LLM path (regex/lexicon) so synthesis is
    never blocked on LLM availability. None = unbounded.
    """
    paperqa_timeout_s: int = 600
    """Per-call LLM timeout for PaperQA2 synthesis (seconds).

    Each LiteLLM/Ollama call during evidence gathering + answer generation
    gets this many seconds before timing out. Increase for slow models
    (9B+), decrease for fast models (3B). Default 600s = 10min.
    """

    synthesis_tier: str = "auto"
    """Which synthesis engine to use for the research brief.

    Options:
        "auto"     — Try Ollama direct → PaperQA2 → template (default, best quality)
        "ollama"   — Ollama direct synthesis only (fastest, ~60s, structured data)
        "paperqa"  — PaperQA2 RAG only (~600s, evidence-grounded, needs full-text)
        "template" — Template synthesis only (instant, no LLM, lowest quality)

    Use "ollama" for speed, "paperqa" for interactive Q&A, "auto" for best results.
    """

    synthesis_quality: str = "thorough"
    """Depth of Ollama synthesis passes.

    Options:
        "fast"     — Single pass (60s, basic synthesis)
        "thorough" — 3-pass: themes + sections + self-review (140s, research-grade)
        "maximum"  — 4-pass: adds numerical value verification (160s, publication-grade)
    """

    # ── Timeouts (⚙ Advanced) ─────────────────────────────────────────
    doi_lookup_timeout: int = 15
    """Per-request timeout for DOI resolution (Crossref, Unpaywall)."""
    semantic_search_timeout: int = 10
    """Per-request timeout for Semantic Scholar / OpenAlex search."""
    wikidata_timeout: int = 30
    """Per-request timeout for Wikidata SPARQL queries."""
    wikipedia_timeout: int = 5
    """Per-request timeout for Wikipedia REST API."""
    ollama_health_timeout: int = 3
    """Timeout for Ollama /api/tags health check."""
    ollama_extract_timeout: int = 60
    """Per-call timeout for Ollama LLM extraction."""
    ollama_synthesis_timeout: int = 180
    """Per-call timeout for Ollama LLM synthesis (longer, multi-pass)."""
    pdf_download_timeout: int = 30
    """Per-file timeout for PDF download."""

    # ── Internal (not user-facing) ───────────────────────────────────
    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("query must not be empty")
        if self.max_papers < 1:
            raise ValueError("max_papers must be >= 1")
        if self.fulltext_dir is None:
            self.fulltext_dir = Path("data/papers")

    @property
    def query_hash(self) -> str:
        """Stable hash of the query for directory naming."""
        import hashlib

        return hashlib.sha256(self.query.encode()).hexdigest()[:16]

    @property
    def results_dir(self) -> Path:
        """Directory where pipeline outputs are saved."""
        if self.output_dir is not None:
            return self.output_dir
        return Path("data/research") / self.query_hash
