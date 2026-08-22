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
    sources: list[str] = field(default_factory=lambda: ["crossref", "openalex", "s2", "eartharxiv", "usgs"])
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
    use_web_search: bool = False
    """Add web-search skill as a discovery source (9 metasearch backends +
    4-layer bot-block bypass + per-URL extract cache). Finds papers
    Crossref/OpenAlex/S2 miss: conference papers, niche journals, preprints."""
    use_web_search_agentic: bool = False
    """Deep-research mode: web-search agentic --adaptive crawls top hit with
    Crawl4AI semantic filter (confidence-scored). Slower (30-120s) but finds
    papers in JS-rendered sites and behind search forms."""
    auto_web_search_threshold: int = 5
    """Auto-enable web_search supplement when discovery yields fewer than this
    many papers. Catches sparse topics where Crossref/OpenAlex/S2 miss the
    field. Set to 0 to disable auto-supplement. Default 5."""
    ollama_delay_s: float = 0.5
    """Delay between sequential Ollama LLM calls.

    Prevents Ollama from crashing under memory pressure when processing
    many papers. 0.5s gives Ollama time to free GPU memory between calls.
    """
    search_timeout: int = 30
    """Per-source timeout for web search (seconds)."""
    abstract_cap: int | None = None  # None = no cap (zero-truncation policy 2026-08-15)
    """Max chars of abstract to process for extraction."""
    fulltext_cap: int | None = None  # None = no cap (lossless; downstream chunks)
    """Max chars of full-text to process for extraction."""
    verify_workers: int = 8
    """Thread pool size for parallel paper verification."""
    year_from: int | None = None
    """Earliest publication year (None = no limit)."""
    year_to: int | None = None
    """Latest publication year (None = no limit)."""
    open_access_only: bool = False
    """Only discover open access papers."""
    publication_type: str = ""
    """Filter by type: journal-article, book-chapter, conference-paper, etc."""
    screen: bool = True
    """PRISMA Phase-2 keyword triage after discovery (non-LLM).

    Include terms are derived dynamically from the query itself (no field
    vocabulary); papers sharing zero query terms are excluded. A sparse
    result (<3 survivors) skips screening rather than shrinking the corpus.
    Set False to disable."""

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
