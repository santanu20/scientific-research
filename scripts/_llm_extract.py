#!/usr/bin/env python3
"""LLM-powered extraction for scientific-research skill.

Auto-detects available Ollama models, picks the smallest that can do JSON
extraction, and uses it for:
    - PICO extraction (population, intervention, comparator, outcome)
    - Effect size extraction (paired groups: mean, SD, n)
    - Study design detection
    - Abbreviation extraction
    - Citation stance classification
    - Key finding extraction

If Ollama is unavailable, falls back to regex/lexicon (_nlp.py).

Model selection:
    1. Check SCIENTIFIC_RESEARCH_LLM_MODEL env var
    2. Query /api/tags for available models
    3. Filter for instruction-following models (qwen, llama, mistral, gemma)
    4. Pick smallest by parameter count
    5. If JSON parsing fails on test, try next larger model

Caching:
    SHA256(abstract + model_name + prompt_version) → cached JSON
    Avoids re-extraction on re-runs.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

log = logging.getLogger("scientific_research.llm_extract")

# Cache directory
_CACHE_DIR = (
    Path(
        os.environ.get(
            "SCIENTIFIC_RESEARCH_CACHE",
            str(Path.home() / ".cache" / "scientific_research"),
        )
    )
    / "llm_extract"
)
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

OLLAMA_URL = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")

# Ollama circuit breaker — prevents warning flood when Ollama crashes under
# load (common during Phase 4 stance detection with many papers). After
# _OLLAMA_CIRCUIT_THRESHOLD consecutive connection failures, all subsequent
# calls skip the HTTP round-trip and immediately fall back.
_ollama_consecutive_failures: int = 0
_ollama_circuit_open: bool = False
_OLLAMA_CIRCUIT_THRESHOLD: int = 3


def reset_ollama_circuit() -> None:
    """Reset the Ollama circuit breaker for a new pipeline run.

    Called at the start of each ``run_pipeline()`` so a transient Ollama
    crash in one topic doesn't cascade to all subsequent topics.
    """
    global _ollama_consecutive_failures, _ollama_circuit_open
    if _ollama_circuit_open:
        log.info("Ollama circuit breaker reset for new pipeline run")
    _ollama_consecutive_failures = 0
    _ollama_circuit_open = False


# Prompt version — bump when prompt changes to invalidate cache
_PROMPT_VERSION = "v3.0"

# Models known to be good at instruction-following / JSON
_KNOWN_GOOD_PREFIXES = (
    "qwen",
    "llama",
    "mistral",
    "gemma",
    "phi",
    "command",
    "deepseek",
    "yi",
    "orca",
    "dolphin",
    "codellama",
)

# Name patterns that indicate non-text-generation models
_EXCLUDE_PATTERNS = (
    "embed",  # embedding models (nomic-embed, mxbai-embed, etc.)
    "ocr",  # OCR/vision models (glm-ocr, etc.)
    "vision",  # vision-language models
    "vl",  # vision-language (e.g., qwen2-vl, minicpm-v)
    "clip",  # CLIP models
    "image",  # image generation
    "llava",  # vision-language
    "bakllava",  # vision-language
    "moondream",  # vision-language
    "minicpm-v",  # vision-language
    "coder",  # code-only models (may work but not optimized for extraction)
)

# Minimum parameter sizes (in billions) per task complexity
# Smaller models can do simple tasks; complex tasks need more capacity
_MIN_PARAMS = {
    "simple": 2.0,  # stance classification — use 2b (0.8b too small for JSON)
    "moderate": 1.5,  # PICO + effect size extraction — needs JSON following
}

# Ollama context window per task (tokens)
_NUM_CTX = {
    "simple": 2048,
    "moderate": 4096,  # extraction prompt (~1K tokens) + JSON output (~500 tokens)
}

_NUM_PREDICT = {
    "simple": 256,
    "moderate": 1024,  # JSON extraction output
}


def _is_text_generation_model(name: str, details: dict) -> bool:
    """Check if a model is a text-generation model (not vision/embedding).

    Checks:
    1. Name against exclude patterns
    2. details.capabilities (newer Ollama versions)
    3. details.family for known vision/embedding families
    """
    name_lower = name.lower()

    # Check exclude patterns
    for pattern in _EXCLUDE_PATTERNS:
        if pattern in name_lower:
            return False

    # Check capabilities (Ollama 0.5+)
    capabilities = details.get("capabilities", [])
    if capabilities:
        # If model only has embedding capability, exclude
        if "embedding" in capabilities and "completion" not in capabilities:
            return False
        # If model has vision but not completion, exclude
        if "vision" in capabilities and "completion" not in capabilities:
            return False
        # Must have completion or tools capability
        if "completion" not in capabilities and "tools" not in capabilities:
            return False

    # Check family for known non-text families
    family = (details.get("family") or "").lower()
    non_text_families = {"nomic-bert", "bert", "clip", "t5-vision"}
    if family in non_text_families:
        return False

    # Check if it's a known good text model
    is_good = any(name_lower.startswith(p) for p in _KNOWN_GOOD_PREFIXES)

    return is_good


def _detect_models() -> list[dict[str, Any]]:
    """Query Ollama /api/tags for available text-generation models.

    Excludes vision, embedding, OCR, and image models.
    Returns list sorted by parameter size (smallest first).
    """
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        log.debug("Ollama not available: %s", e)
        return []

    models = []
    for m in data.get("models", []):
        name = m.get("name", "")
        size = m.get("size", 0)
        details = m.get("details", {})
        param_str = details.get("parameter_size", "0B")

        # Parse parameter size (e.g., "9.7B" → 9.7, "137M" → 0.137)
        try:
            if param_str.endswith("B"):
                param_gb = float(param_str[:-1])
            elif param_str.endswith("M"):
                param_gb = float(param_str[:-1]) / 1000
            else:
                param_gb = 0.0
        except ValueError:
            param_gb = 0.0

        if not _is_text_generation_model(name, details):
            log.debug(
                "Excluding non-text model: %s (family=%s, caps=%s)",
                name,
                details.get("family", "?"),
                details.get("capabilities", []),
            )
            continue

        models.append(
            {
                "name": name,
                "size_bytes": size,
                "param_gb": param_gb,
                "quantization": details.get("quantization_level", ""),
                "family": details.get("family", ""),
                "capabilities": details.get("capabilities", []),
            }
        )

    # Sort by parameter size (smallest first)
    models.sort(key=lambda m: m["param_gb"])
    return models


def _pick_model(task: str = "moderate") -> str | None:
    """Pick the smallest available model suitable for the task.

    Priority:
    1. Manual override (set via set_model_override())
    2. SCIENTIFIC_RESEARCH_LLM_MODEL env var
    3. Smallest model meeting the minimum param requirement
    """
    # Check manual override
    if _manual_model:
        return _manual_model

    # Check env var override
    env_model = os.environ.get("SCIENTIFIC_RESEARCH_LLM_MODEL")
    if env_model:
        log.info("Using model from env var: %s", env_model)
        return env_model

    models = _detect_models()
    if not models:
        log.warning(
            "No Ollama text-generation models available — will fall back to regex"
        )
        return None

    min_params = _MIN_PARAMS.get(task, 1.5)

    # Filter by minimum parameter size for the task
    suitable = [m for m in models if m["param_gb"] >= min_params]

    if not suitable:
        # No model meets the minimum — use the largest available
        log.warning(
            "No model ≥%.1fB params for task '%s' — using largest available (%.1fB)",
            min_params,
            task,
            models[-1]["param_gb"],
        )
        suitable = [models[-1]]

    # Quality policy determines model size preference.
    # _detect_models() sorts smallest-first; min_params filter ensures the
    # model is suitable for the task tier. The policy only decides whether
    # we pick the smallest (fast/balanced) or largest (quality) suitable.
    if _quality_policy == "quality":
        suitable = sorted(suitable, key=lambda m: -m["param_gb"])  # largest first
    # fast + balanced: smallest first (default from _detect_models sort)
    chosen = suitable[0]
    log.info(
        "Auto-selected model for %s: %s (%.1fB params, %s, %.1fGB) — "
        "policy=%s, %s of %d suitable models",
        task,
        chosen["name"],
        chosen["param_gb"],
        chosen["quantization"],
        chosen["size_bytes"] / 1e9,
        _quality_policy,
        "largest" if _quality_policy == "quality" else "smallest",
        len(suitable),
    )
    return chosen["name"]


# Per-task model cache (picked once per task, reused)
_model_cache: dict[str, str | None] = {}
_manual_model: str | None = None  # set via set_model_override()
_quality_policy: str = "balanced"  # set via set_quality_policy()


def set_model_override(model: str | None) -> None:
    """Override the LLM model used for all tasks.

    Pass None to clear override and revert to auto-detection.
    """
    global _manual_model
    _manual_model = model
    _model_cache.clear()  # force re-pick on next get_model() call
    if model:
        log.info("LLM model override: %s", model)
    else:
        log.info("LLM model override cleared — using auto-detect")


def set_quality_policy(policy: str) -> None:
    """Set the model-selection quality/speed tradeoff policy.

    - 'fast': smallest suitable model for ALL tasks (speed priority)
    - 'balanced': smallest suitable per task tier (speed/quality balance, default)
    - 'quality': largest suitable for ALL tasks (quality priority, slowest)

    Invalid values are logged + ignored (keeps the current policy).
    Clearing the model cache forces re-pick on the next get_model() call.
    """
    global _quality_policy
    valid = ("fast", "balanced", "quality")
    if policy not in valid:
        log.warning("Unknown quality policy '%s' (valid: %s) — ignoring", policy, valid)
        return
    if policy == _quality_policy:
        return
    _quality_policy = policy
    _model_cache.clear()
    log.info("LLM quality policy: %s", policy)


_ctx_override: int | None = None
_predict_override: int | None = None


def set_llm_params(num_ctx: int | None = None, num_predict: int | None = None) -> None:
    """Override LLM context window and max prediction tokens."""
    global _ctx_override, _predict_override
    _ctx_override = num_ctx
    _predict_override = num_predict


def get_available_models() -> list[dict]:
    """Get list of available Ollama text-generation models for UI selection.

    Returns list of dicts: [{"name": str, "size_gb": float}, ...]
    """
    models = _detect_models()
    return [{"name": m["name"], "size_gb": m["size_bytes"] / 1e9} for m in models]


def get_model(task: str = "moderate") -> str | None:
    """Get the selected model name for a given task (cached per task)."""
    if task not in _model_cache:
        _model_cache[task] = _pick_model(task)
    return _model_cache[task]


def _call_ollama(
    model: str,
    prompt: str,
    task: str = "moderate",
) -> str:
    """Call Ollama /api/chat with proper config.

    Uses chat endpoint for Qwen3.5 compatibility.
    Pattern: think=False + /no_think (fast, no reasoning overhead) +
    NO format:json — Qwen3.5 returns EMPTY responses with format:json on long prompts. JSON parsed by _extract_json_from_text() from the response text.
    JSON is parsed by _extract_json_from_text() from the response text.

    Config:
    - think: False + /no_think prefix — fast mode (no reasoning overhead)
    - num_ctx: task-dependent
    - keep_alive: 30m — model stays in GPU memory between calls

    Agent-side LLM hook (separation of concerns per AGENTS.md §4.23):
    Set SCIENTIFIC_RESEARCH_LLM_CALLBACK=/path/to/executable to redirect ALL
    LLM calls to an external agent-owned process. The executable receives
    JSON on stdin: {"model": str, "prompt": str, "task": str} and must
    return the model's response text on stdout. Skill stays infrastructure;
    agent owns cognition.
    """
    global _ollama_consecutive_failures, _ollama_circuit_open

    # Agent-side LLM hook — bypass Ollama entirely when callback registered
    callback = os.environ.get("SCIENTIFIC_RESEARCH_LLM_CALLBACK")
    if callback:
        import subprocess as _sp

        proc = _sp.run(
            [callback],
            input=json.dumps({"model": model, "prompt": prompt, "task": task}),
            capture_output=True,
            text=True,
            timeout=180,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LLM callback {callback} failed exit={proc.returncode}: "
                f"{proc.stderr[:200]}"
            )
        return proc.stdout

    # Circuit breaker — skip HTTP entirely when Ollama is down
    if _ollama_circuit_open:
        raise ConnectionError("Ollama circuit breaker open — server unavailable")

    num_ctx = _ctx_override or _NUM_CTX.get(task, 4096)
    # Auto-expand context for long prompts (abstracts up to 6000 chars)
    estimated_tokens = len(prompt) // 4 + 500  # rough char→token estimate
    if estimated_tokens > num_ctx:
        num_ctx = min(estimated_tokens + 1024, 32768)  # cap at 32K for safety
    num_predict = _predict_override or _NUM_PREDICT.get(task, 2048)

    # Use format:json for SHORT prompts only.
    # Qwen3.5 returns EMPTY responses with format:json on long prompts (>2K chars).
    # For long prompts, JSON is parsed by _extract_json_from_text() from free-form text.
    use_format_json = len(prompt) < 2000
    payload_dict = {
        "model": model,
        "messages": [
            {"role": "user", "content": "/no_think\n" + prompt},
        ],
        "stream": False,
        "think": False,
        "keep_alive": "30m",
        "options": {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": 1,
            "num_ctx": num_ctx,
            "num_predict": num_predict,
            "repeat_penalty": 1.1,
        },
    }
    if use_format_json:
        payload_dict["format"] = "json"
    payload = json.dumps(payload_dict).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
    except (urllib.error.URLError, ConnectionError, OSError) as e:
        _ollama_consecutive_failures += 1
        if (
            _ollama_consecutive_failures >= _OLLAMA_CIRCUIT_THRESHOLD
            and not _ollama_circuit_open
        ):
            _ollama_circuit_open = True
            log.warning(
                "Ollama circuit breaker OPENED after %d consecutive failures "
                "— LLM calls disabled for remainder of session",
                _ollama_consecutive_failures,
            )
        raise

    # Success — reset circuit breaker
    _ollama_consecutive_failures = 0
    _ollama_circuit_open = False

    raw = data.get("message", {}).get("content", "")

    # Fallback: if format:json produced empty response, retry without it
    if not raw.strip() and use_format_json:
        log.debug("format:json returned empty — retrying without format constraint")
        payload_dict.pop("format", None)
        payload = json.dumps(payload_dict).encode()
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode())
        raw = data.get("message", {}).get("content", "")

    # Strip leaked thinking tokens (<think>...</think>) that consume output budget
    raw = re.sub(r"<think>.*?</think>\s*", "", raw, flags=re.DOTALL)
    raw = re.sub(r"<think>.*$", "", raw, flags=re.DOTALL)
    raw = re.sub(r"^.*</think>\s*", "", raw, flags=re.DOTALL)
    return raw.strip()


def _cache_key(abstract: str, model: str, task: str) -> Path:
    """Generate cache file path for a given abstract + model + task."""
    h = hashlib.sha256(
        f"{task}|{_PROMPT_VERSION}|{model}|{abstract}".encode()
    ).hexdigest()
    return _CACHE_DIR / f"{task}_{h}.json"


import time as _time_module

_LLM_CACHE_TTL = 90 * 86400  # 90 days


def _cached_get(key: Path) -> dict | None:
    """Get cached result if exists and fresh (<90 days)."""
    if key.exists():
        try:
            data = json.loads(key.read_text())
            # Check TTL (prompt_version already in cache key, so TTL is secondary)
            cached_at = data.pop("_cached_at", 0)
            if _time_module.time() - cached_at < _LLM_CACHE_TTL:
                return data
            log.debug("LLM cache expired (>%d days)", _LLM_CACHE_TTL // 86400)
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _cached_put(key: Path, data: dict) -> None:
    """Store result in cache with timestamp."""
    try:
        data["_cached_at"] = _time_module.time()
        key.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    except OSError as e:
        log.warning("Cache write failed: %s", e)


# =============================================================================
# Extraction prompt
# =============================================================================
_EXTRACTION_PROMPT = """Extract research data from this scientific paper. Return ONLY a JSON object.

Title: {title}
Topic: {topic}
Abstract: {abstract}

Return this JSON format (all fields required, use "" if not applicable):
{{
  "discipline": "geochemistry, geology, petrology, geophysics, economic_geology, structural_geology, volcanology, etc.",
  "key_finding": "the MOST SPECIFIC result: name the method, the minerals/elements, the quantitative values. NOT a generic summary — state the geological fact with precision. Example: 'Garnet-biotite Fe-Mg exchange thermometry yields 550-620°C in Barrovian metapelites, consistent with garnet zone metamorphic grade.' NOT: 'The study investigates temperature conditions.'",
  "study_type": "experimental, field_study, review, theoretical, computational, or method_development",
  "novelty": "milestone, methodological, incremental, confirmation, or review",
  "interpretation": "the authors' geological interpretation: what process, condition, or history do the data imply? Name the geological concept (e.g. 'partial melting under vapor-absent conditions', 'fluid-assisted metamorphism at amphibolite facies', 'crustal contamination during arc magma ascent').",
  "geological_concepts": "key minerals, rock types, or methods mentioned (e.g. 'garnet, biotite, metapelite, Fe-Mg exchange, KD, geothermometry')",
  "quantitative_data": "P-T ranges, ages, compositions, partition coefficients WITH units (e.g. 'T=550-620°C, P=4-6 kbar, KD=0.2, SiO2=52wt%') or empty string"
}}

Return ONLY the JSON object. No other text."""


def extract_paper(
    abstract: str,
    title: str = "",
    topic: str = "",
    fallback: dict | None = None,
) -> dict:
    """Extract structured data from a paper abstract using LLM.

    Falls back to the provided `fallback` dict (from regex/lexicon) if LLM
    is unavailable or fails.

    Returns dict with keys: pico, study_design, effect_sizes,
    stance_toward_topic, key_finding, abbreviations.
    """
    if not abstract or not abstract.strip():
        return fallback or _empty_result()

    model = get_model("moderate")
    if not model:
        log.debug("No LLM model — using fallback")
        return fallback or _empty_result()

    # Check cache
    cache_key = _cache_key(abstract, model, "extract")
    cached = _cached_get(cache_key)
    if cached is not None:
        log.debug("Cache hit for extraction")
        return cached

    # Call LLM
    prompt = _EXTRACTION_PROMPT.format(
        topic=topic or "(not specified)",
        title=title or "(not provided)",
        abstract=abstract[:6000],  # cap at 6000 chars (~1500 tokens, within 4096 ctx)
    )

    # Retry LLM call up to 2 times on failure (rate limit, timeout, bad JSON)
    for llm_attempt in range(3):
        # Skip retries entirely when circuit breaker is open — no point
        # wasting 6s per paper on backoff for a server we know is down
        if _ollama_circuit_open and llm_attempt > 0:
            return fallback or _empty_result()
        try:
            response_text = _call_ollama(model, prompt, task="moderate")
            # Empty response — don't retry, fall back immediately (saves 15s)
            if not response_text or not response_text.strip():
                log.debug(
                    "Ollama returned empty response for '%s' — using regex fallback",
                    title[:40],
                )
                return fallback or _empty_result()
            # Try strict JSON parse first
            try:
                result = json.loads(response_text)
            except json.JSONDecodeError:
                # Fallback: extract JSON object from response
                result = _extract_json_from_text(response_text)
                if result is None:
                    # Don't retry — same model + same prompt = same non-JSON output.
                    # Fall back to regex immediately (saves 15s, eliminates warning spam).
                    log.debug(
                        "LLM returned non-JSON for '%s' — using regex fallback",
                        title[:40],
                    )
                    return fallback or _empty_result()
            # Validate structure
            result = _validate_extraction(result)
            # Data quality check: ensure at least discipline was extracted
            if not result.get("discipline"):
                log.debug(
                    "LLM returned no discipline for '%s' — accepting partial",
                    title[:40],
                )
            _cached_put(cache_key, result)
            log.debug("LLM extraction successful for: %s", title[:50])
            return result
        except (urllib.error.URLError, OSError) as e:
            if llm_attempt < 2:
                wait = 2.0 * (llm_attempt + 1)
                log.warning(
                    "LLM network error (attempt %d/3): %s — retry in %.1fs",
                    llm_attempt + 1,
                    e,
                    wait,
                )
                import time as _t

                _t.sleep(wait)
                continue
            log.warning(
                "LLM extraction failed after 3 attempts: %s — using fallback", e
            )
            return fallback or _empty_result()
        except (json.JSONDecodeError, KeyError) as e:
            log.warning("LLM extraction failed: %s — using fallback", e)
            return fallback or _empty_result()

    return fallback or _empty_result()


# =============================================================================
# Batch extraction — multiple papers per LLM call (3-5× fewer round-trips)
# =============================================================================
_BATCH_EXTRACTION_PROMPT = """Extract research data from these {n} scientific papers. Return ONLY a JSON array.

Topic: {topic}

{papers_block}

Return a JSON array with EXACTLY {n} objects (one per paper, SAME ORDER). Each object MUST have ALL these fields:
[{{"discipline":"...","key_finding":"...","study_type":"...","novelty":"...","interpretation":"...","geological_concepts":"...","quantitative_data":"..."}}]

Fields per object:
- discipline: geochemistry, geology, petrology, geophysics, economic_geology, etc.
- key_finding: MOST SPECIFIC geological result — name the method, minerals, and quantitative values. NOT generic summary.
- study_type: experimental, field_study, review, theoretical, computational, or method_development
- novelty: milestone, methodological, incremental, confirmation, or review
- interpretation: geological interpretation — what process or history do the data imply?
- geological_concepts: key minerals, rock types, methods mentioned (comma-separated)
- quantitative_data: P-T ranges, ages, compositions WITH units, or empty string

Return ONLY the JSON array."""


def extract_papers_batch(
    papers: list[dict],
    topic: str = "",
    batch_size: int = 3,
) -> list[dict]:
    """Extract structured data from multiple paper abstracts in batches.

    Groups papers into batches of ``batch_size`` (default 3) and sends one
    LLM call per batch — 3-5× fewer round-trips than per-paper extraction.

    Each paper dict must have keys: ``abstract`` (str), ``title`` (str).
    Optional: ``fallback`` (dict from regex/lexicon pre-extraction).

    Falls back to individual ``extract_paper`` calls when:
    - LLM is unavailable
    - A batch response fails to parse as a JSON array
    - The array length does not match the batch size (count mismatch)

    Uses the same SHA256 per-paper cache as ``extract_paper``, so cached
    papers are served from disk without hitting the LLM. Only uncached
    papers are batched.

    Returns a list of extraction dicts in the SAME ORDER as the input.
    """
    if not papers:
        return []

    model = get_model("moderate")
    if not model or _ollama_circuit_open:
        # No LLM — fall back to per-paper regex extraction
        return [
            extract_paper(
                p.get("abstract", ""),
                title=p.get("title", ""),
                topic=topic,
                fallback=p.get("fallback"),
            )
            for p in papers
        ]

    # Phase 1: serve cached papers + collect uncached for batching
    results: list[dict | None] = [None] * len(papers)
    uncached_indices: list[int] = []
    for i, p in enumerate(papers):
        abstract = p.get("abstract", "")
        if not abstract or not abstract.strip():
            results[i] = p.get("fallback") or _empty_result()
            continue
        cache_key = _cache_key(abstract, model, "extract")
        cached = _cached_get(cache_key)
        if cached is not None:
            results[i] = cached
        else:
            uncached_indices.append(i)

    if not uncached_indices:
        log.debug("Batch extraction: all %d papers served from cache", len(papers))
        return results  # type: ignore[return-value]

    # Phase 2: batch the uncached papers
    for batch_start in range(0, len(uncached_indices), batch_size):
        batch_idx_slice = uncached_indices[batch_start : batch_start + batch_size]
        batch_papers = [papers[i] for i in batch_idx_slice]

        # Build the papers block for the prompt
        paper_lines = []
        for j, p in enumerate(batch_papers, 1):
            title = p.get("title", "(not provided)")
            abstract = (p.get("abstract") or "")[:3000]
            paper_lines.append(f"Paper {j}:\nTitle: {title}\nAbstract: {abstract}")
        papers_block = "\n\n".join(paper_lines)

        prompt = _BATCH_EXTRACTION_PROMPT.format(
            n=len(batch_papers),
            topic=topic or "(not specified)",
            papers_block=papers_block,
        )

        batch_success = False
        try:
            response_text = _call_ollama(model, prompt, task="moderate")
            if response_text and response_text.strip():
                # Try strict JSON array parse first
                try:
                    batch_result = json.loads(response_text)
                except json.JSONDecodeError:
                    batch_result = _extract_json_array_from_text(response_text)

                if isinstance(batch_result, list) and len(batch_result) == len(
                    batch_papers
                ):
                    # Validate + cache each result
                    for j, raw in enumerate(batch_result):
                        if not isinstance(raw, dict):
                            batch_success = False
                            break
                        validated = _validate_extraction(raw)
                        paper_idx = batch_idx_slice[j]
                        abstract = batch_papers[j].get("abstract", "")
                        cache_key = _cache_key(abstract, model, "extract")
                        _cached_put(cache_key, validated)
                        results[paper_idx] = validated
                    else:
                        batch_success = True
                        log.debug(
                            "Batch extraction: %d/%d papers extracted in one call",
                            len(batch_papers),
                            len(uncached_indices),
                        )
        except (urllib.error.URLError, OSError) as e:
            log.warning("Batch LLM call failed: %s — falling back to individual", e)

        # Phase 3: fallback to individual extraction for failed batches
        if not batch_success:
            log.debug(
                "Batch failed for %d papers — individual extraction", len(batch_papers)
            )
            for j, p in enumerate(batch_papers):
                paper_idx = batch_idx_slice[j]
                if results[paper_idx] is None:
                    results[paper_idx] = extract_paper(
                        p.get("abstract", ""),
                        title=p.get("title", ""),
                        topic=topic,
                        fallback=p.get("fallback"),
                    )

    return [r if r is not None else _empty_result() for r in results]


def _extract_json_array_from_text(text: str) -> list | None:
    """Extract the first valid JSON array from text with non-JSON wrapping.

    Companion to ``_extract_json_from_text`` for the batch path (which
    expects an array, not a single object).
    """
    clean = text.strip()
    if clean.startswith("```"):
        lines = clean.split("\n")
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines)

    start = clean.find("[")
    end = clean.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = clean[start : end + 1]
    try:
        result = json.loads(json_str)
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            result = json.loads(fixed)
            return result if isinstance(result, list) else None
        except json.JSONDecodeError:
            return None


def _extract_json_from_text(text: str) -> dict | None:
    """Extract the first valid JSON object from text that may contain
    non-JSON prefix/suffix (e.g., LLM thinking tokens, markdown fences)."""
    # Strip markdown code fences
    clean = text.strip()
    if clean.startswith("```"):
        # Remove first line (```json or ```) and last ```
        lines = clean.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        clean = "\n".join(lines)

    # Find first { and last }
    start = clean.find("{")
    end = clean.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    json_str = clean[start : end + 1]
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try fixing common issues: trailing commas, unquoted keys

        # Remove trailing commas before } or ]
        fixed = re.sub(r",\s*([}\]])", r"\1", json_str)
        try:
            return json.loads(fixed)
        except json.JSONDecodeError:
            return None


def _empty_result() -> dict:
    """Return empty extraction result (field-agnostic format).

    All string fields default to '' (never None) to prevent downstream
    NoneType errors. quality_score stays None (numeric: not-yet-scored).
    """
    return {
        "discipline": "",
        "study_type": "",
        "subject": {"system_studied": "", "samples": "", "context": ""},
        "method": {"technique": "", "instrument": "", "conditions": ""},
        "key_results": [],
        "interpretation": "",
        "novelty": "",
        "stance_toward_topic": "",
        "key_finding": "",
        "geological_concepts": "",
        "quantitative_data": "",
        "abbreviations": {},
        "risk_of_bias": "unknown",
        "quality_score": None,
        "pico": {"population": "", "intervention": "", "comparator": "", "outcome": ""},
        "study_design": "",
        "effect_sizes": [],
    }


def _validate_extraction(data: dict) -> dict:
    """Validate and normalize LLM extraction output (field-agnostic format)."""
    result = _empty_result()
    result.update({k: v for k, v in data.items() if k in result})

    # Coerce None → "" for string fields (LLM may return null)
    for _k in (
        "discipline",
        "study_type",
        "interpretation",
        "novelty",
        "stance_toward_topic",
        "key_finding",
        "study_design",
    ):
        if result.get(_k) is None:
            result[_k] = ""

    # Normalize discipline
    disc = result.get("discipline")
    if disc and isinstance(disc, str):
        result["discipline"] = disc.lower().strip()

    # Normalize study_type → also set legacy study_design
    st = result.get("study_type")
    if st and isinstance(st, str):
        result["study_type"] = st.lower().strip()
        result["study_design"] = result["study_type"]

    # Normalize subject
    subj = result.get("subject", {})
    if not isinstance(subj, dict):
        subj = {}
    for key in ("system_studied", "samples", "context"):
        if key not in subj or subj[key] is None:
            subj[key] = ""
    result["subject"] = subj
    # Legacy: map system_studied → pico.population
    if subj.get("system_studied"):
        result["pico"]["population"] = subj["system_studied"]

    # Normalize method
    meth = result.get("method", {})
    if not isinstance(meth, dict):
        meth = {}
    for key in ("technique", "instrument", "conditions"):
        if key not in meth or meth[key] is None:
            meth[key] = ""
    result["method"] = meth

    # Normalize key_results
    kr = result.get("key_results", [])
    if not isinstance(kr, list):
        kr = []
    result["key_results"] = kr

    # Legacy: convert key_results to effect_sizes format for meta_analyze.py
    # Each key_result with a comparison → a paired effect size
    legacy_effects = []
    for r in kr:
        if not isinstance(r, dict):
            continue
        val = r.get("value")
        if val is None:
            continue
        try:
            val = float(val)
        except (ValueError, TypeError):
            continue
        unc = r.get("uncertainty")
        try:
            unc = float(unc) if unc is not None else None
        except (ValueError, TypeError):
            unc = None
        n = r.get("n")
        try:
            n = int(n) if n is not None else None
        except (ValueError, TypeError):
            n = None
        legacy_effects.append(
            {
                "m1": val,
                "sd1": unc,
                "n": n,
                "m2": None,
                "sd2": None,
                "n2": None,
                "ci_lower": None,
                "ci_upper": None,
                "outcome": r.get("measurement", ""),
                "unit": r.get("unit", ""),
                "comparison": r.get("comparison"),
            }
        )
    result["effect_sizes"] = legacy_effects

    # Normalize abbreviations
    abbrs = result.get("abbreviations", {})
    if not isinstance(abbrs, dict):
        abbrs = {}
    result["abbreviations"] = abbrs

    # Normalize stance
    stance = result.get("stance_toward_topic")
    if stance and isinstance(stance, str):
        stance = stance.lower().strip()
        valid = ("supporting", "contrasting", "extending", "methodology", "mentioning")
        if stance not in valid:
            if "support" in stance or "confirm" in stance:
                stance = "supporting"
            elif "contrast" in stance or "differ" in stance or "conflict" in stance:
                stance = "contrasting"
            elif "extend" in stance or "build" in stance:
                stance = "extending"
            elif "method" in stance:
                stance = "methodology"
            else:
                stance = "mentioning"
    result["stance_toward_topic"] = stance

    return result


# =============================================================================
# Stance classification (standalone, for correlate.py)
# =============================================================================
_STANCE_PROMPT = """Classify the relationship of this sentence to the research topic.

Research topic: {claim}
Sentence from paper: {sentence}

Return JSON:
{{
  "stance": "one of: supporting, contrasting, extending, methodology, mentioning",
  "confidence": 0.0-1.0,
  "reason": "one phrase explaining why"
}}

Definitions (field-agnostic):
- supporting: the sentence confirms or agrees with the topic/hypothesis
- contrasting: the sentence contradicts, disagrees, or presents limitations
- extending: the sentence builds on or extends the topic with new data/interpretation
- methodology: the sentence describes methods, techniques, or data used
- mentioning: the sentence neutrally references the topic without taking a stance
"""


def classify_stance(
    sentence: str,
    claim: str = "",
    fallback: tuple[str, float] | None = None,
) -> tuple[str, float, str]:
    """Classify citation stance using LLM.

    Returns (stance, confidence, reason).
    Falls back to provided (stance, confidence) if LLM unavailable.
    """
    if not sentence or not sentence.strip():
        return ("mentioning", 0.0, "empty sentence")

    model = get_model("simple")
    if not model or _ollama_circuit_open:
        if fallback:
            return (fallback[0], fallback[1], "regex fallback (no LLM)")
        return ("mentioning", 0.0, "no LLM available")

    # Check cache
    cache_key = _cache_key(sentence + claim, model, "stance")
    cached = _cached_get(cache_key)
    if cached is not None:
        return (
            cached.get("stance", "mentioning"),
            cached.get("confidence", 0.5),
            cached.get("reason", "cached"),
        )

    prompt = _STANCE_PROMPT.format(
        claim=claim or "(general topic)",
        sentence=sentence[:1000],
    )

    try:
        response_text = _call_ollama(model, prompt, task="simple")
        try:
            result = json.loads(response_text)
        except json.JSONDecodeError:
            result = _extract_json_from_text(response_text)
            if result is None:
                raise json.JSONDecodeError(
                    "No JSON in stance response", response_text, 0
                )
        stance = result.get("stance", "mentioning").lower().strip()
        valid_stances = (
            "supporting",
            "contrasting",
            "extending",
            "methodology",
            "mentioning",
        )
        if stance not in valid_stances:
            stance = "mentioning"
        confidence = float(result.get("confidence", 0.5))
        reason = result.get("reason", "")
        cached = {"stance": stance, "confidence": confidence, "reason": reason}
        _cached_put(cache_key, cached)
        return (stance, confidence, reason)
    except (json.JSONDecodeError, urllib.error.URLError, OSError, ValueError) as e:
        log.warning("LLM stance failed: %s — using fallback", e)
        if fallback:
            return (fallback[0], fallback[1], f"regex fallback ({e})")
        return ("mentioning", 0.0, f"LLM failed: {e}")


# =============================================================================
# Health check
# =============================================================================
def is_available() -> bool:
    """Check if LLM extraction is available (Ollama running + text models installed)."""
    return get_model("moderate") is not None


def health_check() -> dict:
    """Return health status for diagnostics."""
    models = _detect_models()
    return {
        "ollama_url": OLLAMA_URL,
        "available": len(models) > 0,
        "text_generation_models": [
            {
                "name": m["name"],
                "params": f"{m['param_gb']:.1f}B",
                "size": f"{m['size_bytes'] / 1e9:.1f}GB",
                "family": m.get("family", "?"),
                "capabilities": m.get("capabilities", []),
            }
            for m in models
        ],
        "selected_for_simple": get_model("simple"),
        "selected_for_moderate": get_model("moderate"),
        "cache_dir": str(_CACHE_DIR),
        "prompt_version": _PROMPT_VERSION,
        "config": {
            "think": False,
            "format": "none (free-form, parsed by _extract_json_from_text)",
            "temperature": 0.0,
            "top_k": 1,
            "num_ctx_simple": _NUM_CTX["simple"],
            "num_ctx_moderate": _NUM_CTX["moderate"],
            "num_predict_simple": _NUM_PREDICT["simple"],
            "num_predict_moderate": _NUM_PREDICT["moderate"],
        },
    }


# =============================================================================
# LLM screening judge (single + ensemble) — used by screen_llm.py
# =============================================================================
_SCREEN_PROMPT = """You are a systematic-review screening assistant.
Decide whether this paper is TOPICALLY RELEVANT to the research question.

Research question: {query}
{type_line}

Title: {title}
Abstract: {abstract}

Answer with ONLY a JSON object: {{"relevant": true/false, "reason": "<=15 words"}}"""


def llm_screen_paper(
    paper: Any,
    query: str,
    model: str | None = None,
    research_type: str | None = None,
) -> tuple[bool, str] | None:
    """Single-LLM relevance judge for PRISMA title/abstract screening.

    Returns (relevant, reason) or None when no LLM is available (caller
    falls back to permissive accept).
    """
    title = (getattr(paper, "title", "") or "")[:300]
    abstract = (getattr(paper, "abstract", "") or "")[:4000]
    if not title and not abstract:
        return None
    chosen = model or _pick_model("simple")
    if not chosen:
        return None
    type_line = f"\nReview type: {research_type}" if research_type else ""
    prompt = _SCREEN_PROMPT.format(
        query=query[:500], type_line=type_line, title=title, abstract=abstract
    )
    try:
        response = _call_ollama(chosen, prompt, task="simple")
    except Exception as e:  # noqa: BLE001 — caller handles permissive fallback
        log.debug("screen judge call failed: %s", e)
        return None
    try:
        parsed = json.loads(response)
    except json.JSONDecodeError:
        parsed = _extract_json_from_text(response)
    if not isinstance(parsed, dict) or "relevant" not in parsed:
        return None
    relevant = bool(parsed["relevant"])
    reason = str(parsed.get("reason", ""))[:120]
    return relevant, reason


def llm_screen_paper_ensemble(
    paper: Any,
    query: str,
    n_judges: int = 3,
    research_type: str | None = None,
) -> tuple[bool, str, list[dict]] | None:
    """Majority-vote ensemble screening (Sanghera 2025, JAMIA 32(5):893-904).

    Votes are cast by DISTINCT available Ollama models (diversity is what
    makes an ensemble; one temperature-0 model polled 3× gives identical
    votes). With fewer than 2 models available, degrades to single judge
    and says so in the reason — no fake quorum.

    Returns (decision, reason, votes) or None when no LLM available.
    votes: [{"model": str, "relevant": bool, "reason": str}, ...]
    """
    chosen_model = _pick_model("simple")
    if not chosen_model:
        return None
    # Build a diverse judge panel: distinct model names, prefer size spread
    models = _detect_models()
    panel = [m["name"] for m in models if m["name"] != chosen_model][: n_judges - 1]
    panel = [chosen_model] + panel
    votes: list[dict] = []
    for m in panel:
        v = llm_screen_paper(paper, query, model=m, research_type=research_type)
        if v is None:
            continue
        votes.append({"model": m, "relevant": v[0], "reason": v[1]})
    if not votes:
        return None
    n_yes = sum(1 for v in votes if v["relevant"])
    decision = n_yes * 2 > len(votes)
    if len(votes) < 2:
        note = f"single judge ({votes[0]['model']}) — ensemble unavailable"
    else:
        note = f"{n_yes}/{len(votes)} judges voted include"
    reason = f"{note}; {votes[0].get('reason', '')}"
    return decision, reason, votes


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="LLM extraction health check")
    p.add_argument("--test", action="store_true", help="run a test extraction")
    args = p.parse_args()

    print(json.dumps(health_check(), indent=2))

    if args.test:
        print("\n=== Test extraction ===")
        test_abstract = (
            "We conducted a randomized controlled trial of 200 adults with obesity. "
            "The intervention group (n=100) followed intermittent fasting (IF) 16:8 "
            "for 12 weeks. The control group (n=100) received continuous calorie "
            "restriction (CR). Primary outcome was weight loss. The IF group lost "
            "5.2±1.1 kg vs 4.5±1.0 kg in CR (p=0.001, 95% CI: 0.3-1.1). "
            "IF improved insulin sensitivity (HOMA-IR: -2.1 vs -1.3, p=0.01)."
        )
        result = extract_paper(
            test_abstract,
            title="IF vs CR for Weight Loss",
            topic="intermittent fasting weight loss",
        )
        print(json.dumps(result, indent=2))
