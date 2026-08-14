#!/usr/bin/env python3
"""Mermaid renderers for the scientific-research skill.

Renders citation graphs, PRISMA flow diagrams, forest plots, temporal timelines,
and concept clusters as Mermaid markdown (renderable by GitHub/Obsidian/etc.).
Plus matplotlib publication-grade forest + funnel plots (PNG/SVG).
stdlib + numpy + matplotlib — no heavy deps.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np


# =============================================================================
# Citation graph (Mermaid graph TD)
# =============================================================================
def render_citation_graph(papers: list[dict], max_nodes: int = 50, max_edges: int = 200) -> str:
    """Render citation graph as Mermaid graph TD.
    Each paper = node (DOI label). Edge A -> B means 'A cites B'.
    Papers must be dicts with 'doi' (or 'openalex_id'), 'title', and
    optionally 'referenced_works' (list of OpenAlex IDs).
    """
    by_id: dict[str, dict] = {}
    for p in papers:
        pid = p.get("doi") or p.get("openalex_id") or p.get("arxiv_id")
        if pid:
            by_id[pid] = p
    lines = ["graph TD"]
    seen_nodes: set[str] = set()
    seen_edges: set[tuple[str, str]] = set()
    edge_count = 0
    for p in list(papers)[:max_nodes]:
        pid = p.get("doi") or p.get("openalex_id") or p.get("arxiv_id")
        if not pid:
            continue
        if pid not in seen_nodes:
            label = (p.get("title") or "?")[:60].replace('"', "'")
            lines.append(f'  {node_id(pid)}["{label}"]')
            seen_nodes.add(pid)
        for ref in p.get("referenced_works") or []:
            target = by_id.get(ref) or by_id.get(ref.split("/")[-1])
            if target:
                tid = target.get("doi") or target.get("openalex_id") or target.get("arxiv_id")
                if tid and tid not in seen_nodes:
                    tlabel = (target.get("title") or "?")[:60].replace('"', "'")
                    lines.append(f'  {node_id(tid)}["{tlabel}"]')
                    seen_nodes.add(tid)
                if tid and (pid, tid) not in seen_edges:
                    lines.append(f"  {node_id(pid)} --> {node_id(tid)}")
                    seen_edges.add((pid, tid))
                    edge_count += 1
                    if edge_count >= max_edges:
                        lines.append("  %% max_edges reached")
                        return "\n".join(lines)
                    if len(seen_nodes) >= max_nodes:
                        break
    return "\n".join(lines)


def node_id(s: str) -> str:
    """Sanitize identifier for Mermaid node ID."""
    return "N" + str(abs(hash(s)) % (10**12))


# =============================================================================
# PRISMA 2020 flow diagram (Mermaid flowchart TD)
# =============================================================================
def render_prisma_flow(
    identification: int,
    duplicates_removed: int,
    screened: int,
    excluded_at_screening: int,
    sought_for_eligibility: int,
    not_retrieved: int,
    assessed_for_eligibility: int,
    excluded_at_eligibility: int,
    included: int,
    other_sources: int = 0,
    other_sources_included: int = 0,
) -> str:
    """Render PRISMA 2020 flow diagram (4 phases).
    Reference: http://prisma-statement.org/PRISMAStatement/FlowDiagram.aspx
    """
    return f"""flowchart TD
    %% PRISMA 2020 Flow Diagram (databases + registers + other sources)
    ID["**Identification**
    Records identified from databases/registers (n={identification})
    Records identified from other sources (n={other_sources})"]
    DEDUP["Records removed before screening:
    duplicate records removed (n={duplicates_removed})"]
    SCREEN["**Screening**
    Records screened (n={screened})"]
    EX_SCREEN["Records excluded (n={excluded_at_screening})"]
    ELIG["**Eligibility**
    Records sought for retrieval (n={sought_for_eligibility})
    Records assessed for eligibility (n={assessed_for_eligibility})"]
    NOT_RET["Records not retrieved (n={not_retrieved})"]
    EX_ELIG["Records excluded (n={excluded_at_eligibility})"]
    INC["**Included**
    Studies included in review (n={included + other_sources_included})"]

    ID --> DEDUP
    DEDUP --> SCREEN
    SCREEN --> EX_SCREEN
    SCREEN --> ELIG
    ELIG --> NOT_RET
    ELIG --> EX_ELIG
    ELIG --> INC
"""


# =============================================================================
# Forest plot (Mermaid ASCII-style)
# =============================================================================
def render_forest_plot(
    studies: list[dict],
    pooled: dict | None = None,
    effect_label: str = "Effect size",
    ci_label: str = "95% CI",
) -> str:
    """Render ASCII-style forest plot.
    Each study: {name, effect, ci_lower, ci_upper, weight}.
    Optional pooled: {effect, ci_lower, ci_upper} as last row (diamond).
    Returns a markdown code block (monospace ASCII forest plot)."""
    all_rows = studies + ([pooled] if pooled else [])
    if not all_rows:
        return "```text\n(no studies)\n```"
    all_vals = (
        [r["effect"] for r in all_rows]
        + [r["ci_lower"] for r in all_rows]
        + [r["ci_upper"] for r in all_rows]
    )
    x_min = min(all_vals)
    x_max = max(all_vals)
    if x_max == x_min:
        x_max = x_min + 1
    width = 50
    out = ["```text"]
    name_w = max(len(s["name"]) for s in studies) + 2 if studies else 10
    out.append(f"{'Study':<{name_w}} {'Effect':>8} {'95% CI':>18} {'Weight':>7}  Forest plot")
    out.append("-" * (name_w + 8 + 18 + 7 + 2 + width + 5))

    def scale(v: float) -> int:
        return int((v - x_min) / (x_max - x_min) * width)

    for s in studies:
        e = s["effect"]
        lo, hi = s["ci_lower"], s["ci_upper"]
        w = s.get("weight", 0.0)
        pct = f"{w * 100:5.1f}%" if w <= 1.0 else f"{w:5.1f}%"
        line = list(" " * width)
        a, b = max(scale(lo), 0), min(scale(hi), width - 1)
        for i in range(a, b + 1):
            line[i] = "-"
        mid = max(0, min(scale(e), width - 1))
        line[mid] = "|"
        bar = "".join(line)
        out.append(f"{s['name']:<{name_w}} {e:>8.3f} [{lo:>7.3f}, {hi:>7.3f}] {pct:>7}  {bar}")

    if pooled:
        e = pooled["effect"]
        lo, hi = pooled["ci_lower"], pooled["ci_upper"]
        line = list(" " * width)
        a, b = max(scale(lo), 0), min(scale(hi), width - 1)
        for i in range(a, b + 1):
            line[i] = "="
        mid = max(0, min(scale(e), width - 1))
        line[mid] = "*"
        bar = "".join(line)
        out.append("-" * (name_w + 8 + 18 + 7 + 2 + width + 5))
        out.append(f"{'Pooled':<{name_w}} {e:>8.3f} [{lo:>7.3f}, {hi:>7.3f}] {'':>7}  {bar}")
    out.append("")
    out.append(
        f"{'Scale:':<{name_w + 8 + 18 + 7 + 3}}<{x_min:>8.3f}{'':>{width - 16}}{x_max:>8.3f}>"
    )
    out.append("```")
    return "\n".join(out)


# =============================================================================
# Timeline (Mermaid gantt or simple ASCII histogram)
# =============================================================================
def render_temporal_histogram(papers: list[dict], bucket: str = "year") -> str:
    """ASCII histogram of paper publication years (or decades).
    bucket: 'year' | 'decade'."""
    years = [p.get("year") for p in papers if p.get("year")]
    if not years:
        return "```text\n(no publication years)\n```"
    if bucket == "decade":
        keys = [f"{(y // 10) * 10}s" for y in years]
    else:
        keys = [str(y) for y in years]
    counter = Counter(keys)
    if bucket == "decade":
        sorted_keys = sorted(counter.keys(), key=lambda k: int(k[:-1]))
    else:
        sorted_keys = sorted(counter.keys(), key=int)
    max_count = max(counter.values())
    bar_width = 40
    out = ["```text", f"Publication distribution ({len(years)} papers):", ""]
    for k in sorted_keys:
        c = counter[k]
        bar_len = int((c / max_count) * bar_width) if max_count > 0 else 0
        out.append(f"{k:>6} | {'#' * bar_len} {c}")
    out.append("```")
    return "\n".join(out)


# =============================================================================
# Concept clusters (Mermaid mindmap)
# =============================================================================
def render_concept_clusters(papers: list[dict], top_n: int = 15) -> str:
    """Mermaid mindmap of top concepts across corpus."""
    counter: Counter = Counter()
    for p in papers:
        for c in p.get("concepts") or []:
            if c:
                counter[c] += 1
        for f in p.get("fields_of_study") or []:
            if f:
                counter[f] += 1
    top = counter.most_common(top_n)
    if not top:
        return "```text\n(no concepts found)\n```"
    lines = ["mindmap"]
    lines.append("  root((Corpus Concepts))")
    for concept, count in top:
        lines.append(f"    {concept}::{count}")
    return "\n".join(lines)


# =============================================================================
# Author network (Mermaid graph LR)
# =============================================================================
def render_author_network(papers: list[dict], top_authors: int = 20) -> str:
    """Render co-authorship network as Mermaid graph.
    Edge between two authors = they co-authored at least one paper.
    Node size proportional to paper count."""
    paper_count: Counter = Counter()
    edges: Counter = Counter()
    for p in papers:
        authors = [a["name"] for a in (p.get("authors") or []) if a.get("name")]
        for a in authors:
            paper_count[a] += 1
        # All pairs (limit to first 8 authors per paper to avoid K_n blowup)
        for i, a1 in enumerate(authors[:8]):
            for j in range(i + 1, min(len(authors), 8)):
                a2 = authors[j]
                key = tuple(sorted([a1, a2]))
                edges[key] += 1
    top_ppl = [name for name, _ in paper_count.most_common(top_authors)]
    top_set = set(top_ppl)
    lines = ["graph LR"]
    for name in top_ppl:
        n = paper_count[name]
        lines.append(f'  {author_id(name)}("{name}<br/>{n} papers")')
    seen_edges: set = set()
    for (a1, a2), count in edges.most_common():
        if a1 in top_set and a2 in top_set:
            key = tuple(sorted([a1, a2]))
            if key not in seen_edges:
                lines.append(f"  {author_id(a1)} ---|{count}| {author_id(a2)}")
                seen_edges.add(key)
                if len(seen_edges) >= 80:
                    break
    return "\n".join(lines)


def author_id(name: str) -> str:
    return "A" + str(abs(hash(name)) % (10**10))


# =============================================================================
# Funding source bar chart
# =============================================================================
def render_funding_sources(papers: list[dict], top_n: int = 15) -> str:
    """ASCII bar chart of top funders across corpus."""
    counter: Counter = Counter()
    for p in papers:
        for f in p.get("funders") or []:
            name = (f or {}).get("name") if isinstance(f, dict) else str(f)
            if name:
                counter[name] += 1
    if not counter:
        return "```text\n(no funder data)\n```"
    top = counter.most_common(top_n)
    max_c = top[0][1] if top else 1
    bar_width = 40
    out = ["```text", "Top funding sources:", ""]
    for name, count in top:
        bar_len = int((count / max_c) * bar_width)
        out.append(f"{name[:40]:<40} | {'#' * bar_len} {count}")
    out.append("```")
    return "\n".join(out)


# =============================================================================
# Geographic / institutional distribution
# =============================================================================
def render_institutional_distribution(papers: list[dict], top_n: int = 15) -> str:
    """Top institutions across corpus."""
    counter: Counter = Counter()
    for p in papers:
        for a in p.get("authors") or []:
            for inst in a.get("affiliation") or []:
                if inst:
                    counter[inst] += 1
            for country in a.get("countries") or []:
                if country:
                    counter[country] += 1
    if not counter:
        return "```text\n(no institution/country data)\n```"
    top = counter.most_common(top_n)
    max_c = top[0][1] if top else 1
    bar_width = 40
    out = ["```text", "Top institutions/countries:", ""]
    for name, count in top:
        bar_len = int((count / max_c) * bar_width)
        out.append(f"{name[:40]:<40} | {'#' * bar_len} {count}")
    out.append("```")
    return "\n".join(out)


# =============================================================================
# matplotlib forest plot (publication-grade PNG/SVG)
# =============================================================================
def render_forest_plot_png(
    studies: list[dict],
    pooled: dict | None = None,
    effect_label: str = "Effect size (95% CI)",
    title: str = "Forest plot",
    output_path: str | Path = "forest.png",
    figsize: tuple[float, float] = (8, 1.5 + 0.4 * 20),
) -> Path:
    """Render publication-grade forest plot via matplotlib (PNG/SVG).
    studies: list of {name, effect, ci_lower, ci_upper, weight} (weight 0-1).
    pooled: optional {effect, ci_lower, ci_upper} drawn as diamond.
    Returns the output path."""
    import matplotlib

    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from matplotlib import patches

    n = len(studies)
    fig_h = max(2.0, 1.0 + 0.4 * (n + (2 if pooled else 1)))
    fig, ax = plt.subplots(figsize=(8, fig_h))
    # Compute x-axis range
    all_vals = []
    for s in studies:
        all_vals.extend([s["effect"], s["ci_lower"], s["ci_upper"]])
    if pooled:
        all_vals.extend([pooled["effect"], pooled["ci_lower"], pooled["ci_upper"]])
    if not all_vals:
        return Path(output_path)
    x_min, x_max = min(all_vals), max(all_vals)
    pad = (x_max - x_min) * 0.1 if x_max > x_min else 1
    x_min -= pad
    x_max += pad
    # Zero / null-effect line
    ax.axvline(x=0, color="#666666", linestyle="--", linewidth=0.8, zorder=1)
    # Plot per-study
    y_positions = list(range(n, 0, -1))  # top-to-bottom
    for i, s in enumerate(studies):
        y = y_positions[i]
        eff = s["effect"]
        lo = s["ci_lower"]
        hi = s["ci_upper"]
        w = s.get("weight", 1.0)
        # CI line
        ax.plot([lo, hi], [y, y], color="#1f77b4", linewidth=1.5, zorder=2)
        # Squares scaled by weight (4-12 range)
        size = 40 + 200 * min(max(w, 0.05), 1.0)
        ax.scatter(
            [eff],
            [y],
            s=size,
            color="#1f77b4",
            marker="s",
            zorder=3,
            edgecolors="white",
            linewidth=0.5,
        )
    # Pooled diamond
    if pooled:
        y_pool = 0
        eff = pooled["effect"]
        lo = pooled["ci_lower"]
        hi = pooled["ci_upper"]
        diamond = patches.Polygon(
            [(lo, y_pool), (eff, 0.25), (hi, y_pool), (eff, -0.25)],
            facecolor="#d62728",
            edgecolor="#d62728",
            alpha=0.85,
            zorder=4,
        )
        ax.add_patch(diamond)
        ax.axhline(y=0.5, color="#cccccc", linestyle="-", linewidth=0.5, zorder=0)
    # Labels
    names = [s["name"][:40] for s in studies]
    labels = names + (["Pooled"] if pooled else [])
    y_labels = y_positions + ([0] if pooled else [])
    ax.set_yticks(y_labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel(effect_label)
    ax.set_title(title, fontsize=11)
    ax.set_xlim(x_min, x_max)
    ax.grid(axis="x", alpha=0.3, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


# =============================================================================
# Funnel plot (publication-bias visualization, matplotlib)
# =============================================================================
def render_funnel_plot_png(
    funnel_data: list[dict],
    egger_intercept: float | None = None,
    pooled_effect: float | None = None,
    title: str = "Funnel plot (publication-bias check)",
    output_path: str | Path = "funnel.png",
) -> Path:
    """Render funnel plot from per-study (effect, se) data.
    funnel_data: list of {effect, se, weight}.
    egger_intercept: optional, shows regression line.
    pooled_effect: optional, shows center vertical line.
    Returns output path."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not funnel_data:
        return Path(output_path)
    fig, ax = plt.subplots(figsize=(6, 5))
    effects = [d["effect"] for d in funnel_data]
    ses = [d["se"] for d in funnel_data]
    ax.scatter(effects, ses, s=40, color="#1f77b4", alpha=0.7, edgecolors="white", linewidth=0.5)
    # Pooled effect vertical line
    if pooled_effect is not None:
        ax.axvline(x=pooled_effect, color="#666666", linestyle="--", linewidth=0.8)
    # Pseudo-95% confidence funnel (effect ± 1.96*SE)
    if pooled_effect is not None and ses:
        se_range = np.linspace(max(min(ses) * 0.5, 0.001), max(ses) * 1.2, 100)
        upper = pooled_effect + 1.96 * se_range
        lower = pooled_effect - 1.96 * se_range
        ax.plot(upper, se_range, color="#999999", linestyle="--", linewidth=0.8)
        ax.plot(lower, se_range, color="#999999", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Effect size")
    ax.set_ylabel("Standard error (precision ↓ → top)")
    ax.invert_yaxis()  # high precision (low SE) on top
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out


if __name__ == "__main__":
    # Self-test renderers
    sample_papers = [
        {
            "doi": "10.1/x",
            "title": "Alpha paper",
            "year": 2020,
            "referenced_works": ["10.2/y", "10.3/z"],
            "concepts": ["ML", "Climate"],
            "authors": [{"name": "Alice"}, {"name": "Bob"}],
        },
        {
            "doi": "10.2/y",
            "title": "Beta paper",
            "year": 2019,
            "referenced_works": [],
            "concepts": ["ML"],
            "authors": [{"name": "Alice"}],
        },
        {
            "doi": "10.3/z",
            "title": "Gamma paper",
            "year": 2021,
            "referenced_works": ["10.2/y"],
            "concepts": ["Climate"],
            "authors": [{"name": "Charlie"}, {"name": "Bob"}],
        },
    ]
    print("=== Citation graph ===")
    print(render_citation_graph(sample_papers))
    print("\n=== PRISMA flow ===")
    print(render_prisma_flow(100, 20, 80, 30, 50, 5, 45, 15, 30))
    print("\n=== Temporal histogram ===")
    print(render_temporal_histogram(sample_papers))
    print("\n=== Concept clusters ===")
    print(render_concept_clusters(sample_papers))
    print("\n=== Author network ===")
    print(render_author_network(sample_papers))


def render_prisma_png(
    identification: int,
    duplicates_removed: int,
    screened: int,
    excluded_at_screening: int,
    included: int,
    output_path: str = "research_outputs/prisma.png",
) -> str:
    """Render PRISMA 2020 flow diagram as PNG (matplotlib).

    Simplified 3-phase: Identification → Screening → Included.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("PRISMA 2020 Flow Diagram", fontsize=14, fontweight="bold")

    # Boxes
    boxes = [
        (
            2,
            7,
            6,
            1.5,
            f"Identification\nRecords identified (n={identification})\nDuplicates removed (n={duplicates_removed})",
        ),
        (
            2,
            4.5,
            6,
            1.5,
            f"Screening\nRecords screened (n={screened})\nRecords excluded (n={excluded_at_screening})",
        ),
        (2, 2, 6, 1.5, f"Included\nStudies included in review (n={included})"),
    ]
    for x, y, w, h, text in boxes:
        rect = mpatches.FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.1",
            facecolor="#e8f0fe",
            edgecolor="#4285f4",
            linewidth=1.5,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    # Arrows
    for y_start in [7, 4.5]:
        ax.annotate(
            "",
            xy=(5, y_start - 0.5),
            xytext=(5, y_start),
            arrowprops=dict(arrowstyle="->", color="#4285f4", lw=2),
        )

    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
