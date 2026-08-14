"""Visualization module for research pipeline output.

Generates matplotlib figures for:
    - Measurement density plots (KDE + histogram)
    - Method comparison radar
    - Evidence pyramid
    - Citation network (via citation_graph integration)
    - Research gap map

All figures are saved as PNG and paths returned for GUI embedding.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

log = logging.getLogger("scientific_research.viz")


def plot_measurement_density(
    measurements: dict[str, list[float]],
    output_path: str | Path,
    title: str = "Measurement Distributions",
) -> str | None:
    """Generate KDE + histogram density plots for extracted measurements.

    Args:
        measurements: {"temperature": [750, 800, 850], "pressure": [8.5, 10.2]}
        output_path: PNG file path
        title: plot title

    Returns:
        Path string if successful, None if failed
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # headless
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        log.debug("matplotlib unavailable for density plot")
        return None

    valid = {k: v for k, v in measurements.items() if len(v) >= 2}
    if not valid:
        return None

    n = len(valid)
    cols = min(n, 2)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, (mtype, values) in enumerate(valid.items()):
        ax = axes[idx // cols][idx % cols]
        arr = np.array(values, dtype=float)

        # Histogram
        ax.hist(arr, bins=min(15, max(5, len(arr) // 2)), density=True, alpha=0.6, color="steelblue", edgecolor="white")

        # KDE overlay
        if len(arr) >= 5:
            try:
                from scipy.stats import gaussian_kde
                kde = gaussian_kde(arr)
                x_range = np.linspace(arr.min() - arr.std(), arr.max() + arr.std(), 100)
                ax.plot(x_range, kde(x_range), "r-", linewidth=2, label="KDE")
                ax.legend(fontsize=9)
            except Exception:
                pass

        # Median line
        median = float(np.median(arr))
        ax.axvline(median, color="green", linestyle="--", linewidth=1.5, label=f"Median: {median:.1f}")
        ax.legend(fontsize=8)

        ax.set_xlabel(mtype.title(), fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{mtype.title()} (n={len(arr)})", fontsize=11)

    # Hide unused subplots
    for idx in range(n, rows * cols):
        axes[idx // cols][idx % cols].set_visible(False)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Density plot saved: %s", output_path)
    return str(output_path)


def plot_evidence_pyramid(
    claims: list[dict[str, Any]],
    output_path: str | Path,
    title: str = "Evidence Pyramid",
) -> str | None:
    """Generate an evidence pyramid showing claim strength distribution.

    Layers (bottom→top): D (very low) → C (low) → B (moderate) → A (high)
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.patches as mpatches
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    grades = ["D", "C", "B", "A"]
    colors = ["#ff6b6b", "#ffd93d", "#6bcf7f", "#4d96ff"]
    labels = ["Very Low (D)", "Low (C)", "Moderate (B)", "High (A)"]
    counts = [sum(1 for c in claims if c.get("evidence_grade") == g) for g in grades]

    if sum(counts) == 0:
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    max_width = 1.0
    layer_height = 0.2

    for i, (grade, count, color, label) in enumerate(zip(grades, counts, colors, labels)):
        width = max_width * (i + 1) / len(grades)
        x_center = 0.5
        y_bottom = i * layer_height
        triangle = mpatches.FancyBboxPatch(
            (x_center - width / 2, y_bottom),
            width,
            layer_height * 0.9,
            boxstyle="round,pad=0.01",
            facecolor=color,
            edgecolor="white",
            linewidth=2,
        )
        ax.add_patch(triangle)
        ax.text(x_center, y_bottom + layer_height * 0.45, f"{label}\nn={count}", ha="center", va="center", fontsize=9, fontweight="bold")

    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.05, len(grades) * layer_height + 0.1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(output_path)


def sensitivity_analysis_jackknife(
    values: list[float],
    statistic: str = "median",
) -> dict[str, float]:
    """Leave-one-out jackknife sensitivity analysis.

    Returns:
        {"jackknife_mean": ..., "jackknife_std": ..., "sensitivity": ...}
        where sensitivity = how much the statistic changes when each point is removed.
    """
    import statistics as stats_mod

    if len(values) < 3:
        return {"jackknife_mean": 0.0, "jackknife_std": 0.0, "sensitivity": "n/a"}

    def compute(vals):
        if statistic == "median":
            return stats_mod.median(vals)
        elif statistic == "mean":
            return stats_mod.mean(vals)
        else:
            return stats_mod.median(vals)

    full_stat = compute(values)
    jackknife_stats = []
    for i in range(len(values)):
        reduced = values[:i] + values[i + 1:]
        jackknife_stats.append(compute(reduced))

    jk_mean = stats_mod.mean(jackknife_stats)
    jk_std = stats_mod.stdev(jackknife_stats) if len(jackknife_stats) >= 2 else 0.0

    # Sensitivity: max % change from removing one point
    max_change = max(abs(js - full_stat) for js in jackknife_stats)
    sensitivity_pct = (max_change / abs(full_stat) * 100) if full_stat != 0 else 0.0

    return {
        "jackknife_mean": round(jk_mean, 2),
        "jackknife_std": round(jk_std, 2),
        "sensitivity": f"±{sensitivity_pct:.1f}%",
        "full_statistic": round(full_stat, 2),
        "n": len(values),
    }
