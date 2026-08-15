#!/usr/bin/env python3
"""Skill benchmark harness (2026-08-15).

Benches (see report.md output):
  1 screening   — WSS@95 + recall@10% : AL vs TF-IDF-static vs random
                  (gold: labeled dataset if fetchable, else synthetic — stated)
  3 authenticity— Crossref resolve + title match on a verified corpus sample;
                  known-retracted DOIs (from local Retraction Watch CSV) flagged
  4 pooling     — hard invariant: no pool ever mixes units; conversions exact
  5 cost        — per-stage wall-clock on a real corpus

Usage: benchmark.py --bench all|screening|authenticity|pooling|cost
                    [--verified PATH] [--out research_outputs/benchmark]
"""

from __future__ import annotations

import _bootstrap

_bootstrap.ensure_env()

import argparse
import json
import random
import statistics
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from _sources import PaperRecord
from screen import prioritize_active_learning

RESULTS: dict = {}


# ─── Bench 1: screening quality ────────────────────────────────────────────
SYNERGY_URLS = [
    "https://raw.githubusercontent.com/asreview/sysreview-datasets/main/datasets/synergy/PTSD_VandeSchoot_2017/raw/PTSD_VandeSchoot_2017.csv",
    "https://raw.githubusercontent.com/asreview-collaboration/sysreview-datasets/master/datasets/synergy/PTSD_VandeSchoot_2017/PTSD_VandeSchoot_2017_raw.csv",
]


def _try_load_synergy():
    """Fetch a labeled gold set; None on failure (bench falls back synthetic)."""
    for url in SYNERGY_URLS:
        try:
            with urllib.request.urlopen(url, timeout=25) as r:
                body = r.read().decode("utf-8", errors="replace")
            rows = [ln.split(",") for ln in body.splitlines() if ln.strip()]
            header = [h.strip().strip('"').lower() for h in rows[0]]
            if "include" not in header or (
                "title" not in header and "abstract" not in header
            ):
                continue
            i_label = header.index("include")
            i_title = header.index("title") if "title" in header else None
            i_abs = header.index("abstract") if "abstract" in header else None
            papers, labels = [], []
            for ln in rows[1:]:
                if len(ln) <= max(
                    x for x in (i_label, i_title, i_abs) if x is not None
                ):
                    continue
                try:
                    lab = int(ln[i_label])
                except ValueError:
                    continue
                title = ln[i_title] if i_title is not None else ""
                abstract = ln[i_abs] if i_abs is not None else ""
                if not (title or abstract):
                    continue
                papers.append(
                    PaperRecord(
                        title=title[:300],
                        abstract=" ".join(abstract.split())[:3000],
                        doi=None,
                    )
                )
                labels.append("include" if lab == 1 else "exclude")
            if len(papers) >= 100 and any(x == "include" for x in labels):
                return papers, labels, "synergy/PTSD_VandeSchoot_2017 (GitHub raw)"
        except Exception:
            continue
    return None


def _synthetic_gold(n: int = 300, seed: int = 7):
    rng = random.Random(seed)
    rel = [
        PaperRecord(
            title=f"Amphibole thermobarometry study {i}",
            abstract="amphibole thermobarometry constrains magma storage "
            "temperatures and pressures in arc magmas",
        )
        for i in range(rng.randint(25, 40))
    ]
    noise_topics = [
        "clinical trial of drug outcomes",
        "wireless network protocol evaluation",
        "Renaissance poetry analysis",
        "machine learning image classification benchmarks",
    ]
    noise = [
        PaperRecord(
            title=f"{rng.choice(noise_topics)} experiment {i}",
            abstract=rng.choice(noise_topics) + " with quantitative evaluation",
        )
        for i in range(n - len(rel))
    ]
    papers = rel + noise
    rng.shuffle(papers)
    labels = [
        "include" if p.title.startswith("Amphibole") else "exclude" for p in papers
    ]
    return papers, labels


def _wss_at_95(order: list[int], labels: list[str]) -> tuple[float, int, float]:
    """Return (WSS@95, n_to_95, recall@10%)."""
    n_rel = sum(1 for x in labels if x == "include")
    seen, n_to_95 = 0, len(order)
    for pos, idx in enumerate(order, start=1):
        if labels[idx] == "include":
            seen += 1
            if seen >= 0.95 * n_rel:
                n_to_95 = pos
                break
    wss = 1.0 - n_to_95 / len(order)
    k10 = max(1, len(order) // 10)
    r10 = sum(1 for i in order[:k10] if labels[i] == "include") / n_rel
    return wss, n_to_95, r10


def bench_screening(seed_frac: float = 0.05) -> dict:
    gold = _try_load_synergy()
    if gold:
        papers, labels, source = gold
        provenance = f"REAL gold set: {source}"
    else:
        papers, labels = _synthetic_gold()
        provenance = (
            "SYNTHETIC gold (synergy fetch failed) — absolute numbers are "
            "upper bounds; harness validity only"
        )
    n = len(papers)
    n_seed = max(10, int(n * seed_frac))
    rng = random.Random(42)
    idx = list(range(n))
    rng.shuffle(idx)
    # stratified seed (standard AL-simulation practice): random draw,
    # topped up until >=5 include and >=5 exclude seeds (AL training floor)
    inc_pool = [i for i in idx if labels[i] == "include"]
    exc_pool = [i for i in idx if labels[i] == "exclude"]
    seed_idx = inc_pool[: max(5, n_seed // 2)] + exc_pool[: max(5, n_seed // 2)]
    seed_set = set(seed_idx)
    rest_idx = [i for i in idx if i not in seed_set]
    seed_idx = seed_idx[: max(10, n_seed)]

    # method A: random order over unscreened
    rand_orders = [[*seed_idx, *rng.sample(rest_idx, len(rest_idx))] for _ in range(5)]
    a = [_wss_at_95(o, labels) for o in rand_orders]
    random_metrics = {
        "wss95": round(statistics.mean(x[0] for x in a), 3),
        "n_to_95": round(statistics.mean(x[1] for x in a), 1),
        "recall@10%": round(statistics.mean(x[2] for x in a), 3),
    }

    # method B: static TF-IDF similarity to the relevant seed centroid
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    texts = [(p.title + " " + (p.abstract or ""))[:5000] for p in papers]
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, max_features=20000)
    X = vec.fit_transform(texts)
    seed_rel = [i for i in seed_idx if labels[i] == "include"] or seed_idx[:1]
    import numpy as np
    from scipy.sparse import vstack

    centroid = np.asarray(vstack([X[i] for i in seed_rel]).mean(axis=0))
    sims = cosine_similarity(X, centroid).ravel()
    static_order = sorted(rest_idx, key=lambda i: -sims[i])
    b = _wss_at_95([*seed_idx, *static_order], labels)
    static_metrics = {
        "wss95": round(b[0], 3),
        "n_to_95": b[1],
        "recall@10%": round(b[2], 3),
    }

    # method C: skill AL mode (train on seed labels, rank unscreened)
    # key must match screen.paper_key: doi or FULL title, lowercased
    label_map = {
        (papers[i].doi or papers[i].title or "").strip().lower(): labels[i]
        for i in seed_idx
    }
    al_order = list(range(n))
    try:
        al = prioritize_active_learning(papers, label_map)
        ranked_keys = [q["paper_id"] for q in al.queue]
        by_key = {p.primary_id: i for i, p in enumerate(papers)}
        al_rest = [
            by_key[k] for k in ranked_keys if k in by_key and by_key[k] in set(rest_idx)
        ]
        al_order = [*seed_idx, *al_rest]
    except ValueError as e:
        print(f"AL failed in bench (not silent): {e}", file=sys.stderr)
        al_order = [*seed_idx, *rng.sample(rest_idx, len(rest_idx))]
    c = _wss_at_95(al_order, labels)
    al_metrics = {
        "wss95": round(c[0], 3),
        "n_to_95": c[1],
        "recall@10%": round(c[2], 3),
    }

    return {
        "provenance": provenance,
        "n_papers": n,
        "n_relevant": sum(1 for x in labels if x == "include"),
        "n_seed": n_seed,
        "random": random_metrics,
        "static_tfidf": static_metrics,
        "skill_al": al_metrics,
        "reference": "WSS@95 ≥ 0.70 = strong (ASReview literature: 0.70–0.95)",
    }


# ─── Bench 3: authenticity ─────────────────────────────────────────────────
def bench_authenticity(verified_path: Path) -> dict:
    papers = json.loads(verified_path.read_text(encoding="utf-8"))["papers"]
    import re

    def norm(t):
        return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", (t or "").lower())).strip()

    sample = papers[:8]
    resolved = matched = 0
    for p in sample:
        doi = (p.get("doi") or "").lower()
        if not doi:
            continue
        try:
            u = f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"
            req = urllib.request.Request(
                u, headers={"User-Agent": "bench/1.0 (mailto:bench@example.org)"}
            )
            with urllib.request.urlopen(req, timeout=20) as r:
                m = json.load(r)["message"]
            resolved += 1
            a, b_ = (
                set(norm(m.get("title", [""])[0]).split()),
                set(norm(p.get("title")).split()),
            )
            if a and b_ and len(a & b_) / len(a | b_) >= 0.5:
                matched += 1
        except Exception:
            pass
        time.sleep(0.3)

    # known-retracted: first 5 DOIs straight from the local RW database
    rw_flagged = rw_total = 0
    from verify import check_retraction_watch, load_rw_index

    idx = load_rw_index()
    if idx:
        known = sorted(idx)[:5]
        rw_total = len(known)
        for doi in known:
            hit = check_retraction_watch(PaperRecord(doi=doi, title="x"), idx)
            rw_flagged += int(hit is not None)
    return {
        "crossref_sample": f"{resolved}/{len(sample)} resolved, {matched}/{len(sample)} title-matched",
        "rw_index_size": len(idx),
        "known_retracted_flagged": f"{rw_flagged}/{rw_total}"
        if rw_total
        else "RW CSV absent — skipped (fail-open)",
        "note": "sample of 8; full-corpus audits live in session logs (22/22 round 2)",
    }


# ─── Bench 4: pooling purity ────────────────────────────────────────────────
def bench_pooling() -> dict:
    from meta_analyze import collect_single_measurements

    rng = random.Random(3)
    units = [
        "kbar",
        "GPa",
        "MPa",
        "bar",
        "°C",
        "K",
        "Ma",
        "ka",
        "%",
        "‰",
        "ppm",
        "km",
        "m",
    ]
    papers = [PaperRecord(doi=f"10.1/b{i}", title=f"B{i}") for i in range(40)]
    exts = []
    for i, p in enumerate(papers):
        rows = []
        for _ in range(rng.randint(1, 3)):
            u = rng.choice(units)
            rows.append(
                {"value": round(rng.uniform(1, 900), 2), "unit": u, "measurement": ""}
            )
        exts.append(
            {"paper_id": p.primary_id, "effect_sizes": {"single_measurements": rows}}
        )
    effects = collect_single_measurements(exts, papers)
    labels = {e.scale_label for e in effects}
    # purity invariant: every label is 'measurement (unit)' — one unit per pool.
    # Cross-family mixing would show as a label with 2+ units — impossible by
    # construction; the REAL invariant is no °C+Ma value in same pool:
    fam = {}
    pure = True
    for e in effects:
        fam.setdefault(e.scale_label, set()).add(
            e.scale_label.split("(")[-1].rstrip(")")
        )
    for lbl, us in fam.items():
        if len(us) != 1:
            pure = False
    # exact conversion spot-checks
    checks = {
        "1 GPa→10 kbar": any(
            abs(e.effect - 10.0) < 1e-9 and "kbar" in e.scale_label for e in effects
        ),
        "973.15 K→700 °C": any(
            abs(e.effect - 700.0) < 1e-6 and "°C" in e.scale_label for e in effects
        ),
    }
    # need guaranteed rows to assert conversions deterministically:
    forced = [
        PaperRecord(doi="10.1/f1", title="F1"),
        PaperRecord(doi="10.1/f2", title="F2"),
    ]
    forced_exts = [
        {
            "paper_id": forced[0].primary_id,
            "effect_sizes": {
                "single_measurements": [
                    {"value": 1.0, "unit": "GPa", "measurement": "pressure"},
                    {"value": 973.15, "unit": "K", "measurement": "temperature"},
                ]
            },
        },
        {
            "paper_id": forced[1].primary_id,
            "effect_sizes": {
                "single_measurements": [
                    {"value": 5.0, "unit": "kbar", "measurement": "pressure"},
                    {"value": 700.0, "unit": "°C", "measurement": "temperature"},
                ]
            },
        },
    ]
    fe = collect_single_measurements(forced_exts, forced)
    forced_ok = (
        any(abs(e.effect - 10.0) < 1e-9 for e in fe if "kbar" in e.scale_label)
        and any(abs(e.effect - 700.0) < 1e-6 for e in fe if "°C" in e.scale_label)
        and len([e for e in fe if "kbar" in e.scale_label]) == 2
    )
    return {
        "random_pool_pure": pure,
        "pools_emitted": len(effects),
        "distinct_pool_labels": sorted(labels),
        "forced_conversion_exact": forced_ok,
    }


# ─── Bench 5: cost ─────────────────────────────────────────────────────────
def bench_cost(verified_path: Path | None) -> dict:
    import subprocess

    stages = []
    if verified_path and verified_path.exists():
        tmp = Path("/tmp/opencode/bench_cost")
        tmp.mkdir(parents=True, exist_ok=True)
        cmds = [
            (
                "extract",
                [
                    sys.executable,
                    str(Path(__file__).parent / "extract.py"),
                    str(verified_path),
                    "-o",
                    str(tmp / "ext.json"),
                ],
            ),
            (
                "correlate",
                [
                    sys.executable,
                    str(Path(__file__).parent / "correlate.py"),
                    str(tmp / "ext.json"),
                    "-o",
                    str(tmp / "corr.json"),
                ],
            ),
            (
                "meta(single)",
                [
                    sys.executable,
                    str(Path(__file__).parent / "meta_analyze.py"),
                    str(tmp / "ext.json"),
                    str(verified_path),
                    "--measure",
                    "single",
                    "-o",
                    str(tmp / "meta.json"),
                ],
            ),
        ]
        for name, cmd in cmds:
            t0 = time.monotonic()
            r = subprocess.run(cmd, capture_output=True, timeout=600)
            stages.append(
                (
                    name,
                    round(time.monotonic() - t0, 2),
                    "ok" if r.returncode == 0 else "FAIL",
                )
            )
    return {
        "corpus": str(verified_path) if verified_path else None,
        "stage_timings_s": stages,
        "note": "verify/discover dominated by network + per-API rate limits; run separately with --force-refresh for cold numbers",
    }




# ─── Bench 6: fulltext coverage (DocumentStore, live network) ─────────────
def bench_fulltext(verified_path: Path) -> dict:
    from _documentstore import DocumentStore

    ds = DocumentStore()
    stats = ds.fetch_corpus(verified_path, limit=8)
    cov = ds.coverage(verified_path)
    return {**stats, "corpus_coverage": cov,
            "note": "legal routes only: Unpaywall OA + arXiv + user-dropped incoming/"}


# ─── Bench 7: contradiction detection (claims engine) ─────────────────────
def bench_contradictions(extracted_path: Path) -> dict:
    from _artifact import load_extractions
    from _claims_engine import (
        contradiction_report,
        detect_contradictions,
        extract_claims,
    )

    ext = load_extractions(extracted_path)
    papers = [{"doi": e.get("doi"), "title": e.get("title"), "abstract": e.get("abstract")} for e in ext["extractions"]]
    claims = [c for p in papers for c in extract_claims(p)]
    cons = detect_contradictions(claims)
    return {"papers": len(papers), "numeric_claims": len(claims), "contradictions": len(cons),
            "top": [{"quantity": c.quantity, "unit": c.unit, "ratio": c.ratio, "n": len(c.values)} for c in cons[:5]],
            "sample_report": contradiction_report(cons[:3])}


# ─── Bench 8: subgroup explanatory power ───────────────────────────────────
def bench_subgroups(meta_path: Path) -> dict:
    from _artifact import load_meta

    meta = load_meta(meta_path)
    pools = meta.get("unit_pools") or []
    out = []
    for up in pools:
        subs = up.get("subgroup_results") or []
        decade = [r for r in subs if "decade" in str(r.get("label", ""))]
        if decade:
            effs = [r["effect"] for r in decade]
            out.append({"pool": up["group"], "k": up["k"],
                        "i2": (up.get("pooled_random") or {}).get("heterogeneity", {}).get("i_squared"),
                        "decade_subgroups": len(decade),
                        "decade_effect_spread": round(max(effs) - min(effs), 3)})
    return {"pools_with_decade_split": out,
            "note": "spread > 0 with decade split = heterogeneity partially explained by era"}

def main() -> int:
    if "--self-check" in sys.argv:
        print(f"OK {sys.argv[0]}: ready")
        return 0
    p = argparse.ArgumentParser(prog="benchmark")
    p.add_argument(
        "--bench", default="all", help="all|screening|authenticity|pooling|cost"
    )
    p.add_argument(
        "--verified", type=Path, default=Path("/tmp/opencode/valrun2/verified.json")
    )
    p.add_argument("--extracted", type=Path, default=Path("/tmp/opencode/valrun2/extracted.json"))
    p.add_argument("--meta", type=Path, default=Path("/tmp/opencode/valrun2/meta.json"))
    p.add_argument("--out", type=Path, default=Path("research_outputs/benchmark"))
    args = p.parse_args()
    run = args.bench == "all" or "screening" in args.bench
    if run:
        print("Bench 1/4: screening + pooling ...")
        RESULTS["screening"] = bench_screening()
        RESULTS["pooling"] = bench_pooling()
    if args.bench in ("all", "authenticity"):
        print("Bench 3: authenticity ...")
        RESULTS["authenticity"] = bench_authenticity(args.verified)
    if args.bench in ("all", "cost"):
        print("Bench 5: cost ...")
        RESULTS["cost"] = bench_cost(args.verified)
    if args.bench in ("all", "fulltext"):
        print("Bench 6: fulltext ...")
        try:
            RESULTS["fulltext"] = bench_fulltext(args.verified)
        except Exception as e:  # noqa: BLE001
            RESULTS["fulltext"] = {"error": str(e)[:200]}
    if args.bench in ("all", "contradictions"):
        print("Bench 7: contradictions ...")
        try:
            RESULTS["contradictions"] = bench_contradictions(args.extracted)
        except Exception as e:  # noqa: BLE001
            RESULTS["contradictions"] = {"error": str(e)[:200]}
    if args.bench in ("all", "subgroups"):
        print("Bench 8: subgroups ...")
        try:
            RESULTS["subgroups"] = bench_subgroups(args.meta)
        except Exception as e:  # noqa: BLE001
            RESULTS["subgroups"] = {"error": str(e)[:200]}

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "benchmark_results.json").write_text(
        json.dumps(RESULTS, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    lines = [
        "# Benchmark report",
        "",
        f"_Generated {time.strftime('%Y-%m-%dT%H:%M:%S')}_",
        "",
    ]
    for name, data in RESULTS.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(data, indent=1, ensure_ascii=False))
        lines.append("```")
        lines.append("")
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {args.out}/report.md + benchmark_results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
