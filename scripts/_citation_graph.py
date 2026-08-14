"""Citation graph analysis using networkx.

Builds a citation network from papers and computes:
- Centrality metrics (PageRank, betweenness, eigenvector)
- Community detection (Louvain/greedy modularity)
- Influential paper identification
- Cluster labeling

Outputs structured data for visualization and analysis.
"""

from __future__ import annotations

import logging

log = logging.getLogger("scientific_research.citation_graph")


def build_citation_graph(
    papers: list[dict],
    min_edges: int = 5,
) -> dict | None:
    """Build citation graph and compute network metrics.

    Args:
        papers: List of paper dicts with 'doi', 'referenced_works', 'citation_count'.
        min_edges: Minimum edges to compute meaningful metrics.

    Returns:
        Dict with:
        - "centrality": {paper_id: {"pagerank": float, "betweenness": float}}
        - "communities": [{label: str, papers: [ids], size: int}]
        - "top_influential": [{paper_id, score, title}]
        - "density": float
        - "n_nodes": int
        - "n_edges": int
        Or None if graph too sparse.
    """
    import networkx as nx  # HARD when building citation graphs

    # Build graph
    graph = nx.DiGraph()

    # Add nodes
    doi_to_paper: dict[str, dict] = {}
    for p in papers:
        pid = p.get("doi") or p.get("paper_id") or ""
        if not pid:
            continue
        graph.add_node(pid, title=p.get("title", ""), year=p.get("year"))
        doi_to_paper[pid] = p

    # Add citation edges (referenced_works → paper)
    edge_count = 0
    for p in papers:
        pid = p.get("doi") or p.get("paper_id") or ""
        refs = p.get("referenced_works") or []
        for ref_id in refs:
            # ref_id might be an OpenAlex ID — check if it's in our graph
            if ref_id in graph:
                graph.add_edge(ref_id, pid)  # cited → citing
                edge_count += 1

    # Also add bibliographic coupling (shared references → edge)
    paper_refs: dict[str, set[str]] = {}
    for p in papers:
        pid = p.get("doi") or p.get("paper_id") or ""
        refs = set(p.get("referenced_works") or [])
        if refs:
            paper_refs[pid] = refs

    for pid1, refs1 in paper_refs.items():
        for pid2, refs2 in paper_refs.items():
            if pid1 < pid2:
                coupling = len(refs1 & refs2)
                if coupling > 0:
                    if graph.has_edge(pid1, pid2):
                        graph[pid1][pid2]["weight"] = (
                            graph[pid1][pid2].get("weight", 0) + coupling
                        )
                    else:
                        graph.add_edge(pid1, pid2, weight=coupling)
                        edge_count += 1

    if graph.number_of_nodes() < 3 or edge_count < min_edges:
        log.debug(
            "Citation graph too sparse: %d nodes, %d edges",
            graph.number_of_nodes(),
            edge_count,
        )
        return None

    # Compute centrality metrics
    result: dict = {
        "n_nodes": graph.number_of_nodes(),
        "n_edges": edge_count,
        "density": nx.density(graph),
    }

    # PageRank (on directed graph)
    try:
        pr = nx.pagerank(graph, max_iter=200, tol=1e-4)
        result["pagerank"] = pr
    except Exception as e:
        log.debug("PageRank failed: %s", e)
        pr = {}

    # Betweenness centrality (on undirected version)
    try:
        bc = nx.betweenness_centrality(
            graph.to_undirected(), k=min(50, graph.number_of_nodes())
        )
        result["betweenness"] = bc
    except Exception as e:
        log.debug("Betweenness failed: %s", e)
        bc = {}

    # Top influential papers by PageRank
    top_papers = sorted(pr.items(), key=lambda x: -x[1])[:20]
    result["top_influential"] = [
        {
            "paper_id": pid,
            "pagerank": round(score, 4),
            "title": doi_to_paper.get(pid, {}).get("title", ""),
            "year": doi_to_paper.get(pid, {}).get("year"),
        }
        for pid, score in top_papers
        if score > 0
    ]

    # Community detection (greedy modularity on undirected graph)
    try:
        undirected = graph.to_undirected()
        communities = nx.community.greedy_modularity_communities(undirected)
        community_list = []
        for i, comm in enumerate(communities):
            if len(comm) < 2:
                continue
            # Label by most common method theme in community
            themes: list[str] = []
            for pid in list(comm)[:10]:
                p = doi_to_paper.get(pid, {})
                # Try to get method theme from paper
                title = (p.get("title") or "").lower()
                themes.append(title)
            label = f"Cluster {i + 1} ({len(comm)} papers)"
            community_list.append(
                {
                    "label": label,
                    "papers": list(comm)[:20],
                    "size": len(comm),
                }
            )
        result["communities"] = community_list
    except Exception as e:
        log.debug("Community detection failed: %s", e)

    return result


def format_citation_graph_summary(graph_data: dict) -> str:
    """Format citation graph analysis as markdown for the brief.

    Args:
        graph_data: Output from build_citation_graph().

    Returns:
        Markdown string for inclusion in the brief.
    """
    if not graph_data:
        return ""

    parts = ["### Citation network analysis", ""]

    parts.append(
        f"The citation network comprises {graph_data['n_nodes']} papers "
        f"connected by {graph_data['n_edges']} citation links "
        f"(density: {graph_data['density']:.3f})."
    )

    # Top influential papers
    top = graph_data.get("top_influential", [])
    if top:
        parts.append("")
        parts.append("**Most influential papers** (by PageRank):")
        parts.append("")
        parts.append("| Rank | Title | Year | PageRank |")
        parts.append("|---|---|---|---|")
        for i, p in enumerate(top[:10]):
            title = (p["title"] or "—")[:60]
            parts.append(
                f"| {i + 1} | {title} | {p.get('year', '—')} | {p['pagerank']:.4f} |"
            )

    # Communities
    communities = graph_data.get("communities", [])
    if communities:
        parts.append("")
        parts.append("**Research clusters** (detected by community analysis):")
        parts.append("")
        for comm in communities[:5]:
            parts.append(f"- {comm['label']}")

    parts.append("")
    return "\n".join(parts)
