"""
Graph builder — where "The Eyes" (NLP) and "The Brain" (stats) become one graph.

The Nexus is a *directed* graph (Section 4d):
    node  = a company (ticker)
    edge  A -> B means "a shock in A tends to propagate to B"
    edge attributes carry WHY the edge exists and HOW strong it is.

The core design principle from the spec sheet: an edge should ideally be
confirmed by BOTH a structural reason (NLP/seed relationship) AND a statistical
signal. That AND is what stops the graph collapsing into a correlation hairball.
We record the evidence on each edge so nothing is a black box — you can always
ask "why is this edge here?".

STATUS: functional for statistical + seed edges. NLP edges plug in via
`add_nlp_edges` once nlp/ is implemented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings, universe  # noqa: E402
from nexus.stats import cointegration, correlation, transfer_entropy  # noqa: E402


def _add_edge_evidence(g: nx.DiGraph, u: str, v: str, kind: str, weight: float) -> None:
    """Attach or reinforce an edge, accumulating the evidence that supports it."""
    if g.has_edge(u, v):
        g[u][v]["evidence"].append({"kind": kind, "weight": weight})
        g[u][v]["weight"] = g[u][v].get("weight", 0.0) + abs(weight)
    else:
        g.add_edge(u, v, weight=abs(weight), evidence=[{"kind": kind, "weight": weight}])


def seed_group_edges(g: nx.DiGraph) -> None:
    """
    Seed the graph with hard, known relationships (same business group).
    These are the archive's 'Tata Steel -> Tata Motors' spine: high-prior edges
    we trust before any statistics. Bidirectional, since group contagion runs
    both ways until the data tells us otherwise.
    """
    by_group: dict[str, list[str]] = {}
    for ticker, group in universe.GROUPS.items():
        if ticker in g:  # only seed links between nodes that actually exist
            by_group.setdefault(group, []).append(ticker)
    for members in by_group.values():
        for a in members:
            for b in members:
                if a != b:
                    _add_edge_evidence(g, a, b, "same_group", 1.0)


def build_statistical_graph(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    thresholds: type[settings.EdgeThresholds] = settings.EdgeThresholds,
) -> nx.DiGraph:
    """
    Build the directed graph from PRICE data alone (no NLP yet). Combines:
      - cointegration  -> undirected structural link (added both directions)
      - transfer entropy -> the directional backbone (A -> B)
      - correlation    -> baseline weight only, never creates an edge by itself
    """
    g = nx.DiGraph()
    for t in prices.columns:
        g.add_node(
            t,
            name=universe.NIFTY50.get(t, t),
            sector=universe.SECTORS.get(t, "Unknown"),
            group=universe.GROUPS.get(t, None),
        )

    # 1) Seed with known group structure.
    seed_group_edges(g)

    # 2) Cointegration: symmetric structural evidence.
    for a, b, pval in cointegration.cointegrated_pairs(
        prices, thresholds.COINTEGRATION_PVALUE, thresholds.MIN_OBSERVATIONS
    ):
        strength = 1.0 - pval  # lower p => stronger
        _add_edge_evidence(g, a, b, "cointegration", strength)
        _add_edge_evidence(g, b, a, "cointegration", strength)

    # 3) Transfer entropy: the DIRECTED backbone. Threshold against a shuffled
    #    null so we only keep meaningful directed flows.
    te_edges = transfer_entropy.directed_te_pairs(returns)
    if te_edges:
        te_values = np.array([w for *_e, w in te_edges])
        cutoff = np.percentile(te_values, thresholds.TRANSFER_ENTROPY_PERCENTILE)
        for src, dst, te in te_edges:
            if te >= cutoff:
                _add_edge_evidence(g, src, dst, "transfer_entropy", te)

    # 4) Correlation: reinforce existing edges only (baseline, no new edges).
    corr = correlation.correlation_matrix(returns)
    for u, v in list(g.edges()):
        if u in corr.columns and v in corr.columns:
            g[u][v]["corr"] = float(corr.loc[u, v])

    return g


def add_nlp_edges(g: nx.DiGraph, relations: list[tuple[str, str, str, float]]) -> nx.DiGraph:
    """
    Fold NLP-extracted relations into an existing graph.

    `relations` is a list of (src_ticker, dst_ticker, relation_type, confidence)
    as produced by nexus.nlp.relation_extraction. STATUS: interface ready;
    the producing side is a stub until the NLP layer lands.
    """
    for src, dst, rel_type, conf in relations:
        _add_edge_evidence(g, src, dst, f"nlp:{rel_type}", conf)
    return g


def save_graph(g: nx.DiGraph, path: Path | None = None) -> Path:
    """Persist as GraphML (portable, opens in Gephi/Neo4j import too)."""
    path = path or (settings.ARTIFACTS_DIR / "nexus_graph.graphml")
    # GraphML can't store list/dict attrs, so serialise evidence to a summary.
    h = g.copy()
    for _u, _v, d in h.edges(data=True):
        ev = d.pop("evidence", [])
        d["evidence_kinds"] = ",".join(sorted({e["kind"] for e in ev}))
    nx.write_graphml(h, path)
    return path
