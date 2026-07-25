"""
Graph metrics — turning structure into ranked insight (Section 1B: discovery).

Centrality answers "which companies are the most systemically important nodes?"
A high out-centrality name is a *shock source* (its moves ripple outward); a high
in-centrality name is a *shock sink* (everyone's moves land on it).

STATUS: functional.
"""
from __future__ import annotations

import networkx as nx
import pandas as pd


def centrality_table(g: nx.DiGraph) -> pd.DataFrame:
    """Per-node centrality summary, sorted by systemic importance (PageRank)."""
    if g.number_of_edges() == 0:
        return pd.DataFrame()
    pr = nx.pagerank(g, weight="weight")
    out_deg = dict(g.out_degree(weight="weight"))
    in_deg = dict(g.in_degree(weight="weight"))
    try:
        btw = nx.betweenness_centrality(g, weight="weight")
    except Exception:
        btw = {n: float("nan") for n in g.nodes}

    rows = []
    for n in g.nodes:
        rows.append({
            "ticker": n,
            "name": g.nodes[n].get("name", n),
            "sector": g.nodes[n].get("sector", "Unknown"),
            "pagerank": pr.get(n, 0.0),
            "out_strength": out_deg.get(n, 0.0),   # shock-source score
            "in_strength": in_deg.get(n, 0.0),     # shock-sink score
            "betweenness": btw.get(n, 0.0),        # bridge score
        })
    return pd.DataFrame(rows).sort_values("pagerank", ascending=False).reset_index(drop=True)


def hidden_linkages(g: nx.DiGraph) -> list[tuple[str, str, dict]]:
    """
    The 'X-factor' discovery report (dream feature): statistically-supported
    directed edges that are NOT explained by same-sector or same-group
    membership — i.e. the non-obvious connections.
    """
    out = []
    for u, v, d in g.edges(data=True):
        same_sector = g.nodes[u].get("sector") == g.nodes[v].get("sector")
        same_group = g.nodes[u].get("group") and g.nodes[u].get("group") == g.nodes[v].get("group")
        kinds = {e["kind"] for e in d.get("evidence", [])}
        statistical = any(k in kinds for k in ("transfer_entropy", "cointegration"))
        if statistical and not same_sector and not same_group:
            out.append((u, v, d))
    return sorted(out, key=lambda e: e[2].get("weight", 0.0), reverse=True)
