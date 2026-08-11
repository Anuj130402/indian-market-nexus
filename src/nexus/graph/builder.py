from __future__ import annotations
import sys
from pathlib import Path
import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings, universe
from nexus.stats import cointegration, correlation, transfer_entropy


def _add_edge_evidence(g, u, v, kind, weight):
    if g.has_edge(u, v):
        g[u][v]["evidence"].append({"kind": kind, "weight": weight})
        g[u][v]["weight"] = g[u][v].get("weight", 0.0) + abs(weight)
    else:
        g.add_edge(u, v, weight=abs(weight), evidence=[{"kind": kind, "weight": weight}])


def seed_group_edges(g):
    by_group = {}
    for ticker, group in universe.GROUPS.items():
        if ticker in g:
            by_group.setdefault(group, []).append(ticker)
    for members in by_group.values():
        for a in members:
            for b in members:
                if a != b:
                    _add_edge_evidence(g, a, b, "same_group", 1.0)


def build_statistical_graph(prices, returns, thresholds=settings.EdgeThresholds,
                            n_surrogates=None):
    """
    Build the directed graph from PRICE data. `n_surrogates` overrides
    thresholds.TE_SURROGATES when given (lets rolling/eval callers trade rigour
    for speed without editing config).
    """
    n_surr = thresholds.TE_SURROGATES if n_surrogates is None else n_surrogates

    prices = prices.dropna(axis=1, how="all")
    returns = returns.reindex(columns=prices.columns)

    g = nx.DiGraph()
    for t in prices.columns:
        g.add_node(t,
                   name=universe.NIFTY50.get(t, t),
                   sector=universe.SECTORS.get(t, "Unknown"),
                   group=universe.GROUPS.get(t, ""))

    seed_group_edges(g)

    for a, b, pval in cointegration.cointegrated_pairs(
        prices, thresholds.COINTEGRATION_PVALUE, thresholds.MIN_OBSERVATIONS):
        strength = 1.0 - pval
        _add_edge_evidence(g, a, b, "cointegration", strength)
        _add_edge_evidence(g, b, a, "cointegration", strength)

    for src, dst, te, pval in transfer_entropy.significant_te_pairs(
        returns, alpha=thresholds.TRANSFER_ENTROPY_ALPHA, n_surrogates=n_surr):
        _add_edge_evidence(g, src, dst, "transfer_entropy", te)

    corr = correlation.correlation_matrix(returns)
    for u, v in list(g.edges()):
        if u in corr.columns and v in corr.columns:
            g[u][v]["corr"] = float(corr.loc[u, v])
    return g


# ----------------------- A: rolling / point-in-time graphs -----------------------
def build_graph_asof(prices, returns, as_of, lookback=None, n_surrogates=None):
    """
    Build the graph AS IT WOULD HAVE LOOKED on `as_of`, using only the trailing
    `lookback` trading days up to and including `as_of`. This is the core
    look-ahead-safe primitive: no data after `as_of` touches the graph.
    """
    lookback = lookback or settings.Windows.ROLLING_LOOKBACK
    as_of = pd.Timestamp(as_of)
    win_prices = prices.loc[:as_of].tail(lookback)
    if len(win_prices) < 2:
        return build_statistical_graph(win_prices, returns.loc[:as_of].tail(lookback),
                                       n_surrogates=n_surrogates)
    win_returns = returns.loc[win_prices.index[0]:as_of]
    return build_statistical_graph(win_prices, win_returns, n_surrogates=n_surrogates)


def rolling_graphs(prices, returns, lookback=None, step=None, n_surrogates=None):
    """
    Yield (as_of_date, graph) on a periodic grid. Each graph sees only its own
    trailing window, so relationships strengthen, decay, and die over time
    instead of being frozen over all history.
    """
    lookback = lookback or settings.Windows.ROLLING_LOOKBACK
    step = step or settings.Windows.ROLLING_STEP
    dates = prices.index
    for i in range(lookback, len(dates), step):
        as_of = dates[i]
        yield as_of, build_graph_asof(prices, returns, as_of, lookback, n_surrogates)


def _graphml_safe(d):
    for k, v in list(d.items()):
        if v is None:
            d[k] = ""


def save_graph(g, path=None):
    path = path or (settings.ARTIFACTS_DIR / "nexus_graph.graphml")
    h = g.copy()
    for _n, d in h.nodes(data=True):
        _graphml_safe(d)
    for _u, _v, d in h.edges(data=True):
        ev = d.pop("evidence", [])
        d["evidence_kinds"] = ",".join(sorted({e["kind"] for e in ev}))
        _graphml_safe(d)
    nx.write_graphml(h, path)
    return path
