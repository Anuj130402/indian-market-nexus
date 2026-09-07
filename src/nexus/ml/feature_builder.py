"""
Feature engineering for the aftershock model.

Each row = one (shock in A on day t -> graph-neighbour B) prediction.
14 base features (shock / edge / target-node / market). Optionally joins 5 NEWS
features describing the SOURCE stock A's news on the shock day t -- the causal
context the price data alone can't see.

LOOK-AHEAD: news is contemporaneous with the shock (both day-t events) and we
predict the same forward window as the price-only baseline, so adding it is a
controlled comparison, not a leak.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import networkx as nx

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings
from nexus.propagation import shock_model

VOL = settings.Windows.VOL_LOOKBACK
NEWS_COLS = ["news_count", "news_sentiment", "news_neg_ratio", "news_worst", "news_abs_max"]


def _centrality(graph):
    pr = nx.pagerank(graph, weight="weight") if graph.number_of_edges() else {}
    return pr, dict(graph.in_degree(weight="weight")), dict(graph.out_degree(weight="weight"))


def _news_lookup(news: pd.DataFrame | None):
    """(ticker, normalized-date) -> {news feature: value}, for O(1) row joins."""
    if news is None or news.empty:
        return {}
    n = news.copy()
    n["date"] = pd.to_datetime(n["date"], errors="coerce").dt.normalize()
    n = n.dropna(subset=["date"])
    return {(r["ticker"], r["date"]): {c: r[c] for c in NEWS_COLS if c in r}
            for _, r in n.iterrows()}


def build_dataset(prices, returns, graph, start, end, horizon=None, news=None):
    horizon = settings.Shock.HORIZON_DAYS if horizon is None else horizon
    pr, ins, outs = _centrality(graph)
    mkt = returns.mean(axis=1)
    mkt_vol = mkt.rolling(VOL).std()
    news_map = _news_lookup(news)
    use_news = bool(news_map)

    shocks = shock_model.detect_shocks(returns, start=start, end=end)
    rows, labels, meta = [], [], []
    for ev in shocks:
        if ev.ticker not in graph:
            continue
        t = ev.date
        a_vol = returns[ev.ticker].loc[:t].tail(VOL).std()
        mret, mvol = float(mkt.get(t, np.nan)), float(mkt_vol.get(t, np.nan))
        # SOURCE-stock news on the shock day (neutral defaults if none that day)
        nf = news_map.get((ev.ticker, pd.Timestamp(t).normalize()),
                          {c: 0.0 for c in NEWS_COLS}) if use_news else {}
        for _a, b, d in graph.out_edges(ev.ticker, data=True):
            fwd = shock_model.forward_return(prices, b, t, horizon)
            if np.isnan(fwd):
                continue
            kinds = {e["kind"] for e in d.get("evidence", [])}
            b_vol = returns[b].loc[:t].tail(VOL).std() if b in returns else np.nan
            row = {
                "shock_abs": abs(ev.magnitude),
                "shock_sign": 1.0 if ev.magnitude > 0 else -1.0,
                "source_vol": float(a_vol) if pd.notna(a_vol) else 0.0,
                "edge_weight": float(d.get("weight", 0.0)),
                "edge_corr": float(d.get("corr", 0.0)),
                "has_transfer_entropy": float("transfer_entropy" in kinds),
                "has_cointegration": float("cointegration" in kinds),
                "has_same_group": float("same_group" in kinds),
                "target_pagerank": pr.get(b, 0.0),
                "target_in_strength": ins.get(b, 0.0),
                "target_out_strength": outs.get(b, 0.0),
                "target_vol": float(b_vol) if pd.notna(b_vol) else 0.0,
                "mkt_return": mret if pd.notna(mret) else 0.0,
                "mkt_vol": mvol if pd.notna(mvol) else 0.0,
            }
            if use_news:
                for c in NEWS_COLS:
                    row[c] = float(nf.get(c, 0.0))
            rows.append(row)
            labels.append(int(fwd > 0))
            meta.append({"date": t, "source": ev.ticker, "target": b, "fwd_return": fwd})

    return pd.DataFrame(rows), pd.Series(labels, name="target_up"), pd.DataFrame(meta)