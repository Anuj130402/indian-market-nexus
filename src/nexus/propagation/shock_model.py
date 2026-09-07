from __future__ import annotations
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings


@dataclass
class ShockEvent:
    ticker: str
    date: pd.Timestamp
    magnitude: float   # signed return that triggered it


def detect_shocks(returns, threshold=None, start=None, end=None):
    threshold = settings.Shock.THRESHOLD_RETURN if threshold is None else threshold
    r = returns
    if start is not None: r = r.loc[pd.Timestamp(start):]
    if end is not None:   r = r.loc[:pd.Timestamp(end)]
    events = []
    for ticker in r.columns:
        s = r[ticker].dropna()
        hits = s[s.abs() >= threshold]
        events += [ShockEvent(ticker, idx, float(v)) for idx, v in hits.items()]
    return events


def _forward_return(prices, ticker, date, horizon):
    """Cumulative return of `ticker` from `date` to `date`+horizon trading days."""
    if ticker not in prices.columns: return np.nan
    s = prices[ticker].dropna()
    future = s.loc[date:]
    if len(future) < 2: return np.nan
    end_idx = min(horizon, len(future) - 1)
    return float(future.iloc[end_idx] / future.iloc[0] - 1.0)


def evaluate_predictions(prices, returns, graph=None, graph_fn=None,
                         threshold=None, horizon=None, start=None, end=None):
    """
    THE GO/NO-GO GATE. For every shock in [start, end], use the graph's out-edges
    from the shocked node to predict which neighbours move, and in which
    direction, then check what actually happened over the next `horizon` days.

    Provide exactly one of:
      graph     : a single graph built ONLY from data before `start` (train/test split)
      graph_fn  : callable(as_of) -> graph (walk-forward; use builder.build_graph_asof)

    v1 hypothesis: same-direction contagion (a drop in A predicts a drop in B).

    Returns dict: hit_rate, baseline (0.5), lift, n_predictions, up_rate, detail(df).
    """
    if (graph is None) == (graph_fn is None):
        raise ValueError("Provide exactly one of `graph` or `graph_fn`.")
    horizon = settings.Shock.HORIZON_DAYS if horizon is None else horizon

    shocks = detect_shocks(returns, threshold, start, end)
    rows = []
    for ev in shocks:
        g = graph if graph is not None else graph_fn(ev.date - pd.Timedelta(days=1))
        if g is None or ev.ticker not in g:
            continue
        sign = 1.0 if ev.magnitude > 0 else -1.0
        for _a, b, d in g.out_edges(ev.ticker, data=True):
            fwd = _forward_return(prices, b, ev.date, horizon)
            if np.isnan(fwd):
                continue
            predicted_up = sign > 0
            actual_up = fwd > 0
            rows.append({
                "shock_ticker": ev.ticker, "date": ev.date, "target": b,
                "shock_sign": sign, "forward_return": fwd,
                "hit": bool(predicted_up == actual_up),
                "edge_weight": float(d.get("weight", 0.0)),
                "edge_kinds": ",".join(sorted({e["kind"] for e in d.get("evidence", [])})),
            })

    detail = pd.DataFrame(rows)
    if detail.empty:
        return {"hit_rate": float("nan"), "baseline": 0.5, "lift": float("nan"),
                "n_predictions": 0, "up_rate": float("nan"), "detail": detail}

    hit_rate = float(detail["hit"].mean())
    up_rate = float((detail["forward_return"] > 0).mean())
    return {
        "hit_rate": hit_rate,
        "baseline": 0.5,                       # coin-flip on direction
        "lift": hit_rate - 0.5,
        "n_predictions": int(len(detail)),
        "up_rate": up_rate,                    # context: market's own drift
        "detail": detail,
    }
forward_return = _forward_return