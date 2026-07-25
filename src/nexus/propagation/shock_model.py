"""
Shock propagation — "The Action" (Section 1A: the predictive signal).

Given a shock at node A on day t (a move >= Shock.THRESHOLD_RETURN), predict
which neighbours B experience an aftershock within Shock.HORIZON_DAYS, and in
which direction. This directed, weighted graph traversal is the signal the
backtest ultimately trades.

STATUS: SKELETON. The event-detection + evaluation scaffolding is real; the
propagation scoring is a documented first-cut you'll refine once the graph is
validated. Do NOT wire this to the backtest until the graph itself is trusted.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings  # noqa: E402


@dataclass
class ShockEvent:
    ticker: str
    date: pd.Timestamp
    magnitude: float   # signed daily return that triggered it


def detect_shocks(returns: pd.DataFrame, threshold: float = settings.Shock.THRESHOLD_RETURN) -> list[ShockEvent]:
    """Find all (ticker, date) where |daily return| >= threshold."""
    events: list[ShockEvent] = []
    for ticker in returns.columns:
        s = returns[ticker].dropna()
        hits = s[s.abs() >= threshold]
        events.extend(ShockEvent(ticker, idx, float(val)) for idx, val in hits.items())
    return events


def predict_aftershocks(g: nx.DiGraph, event: ShockEvent) -> list[tuple[str, float]]:
    """
    Predict aftershock targets for a shock event.

    First-cut logic (TODO(anuj): refine): for each out-neighbour B of the shocked
    node A, the predicted aftershock score is edge_weight(A->B) * sign(magnitude).
    Later: multi-hop propagation with decay, and calibrate scores to actual
    return magnitudes rather than just direction.
    """
    if event.ticker not in g:
        return []
    sign = 1.0 if event.magnitude > 0 else -1.0
    preds = []
    for _, b, d in g.out_edges(event.ticker, data=True):
        preds.append((b, float(d.get("weight", 0.0)) * sign))
    return sorted(preds, key=lambda x: abs(x[1]), reverse=True)


def evaluate_predictions(
    g: nx.DiGraph,
    returns: pd.DataFrame,
    horizon: int = settings.Shock.HORIZON_DAYS,
) -> pd.DataFrame:
    """
    Walk history: for every detected shock, did the predicted neighbours actually
    move in the predicted direction within `horizon` days? Returns a per-prediction
    hit/miss table — this is how you measure Section 1's "does it work" number
    (directional hit-rate) BEFORE risking the complexity of a full backtest.

    STATUS: SKELETON — fill in the forward-return lookup and hit logic.
    """
    raise NotImplementedError(
        "evaluate_predictions: implement forward-return hit-rate scoring. "
        "This is the first quantitative milestone once the graph is trusted."
    )
