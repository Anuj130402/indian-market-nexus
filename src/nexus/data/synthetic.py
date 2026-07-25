"""
Synthetic price generator — lets the whole pipeline run with ZERO external
data or network access. This is deliberately here so that on day one you can
clone the repo and immediately see graph + stats + viz working end to end,
before you've sorted out any data source.

We plant a few *known* cointegrated / lead-lag relationships so you can verify
the stats modules actually recover the structure we injected (a mini ground-truth
test for the whole system).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_synthetic_prices(
    n_days: int = 1500,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Returns a wide price DataFrame with these planted structures:
      - A_LEADER  -> B_FOLLOWER   (B lags A by 1 day: a directed link)
      - C_COINT   ~  D_COINT      (cointegrated: share a common stochastic trend)
      - E_INDEP                    (independent random walk: should be an isolated node)
    A good graph builder should recover A->B and C<->D and leave E unconnected.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-01", periods=n_days)

    # A leads; B follows A's returns with a 1-day lag plus noise.
    ra = rng.normal(0, 0.012, n_days)
    a = 100 * np.exp(np.cumsum(ra))
    rb = np.roll(ra, 1) * 0.8 + rng.normal(0, 0.006, n_days)
    b = 120 * np.exp(np.cumsum(rb))

    # C and D share a common trend => cointegrated (spread is mean-reverting).
    common = np.cumsum(rng.normal(0, 0.010, n_days))
    c = 50 * np.exp(common + rng.normal(0, 0.004, n_days))
    d = 75 * np.exp(1.0 * common + rng.normal(0, 0.004, n_days))

    # E is its own random walk, unrelated to everyone.
    e = 200 * np.exp(np.cumsum(rng.normal(0, 0.013, n_days)))

    return pd.DataFrame(
        {"A_LEADER": a, "B_FOLLOWER": b, "C_COINT": c, "D_COINT": d, "E_INDEP": e},
        index=dates,
    )
