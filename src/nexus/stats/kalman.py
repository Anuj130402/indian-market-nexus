"""
Kalman-filter dynamic hedge ratio (Section 4a, your #1 pick).

In classic pairs trading you regress A on B once to get a fixed hedge ratio
beta. Reality: beta drifts. A Kalman filter treats beta (and an intercept) as a
hidden state that evolves over time, updating it every day as new prices arrive.

The output is a *time-varying* spread:  spread_t = A_t - beta_t * B_t - alpha_t
which is what you actually trade / feed to the rolling-edge logic.

This is a compact, dependency-light Kalman implementation (no external Kalman
package needed) specialised to the 2-D state [alpha, beta].

STATUS: functional (basic). TODO(v2): tune the process/observation noise via
MLE instead of the fixed deltas below; consider `pykalman` for the general case.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def dynamic_hedge_ratio(
    y: pd.Series,
    x: pd.Series,
    delta: float = 1e-4,
    obs_noise: float = 1e-3,
) -> pd.DataFrame:
    """
    Estimate a time-varying [alpha_t, beta_t] such that y_t ~= alpha_t + beta_t * x_t.

    Args:
        y, x: aligned price (or log-price) series.
        delta: controls how fast the hidden state is allowed to drift
               (larger => more responsive, noisier).
        obs_noise: measurement-noise variance.

    Returns a DataFrame indexed like the inputs with columns:
        alpha, beta, spread  (spread = y - (alpha + beta*x), i.e. the residual)
    """
    df = pd.concat([y.rename("y"), x.rename("x")], axis=1).dropna()
    n = len(df)

    trans_cov = delta / (1 - delta) * np.eye(2)   # state transition covariance
    state = np.zeros(2)                            # [alpha, beta]
    cov = np.ones((2, 2))

    alphas = np.empty(n)
    betas = np.empty(n)
    spreads = np.empty(n)

    yv = df["y"].to_numpy()
    xv = df["x"].to_numpy()

    for t in range(n):
        obs = np.array([1.0, xv[t]])          # design vector [1, x_t]
        # --- predict ---
        cov = cov + trans_cov
        # --- update ---
        pred = obs @ state
        resid = yv[t] - pred
        s = obs @ cov @ obs + obs_noise       # innovation variance
        k = (cov @ obs) / s                    # Kalman gain
        state = state + k * resid
        cov = cov - np.outer(k, obs) @ cov

        alphas[t] = state[0]
        betas[t] = state[1]
        spreads[t] = resid                     # residual == current spread

    return pd.DataFrame(
        {"alpha": alphas, "beta": betas, "spread": spreads}, index=df.index
    )
