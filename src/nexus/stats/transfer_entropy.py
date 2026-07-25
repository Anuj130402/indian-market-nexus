"""
Transfer Entropy (TE) — the "X-factor" directional measure (Section 4a, rank 2).

TE(A -> B) asks: does knowing A's past *reduce uncertainty about B's future*,
beyond what B's own past already tells us? Unlike correlation it is:
  - directional  (TE(A->B) != TE(B->A))  => gives you a DIRECTED edge
  - model-free   (captures non-linear dependence, not just linear)

This is a discrete (binned) estimator of TE for lag-1, which is enough for a
v1 directed graph. It is the honest, transparent version — good enough to build
on and easy to reason about.

TODO(v2): swap in a k-NN / KSG continuous estimator (e.g. via `pyinform` or a
custom KSG) for better small-sample behaviour, and add a permutation test to
get the null distribution referenced by EdgeThresholds.TRANSFER_ENTROPY_PERCENTILE.

STATUS: functional (basic, binned).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _discretise(x: np.ndarray, bins: int) -> np.ndarray:
    """Bin a continuous series into integer states via quantiles."""
    ranks = pd.Series(x).rank(method="first").to_numpy()
    edges = np.linspace(0, len(x), bins + 1)
    return np.digitize(ranks, edges[1:-1])


def transfer_entropy(source: np.ndarray, target: np.ndarray, bins: int = 4) -> float:
    """
    Estimate TE(source -> target) at lag 1 in nats.

    Uses the identity TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1}),
    computed from empirical joint counts. Returns >= 0; larger = more directed
    information flow from source to target.
    """
    x = _discretise(np.asarray(source), bins)
    y = _discretise(np.asarray(target), bins)

    y_t, y_p, x_p = y[1:], y[:-1], x[:-1]  # align on lag-1

    def _entropy(counts: np.ndarray) -> float:
        p = counts / counts.sum()
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    # H(Y_t, Y_{t-1})
    joint_yy = np.histogram2d(y_t, y_p, bins=bins)[0]
    # H(Y_{t-1})
    hist_yp = np.histogram(y_p, bins=bins)[0]
    # H(Y_t, Y_{t-1}, X_{t-1}) and H(Y_{t-1}, X_{t-1}) via flattened joint states
    yyx = (y_t * bins * bins) + (y_p * bins) + x_p
    yx = (y_p * bins) + x_p
    joint_yyx = np.bincount(yyx, minlength=bins ** 3).astype(float)
    joint_yx = np.bincount(yx, minlength=bins ** 2).astype(float)

    h_yt_given_yp = _entropy(joint_yy) - _entropy(hist_yp)
    h_yt_given_yp_xp = _entropy(joint_yyx) - _entropy(joint_yx)
    return max(0.0, h_yt_given_yp - h_yt_given_yp_xp)


def directed_te_pairs(returns: pd.DataFrame, bins: int = 4) -> list[tuple[str, str, float]]:
    """
    Compute TE in both directions for every ordered pair and return
    (source, target, te) for the *dominant* direction of each pair.
    Thresholding against a permutation null is left to the graph builder.
    """
    cols = list(returns.columns)
    r = returns.dropna(how="any")
    out = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            te_ab = transfer_entropy(r[a].to_numpy(), r[b].to_numpy(), bins)
            te_ba = transfer_entropy(r[b].to_numpy(), r[a].to_numpy(), bins)
            if te_ab >= te_ba:
                out.append((a, b, float(te_ab)))
            else:
                out.append((b, a, float(te_ba)))
    return out
