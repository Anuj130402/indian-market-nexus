from __future__ import annotations
import numpy as np
import pandas as pd

def _discretise(x, bins):
    ranks = pd.Series(x).rank(method="first").to_numpy()
    edges = np.linspace(0, len(x), bins + 1)
    return np.digitize(ranks, edges[1:-1])

def _entropy(counts):
    total = counts.sum()
    if total <= 0:                    # FIX(bug 3): guard divide-by-zero on empty input
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def transfer_entropy(source, target, bins=4):
    source = np.asarray(source); target = np.asarray(target)
    if len(source) < 3 or len(target) < 3:   # FIX: degenerate input -> no information
        return 0.0
    x = _discretise(source, bins); y = _discretise(target, bins)
    y_t, y_p, x_p = y[1:], y[:-1], x[:-1]
    joint_yy = np.histogram2d(y_t, y_p, bins=bins)[0]
    hist_yp = np.histogram(y_p, bins=bins)[0]
    yyx = (y_t*bins*bins)+(y_p*bins)+x_p
    yx = (y_p*bins)+x_p
    joint_yyx = np.bincount(yyx, minlength=bins**3).astype(float)
    joint_yx = np.bincount(yx, minlength=bins**2).astype(float)
    h1 = _entropy(joint_yy) - _entropy(hist_yp)
    h2 = _entropy(joint_yyx) - _entropy(joint_yx)
    return max(0.0, h1 - h2)

def te_significance(source, target, bins=4, n_surrogates=200, seed=None):
    rng = np.random.default_rng(seed)
    src = np.asarray(source); tgt = np.asarray(target)
    n = len(src)
    if n < 20:                        # FIX(bug 1): too little data -> not significant, don't crash
        return 0.0, 1.0
    observed = transfer_entropy(src, tgt, bins)
    null = np.empty(n_surrogates)
    for i in range(n_surrogates):
        shift = int(rng.integers(1, n))      # safe now: n >= 20
        null[i] = transfer_entropy(np.roll(src, shift), tgt, bins)
    p_value = (np.sum(null >= observed) + 1) / (n_surrogates + 1)
    return float(observed), float(p_value)

def significant_te_pairs(returns, alpha=0.05, bins=4, n_surrogates=200, seed=42):
    # FIX(bug 1): drop dead (all-NaN) columns BEFORE the common-window dropna,
    # otherwise one empty ticker deletes the entire dataset.
    r = returns.dropna(axis=1, how="all").dropna(how="any")
    cols = list(r.columns); out = []
    for i, a in enumerate(cols):
        for b in cols[i+1:]:
            te_ab, p_ab = te_significance(r[a].to_numpy(), r[b].to_numpy(), bins, n_surrogates, seed)
            te_ba, p_ba = te_significance(r[b].to_numpy(), r[a].to_numpy(), bins, n_surrogates, seed)
            if te_ab >= te_ba:
                if p_ab <= alpha: out.append((a, b, te_ab, p_ab))
            else:
                if p_ba <= alpha: out.append((b, a, te_ba, p_ba))
    return out
