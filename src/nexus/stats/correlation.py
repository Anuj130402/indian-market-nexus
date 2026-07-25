"""
Correlation — the baseline edge measure (Section 4a, ranked lowest on purpose).

Correlation is symmetric and blind to lead-lag, so it can never give you a
*directed* edge. We keep it only as a cheap first-pass filter and as the
benchmark the fancier measures (Kalman, transfer entropy) must beat.
"""
from __future__ import annotations

import pandas as pd


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pairwise Pearson correlation of return series."""
    return returns.corr()


def correlated_pairs(returns: pd.DataFrame, threshold: float) -> list[tuple[str, str, float]]:
    """Return (a, b, rho) for every pair with |rho| >= threshold (a<b, undirected)."""
    corr = correlation_matrix(returns)
    out = []
    cols = corr.columns
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            rho = corr.loc[a, b]
            if abs(rho) >= threshold:
                out.append((a, b, float(rho)))
    return out
