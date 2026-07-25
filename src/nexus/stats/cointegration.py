"""
Cointegration — the classic pairs-trading foundation (Section 4a).

Two price series are cointegrated if some linear combination of them is
stationary (mean-reverting), even though each series alone wanders. That
mean-reverting spread is exactly what a pairs strategy trades.

Engle-Granger via statsmodels. STATUS: functional.
"""
from __future__ import annotations

import itertools

import pandas as pd
from statsmodels.tsa.stattools import coint


def cointegrated_pairs(
    prices: pd.DataFrame,
    pvalue_max: float,
    min_obs: int,
) -> list[tuple[str, str, float]]:
    """
    Test every pair for cointegration; return (a, b, pvalue) for pairs that
    reject the null of no-cointegration at `pvalue_max`.

    NOTE: testing all pairs means many hypothesis tests -> multiple-comparisons
    risk. TODO(v2): apply a Benjamini-Hochberg FDR correction, or gate the test
    behind an NLP-confirmed relationship so you're not fishing across all C(n,2).
    """
    prices = prices.dropna(how="any")
    out = []
    for a, b in itertools.combinations(prices.columns, 2):
        s_a, s_b = prices[a], prices[b]
        if s_a.notna().sum() < min_obs or s_b.notna().sum() < min_obs:
            continue
        try:
            _, pval, _ = coint(s_a, s_b)
        except Exception:
            continue
        if pval <= pvalue_max:
            out.append((a, b, float(pval)))
    return out
