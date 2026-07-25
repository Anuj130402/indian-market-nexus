"""
Tests that verify the stats modules recover KNOWN planted structure.

These double as living documentation of what each measure is supposed to do —
if a refactor breaks the intuition, these fail loudly.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
from nexus.data.synthetic import make_synthetic_prices
from nexus.data.market_data import log_returns
from nexus.stats import cointegration, transfer_entropy, kalman
from config import settings


def test_cointegration_recovers_planted_pair():
    prices = make_synthetic_prices()
    pairs = cointegration.cointegrated_pairs(
        prices, settings.EdgeThresholds.COINTEGRATION_PVALUE, min_obs=250
    )
    found = {frozenset((a, b)) for a, b, _ in pairs}
    assert frozenset(("C_COINT", "D_COINT")) in found, "should find the planted cointegrated pair"


def test_transfer_entropy_is_directional():
    prices = make_synthetic_prices()
    r = log_returns(prices)
    te_ab = transfer_entropy.transfer_entropy(r["A_LEADER"].to_numpy(), r["B_FOLLOWER"].to_numpy())
    te_ba = transfer_entropy.transfer_entropy(r["B_FOLLOWER"].to_numpy(), r["A_LEADER"].to_numpy())
    assert te_ab > te_ba, "A leads B, so TE(A->B) should exceed TE(B->A)"


def test_kalman_beta_is_positive_for_cointegrated_pair():
    prices = make_synthetic_prices()
    res = kalman.dynamic_hedge_ratio(np.log(prices["D_COINT"]), np.log(prices["C_COINT"]))
    assert res["beta"].iloc[-1] > 0, "positively co-moving pair should have positive hedge ratio"
    assert res["spread"].abs().mean() < 1.0, "spread should be small/mean-reverting"
