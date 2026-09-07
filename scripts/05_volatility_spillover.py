#!/usr/bin/env python
"""
VOLATILITY-SPILLOVER model. Direction is near-efficient (our direction model was
a coin flip), so instead of predicting WHICH WAY a neighbour moves, we predict
WHETHER it has an OUTSIZED move at all after a shock in a connected stock.
Volatility clusters and spreads through a network in a way direction does not —
so this is where the graph structure has a real chance of carrying signal.

Label = 1 if |B's forward return| > SPIKE_K * (B's trailing vol * sqrt(horizon)),
i.e. B moved meaningfully more than normal. Look-ahead-safe: graph + model fit on
pre-2023 data; evaluated on 2023+.

    python scripts/05_volatility_spillover.py

Run 01_fetch_market_data.py first.
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from nexus.data import market_data
from nexus.graph import builder
from nexus.ml import feature_builder

SPLIT = "2023-01-01"
HORIZON = 5
SPIKE_K = 1.5          # "outsized" = move > 1.5x the normal horizon move


def spike_label(X, meta):
    expected = X["target_vol"].to_numpy() * np.sqrt(HORIZON)   # normal horizon move
    keep = expected > 0                                        # need a valid vol baseline
    y = (meta["fwd_return"].abs().to_numpy() > SPIKE_K * expected).astype(int)
    return X[keep].reset_index(drop=True), y[keep]


def main():
    prices = market_data.load_prices()
    returns = market_data.log_returns(prices)
    g = builder.build_statistical_graph(prices.loc[:SPLIT], returns.loc[:SPLIT])
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    Xtr, _, mtr = feature_builder.build_dataset(prices, returns, g,
                    start=returns.index[70], end=SPLIT, horizon=HORIZON)
    Xte, _, mte = feature_builder.build_dataset(prices, returns, g,
                    start=SPLIT, end=returns.index[-1], horizon=HORIZON)
    Xtr, ytr = spike_label(Xtr, mtr)
    Xte, yte = spike_label(Xte, mte)
    print(f"train rows={len(Xtr)}  test rows={len(Xte)}")
    print(f"spike base-rate (test) = {yte.mean():.3f}  (spikes are rarer -> AUC is the honest metric)\n")

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(Xtr), ytr)
    print(f"LogReg    vol-spike  OOS AUC = {roc_auc_score(yte, lr.predict_proba(sc.transform(Xte))[:,1]):.3f}")

    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                        max_iter=400, random_state=42).fit(Xtr, ytr)
    auc = roc_auc_score(yte, gb.predict_proba(Xte)[:, 1])
    print(f"HistGBM   vol-spike  OOS AUC = {auc:.3f}\n")

    imp = permutation_importance(gb, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=42)
    print("Top features (permutation importance):")
    for i in np.argsort(imp.importances_mean)[::-1][:8]:
        print(f"  {Xtr.columns[i]:<22}{imp.importances_mean[i]:+.4f}")

    print("\nInterpretation:")
    print("  AUC ~0.50            -> even volatility spillover isn't graph-predictable here")
    print("  AUC 0.55+ with edge/ -> the NETWORK predicts where volatility spreads:")
    print("  centrality features     the graph's core thesis, vindicated on the right target")
    print("  (contrast with the DIRECTION model, where graph features were ~0)")


if __name__ == "__main__":
    main()
