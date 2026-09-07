#!/usr/bin/env python
"""
LEARNED aftershock model: train a classifier on graph-derived features to
predict whether a shocked stock's neighbour moves up or down. Look-ahead-safe:
the graph and the model are both fit only on pre-2023 data; evaluation is on
2023+ shocks neither ever saw.

    python scripts/04_train_model.py

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
from sklearn.metrics import roc_auc_score, accuracy_score

from nexus.data import market_data
from nexus.graph import builder
from nexus.ml import feature_builder

SPLIT = "2023-01-01"


def main():
    prices = market_data.load_prices()
    returns = market_data.log_returns(prices)
    train_prices, train_returns = prices.loc[:SPLIT], returns.loc[:SPLIT]

    print(f"Building graph on pre-{SPLIT} data...")
    g = builder.build_statistical_graph(train_prices, train_returns)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    print("Assembling feature matrices (this walks every shock)...")
    Xtr, ytr, _ = feature_builder.build_dataset(
        prices, returns, g, start=train_returns.index[70], end=SPLIT)
    Xte, yte, _ = feature_builder.build_dataset(
        prices, returns, g, start=SPLIT, end=returns.index[-1])
    print(f"train rows={len(Xtr)}  test rows={len(Xte)}  features={Xtr.shape[1]}")
    print(f"test up-rate (base rate)={yte.mean():.3f}\n")

    sc = StandardScaler().fit(Xtr)
    lr = LogisticRegression(max_iter=1000).fit(sc.transform(Xtr), ytr)
    p_lr = lr.predict_proba(sc.transform(Xte))[:, 1]
    print(f"LogReg    OOS  AUC={roc_auc_score(yte, p_lr):.3f}  acc={accuracy_score(yte, p_lr>0.5):.3f}")

    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                        max_iter=400, random_state=42).fit(Xtr, ytr)
    p_gb = gb.predict_proba(Xte)[:, 1]
    print(f"HistGBM   OOS  AUC={roc_auc_score(yte, p_gb):.3f}  acc={accuracy_score(yte, p_gb>0.5):.3f}\n")

    imp = permutation_importance(gb, Xte, yte, scoring="roc_auc",
                                 n_repeats=8, random_state=42)
    order = np.argsort(imp.importances_mean)[::-1]
    print("Top features (permutation importance = AUC drop when shuffled):")
    for i in order[:8]:
        print(f"  {Xtr.columns[i]:<22} {imp.importances_mean[i]:+.4f}")

    print("\nInterpretation:")
    print("  AUC ~0.50 -> features carry no directional signal (honest null)")
    print("  AUC 0.53+ -> a learned edge the hand-designed rule missed")
    print("  Read the importances: if mkt_* dominate, it's just riding drift;")
    print("  if edge/graph features rank high, the NETWORK is doing real work.")


if __name__ == "__main__":
    main()
