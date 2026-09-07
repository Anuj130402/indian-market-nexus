#!/usr/bin/env python
"""
THE PAYOFF: does knowing WHY a stock shocked (news) beat the price-only 0.53?

Clean controlled comparison — identical shock events and forward windows, the
only difference being 5 news features (source stock's news on the shock day):
    base   = 14 price/graph features
    +news  = 14 + 5 news features
trained on the SAME rows, evaluated out-of-sample on 2023+.

    python scripts/07_score_sentiment.py finbert   # (produces news_signal.parquet)
    python scripts/08_train_with_news.py
"""
import sys, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from nexus.data import market_data
from nexus.graph import builder
from nexus.ml import feature_builder as fb
from config import settings

SPLIT = "2023-01-01"


def main():
    prices = market_data.load_prices()
    returns = market_data.log_returns(prices)
    news = pd.read_parquet(settings.PROCESSED_DIR / "news_signal.parquet")
    print(f"Loaded news signal: {len(news):,} ticker-day rows")

    g = builder.build_statistical_graph(prices.loc[:SPLIT], returns.loc[:SPLIT])
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    Xtr, ytr, _ = fb.build_dataset(prices, returns, g, returns.index[70], SPLIT, news=news)
    Xte, yte, _ = fb.build_dataset(prices, returns, g, SPLIT, returns.index[-1], news=news)
    base = [c for c in Xtr.columns if c not in fb.NEWS_COLS]
    print(f"train={len(Xtr):,}  test={len(Xte):,}  |  {len(base)} base + {len(fb.NEWS_COLS)} news features")

    # how often does a shock even have same-day news? (coverage matters for interpretation)
    cov = (Xtr["news_count"] > 0).mean()
    print(f"shock-days with source news present: {cov:.1%}\n")

    def fit_auc(cols):
        gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                            max_iter=400, random_state=42).fit(Xtr[cols], ytr)
        return roc_auc_score(yte, gb.predict_proba(Xte[cols])[:, 1]), gb

    auc_base, _ = fit_auc(base)
    auc_news, gb_news = fit_auc(list(Xtr.columns))

    print("================= DIRECTION MODEL =================")
    print(f"  AUC  price+graph only   = {auc_base:.3f}   (the 0.53 baseline)")
    print(f"  AUC  + FinBERT news     = {auc_news:.3f}")
    print(f"  lift from news          = {auc_news - auc_base:+.3f}")
    print("==================================================\n")

    imp = permutation_importance(gb_news, Xte, yte, scoring="roc_auc", n_repeats=8, random_state=42)
    order = np.argsort(imp.importances_mean)[::-1]
    print("Top features (permutation importance):")
    for i in order[:10]:
        tag = "  <- NEWS" if Xtr.columns[i] in fb.NEWS_COLS else ""
        print(f"  {Xtr.columns[i]:<22}{imp.importances_mean[i]:+.4f}{tag}")

    print("\nInterpretation:")
    print("  lift ~0.00  -> public filings are already priced in; news doesn't rescue")
    print("               direction (a real, well-known-market-efficiency finding).")
    print("  lift 0.02+  -> knowing the CAUSE adds directional signal price couldn't see.")
    print("  Either way: check whether any news_* feature ranks above the graph features.")


if __name__ == "__main__":
    main()