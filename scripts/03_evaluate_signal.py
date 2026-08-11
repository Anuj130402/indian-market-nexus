#!/usr/bin/env python
"""
THE GO/NO-GO GATE: does a shock in one stock predict its neighbours' moves,
better than a coin flip? Look-ahead-safe via a train/test split: the graph is
built ONLY from the training window; shocks are evaluated ONLY in the test window.

    python scripts/03_evaluate_signal.py

Run 01_fetch_market_data.py first.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus.data import market_data
from nexus.graph import builder
from nexus.propagation import shock_model

SPLIT_DATE = "2023-01-01"   # train < this date ; test >= this date
HORIZON = 5                  # trading days to look for the aftershock


def main() -> None:
    prices = market_data.load_prices()
    returns = market_data.log_returns(prices)

    train_prices = prices.loc[:SPLIT_DATE]
    train_returns = returns.loc[:SPLIT_DATE]
    print(f"Training graph on data before {SPLIT_DATE} "
          f"({train_prices.shape[0]} days, {train_prices.shape[1]} tickers)...")
    graph = builder.build_statistical_graph(train_prices, train_returns)
    print(f"Train graph: {graph.number_of_nodes()} nodes, {graph.number_of_edges()} edges\n")

    print(f"Evaluating shocks from {SPLIT_DATE} onward (out-of-sample)...")
    res = shock_model.evaluate_predictions(
        prices, returns, graph=graph, horizon=HORIZON, start=SPLIT_DATE)

    if res["n_predictions"] == 0:
        print("No predictions generated (no shocks hit a node with out-edges). "
              "Try lowering the shock threshold or check the train graph has edges.")
        return

    print("\n================ RESULT ================")
    print(f"  predictions : {res['n_predictions']}")
    print(f"  HIT RATE    : {res['hit_rate']:.3f}   (baseline 0.500, lift {res['lift']:+.3f})")
    print(f"  market drift : up_rate {res['up_rate']:.3f} (context)")
    print("========================================\n")

    d = res["detail"]
    print("Hit-rate by edge evidence type (WHERE the signal lives):")
    print(d.groupby("edge_kinds")["hit"].agg(["mean", "count"]).to_string())

    print("\nMost predictable targets (min 20 predictions):")
    by_t = d.groupby("target")["hit"].agg(["mean", "count"])
    by_t = by_t[by_t["count"] >= 20].sort_values("mean", ascending=False)
    print(by_t.head(10).to_string())

    print("\nInterpretation:")
    print("  ~0.50  -> no directional signal (coin flip)")
    print("  0.55+  -> a real edge worth pursuing (esp. if concentrated in an edge type)")
    print("  Look at the per-type/per-target breakdown, not just the headline number.")


if __name__ == "__main__":
    main()
