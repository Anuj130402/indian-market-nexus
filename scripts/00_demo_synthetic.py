#!/usr/bin/env python
"""
DEMO: run the entire pipeline on synthetic data — no network, no data files.

    python scripts/00_demo_synthetic.py

This proves the architecture end-to-end on day one and verifies the stats
modules recover the structure we planted (A_LEADER -> B_FOLLOWER, C_COINT<->D_COINT,
E_INDEP isolated). Look at the printed centrality table and artifacts/demo_graph.png.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import settings
from nexus.data.synthetic import make_synthetic_prices
from nexus.data.market_data import log_returns
from nexus.graph import builder, metrics
from nexus.viz.network_plot import plot_graph


def main() -> None:
    prices = make_synthetic_prices()
    returns = log_returns(prices)
    print(f"Synthetic panel: {prices.shape[0]} days x {prices.shape[1]} names\n")

    g = builder.build_statistical_graph(prices, returns)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    print("Centrality (systemic importance):")
    print(metrics.centrality_table(g).to_string(index=False))

    out = plot_graph(g, path=settings.ARTIFACTS_DIR / "demo_graph.png",
                     title="Nexus — Synthetic Demo")
    print(f"\nGraph image -> {out}")
    print("Expected: A_LEADER points to B_FOLLOWER; C_COINT<->D_COINT linked; E_INDEP isolated.")


if __name__ == "__main__":
    main()
