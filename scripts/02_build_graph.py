#!/usr/bin/env python
"""
Build the Nexus graph from cached real prices, print insights, save artifacts.

    python scripts/02_build_graph.py

Run 01_fetch_market_data.py first.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus.data import market_data
from nexus.graph import builder, metrics
from nexus.viz.network_plot import plot_graph


def main() -> None:
    prices = market_data.load_prices()
    returns = market_data.log_returns(prices)

    g = builder.build_statistical_graph(prices, returns)
    print(f"Graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges\n")

    print("Top 10 systemically important names:")
    print(metrics.centrality_table(g).head(10).to_string(index=False))

    print("\nHidden linkages (non-obvious, statistically supported):")
    for u, v, d in metrics.hidden_linkages(g)[:10]:
        kinds = ",".join(sorted({e["kind"] for e in d.get("evidence", [])}))
        print(f"  {u:>14} -> {v:<14}  weight={d.get('weight', 0):.3f}  [{kinds}]")

    gpath = builder.save_graph(g)
    ppath = plot_graph(g)
    print(f"\nSaved graph -> {gpath}\nSaved image -> {ppath}")


if __name__ == "__main__":
    main()
