"""Tests for the graph builder: structure recovery + directedness."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus.data.synthetic import make_synthetic_prices
from nexus.data.market_data import log_returns
from nexus.graph import builder, metrics


def _build():
    prices = make_synthetic_prices()
    returns = log_returns(prices)
    return builder.build_statistical_graph(prices, returns)


def test_graph_is_directed_and_nonempty():
    g = _build()
    assert g.is_directed()
    assert g.number_of_nodes() == 5
    assert g.number_of_edges() > 0


def test_independent_node_has_few_links():
    g = _build()
    # E_INDEP is a lone random walk; it should be far less connected than the
    # planted, related names.
    assert g.degree("E_INDEP", weight=None) <= g.degree("C_COINT", weight=None)


def test_centrality_table_runs():
    g = _build()
    tbl = metrics.centrality_table(g)
    assert not tbl.empty
    assert {"pagerank", "out_strength", "in_strength"}.issubset(tbl.columns)
