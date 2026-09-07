#!/usr/bin/env python
"""
'EYES BEFORE BRAIN' graph: take the semantically-proposed candidate edges and
keep ONLY those that are also statistically real on price data. The AND filters
out boilerplate-driven noise from the Eyes.

    python scripts/09_build_semantic_eyes.py transformer   # produces candidate_edges.parquet
    python scripts/10_confirm_edges.py

Output: data/processed/nexus_semantic_graph.graphml
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd, networkx as nx
from config import settings, universe
from nexus.graph.edge_confirmation import confirm_edges

SPLIT = "2023-01-01"


def main():
    cand = pd.read_parquet(settings.PROCESSED_DIR / "candidate_edges.parquet")
    prices = pd.read_parquet(settings.RAW_DIR / "prices_eod.parquet")
    import numpy as np
    returns = np.log(prices / prices.shift(1)).dropna(how="all")
    # look-ahead-safe: confirm on pre-2023 data only
    prices, returns = prices.loc[:SPLIT], returns.loc[:SPLIT]

    print(f"Candidate edges from the Eyes: {len(cand)}")
    g, kept = confirm_edges(cand, prices, returns)
    print(f"Confirmed by the Brain (cointegration OR transfer entropy): {kept}\n")

    sec = universe.SECTORS
    same = sum(1 for u, v in g.edges() if sec.get(u) == sec.get(v))
    print(f"Confirmed edges: {g.number_of_edges()}  |  same-sector={same}, "
          f"cross-sector={g.number_of_edges()-same}")
    print("\nConfirmed edges (the ones that are BOTH semantic AND statistical):")
    for u, v, d in sorted(g.edges(data=True), key=lambda e: -e[2]["weight"])[:20]:
        print(f"  {u:<14} -> {v:<14} sim={d['similarity']:.2f}  "
              f"coint_p={d['coint_p']:.3f}  te_p={d['te_p']:.3f}  "
              f"[{sec.get(u,'?')}/{sec.get(v,'?')}]")

    out = settings.PROCESSED_DIR / "nexus_semantic_graph.graphml"
    h = g.copy()
    for _u, _v, d in h.edges(data=True):
        for k in list(d):
            if d[k] is None: d[k] = ""
    nx.write_graphml(h, out)
    print(f"\nSaved the Eyes-before-Brain graph -> {out}")
    print("\nNEXT: re-run propagation & volatility on THIS graph, and compare to the")
    print("pure-statistics graph -- that comparison is what proves the Eyes earned their keep.")


if __name__ == "__main__":
    main()