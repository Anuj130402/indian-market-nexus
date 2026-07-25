"""
Network visualisation (dream feature seed: the interactive explorer starts here).

v1 is a static matplotlib render — enough to eyeball whether the graph is
sensible. The interactive/live versions (shock simulator, zoomable explorer)
are a later frontend concern, but a picture you can look at TODAY is the fastest
way to sanity-check the whole pipeline.

STATUS: functional.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt  # noqa: E402
import networkx as nx  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings  # noqa: E402

# Sector -> colour, so clusters are visible at a glance.
_PALETTE = plt.cm.tab20.colors


def plot_graph(g: nx.DiGraph, path: Path | None = None, title: str = "The Indian Market Nexus") -> Path:
    """Render the directed graph to a PNG and return the path."""
    path = path or (settings.ARTIFACTS_DIR / "nexus_graph.png")

    sectors = sorted({g.nodes[n].get("sector", "Unknown") for n in g.nodes})
    colour_of = {s: _PALETTE[i % len(_PALETTE)] for i, s in enumerate(sectors)}
    node_colours = [colour_of[g.nodes[n].get("sector", "Unknown")] for n in g.nodes]

    # Node size scales with systemic importance (weighted degree).
    deg = dict(g.degree(weight="weight"))
    sizes = [300 + 250 * deg.get(n, 0) for n in g.nodes]

    pos = nx.spring_layout(g, seed=settings.RANDOM_SEED, k=0.6)
    fig, ax = plt.subplots(figsize=(16, 12))
    nx.draw_networkx_edges(g, pos, ax=ax, alpha=0.25, arrows=True,
                           arrowsize=8, connectionstyle="arc3,rad=0.08")
    nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colours,
                           node_size=sizes, alpha=0.9)
    labels = {n: g.nodes[n].get("name", n).split()[0] for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels, ax=ax, font_size=7)

    handles = [plt.Line2D([0], [0], marker="o", linestyle="", markerfacecolor=c,
                          markersize=8, label=s) for s, c in colour_of.items()]
    ax.legend(handles=handles, loc="lower left", fontsize=7, title="Sector")
    ax.set_title(title, fontsize=15)
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path
