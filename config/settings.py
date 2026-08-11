"""
Central configuration for The Indian Market Nexus.

Everything tunable lives here so experiments are reproducible and you're never
hunting for a magic number buried in a function. This is the "make the
threshold robust and configurable" answer from the spec sheet: the defaults
below are sensible starting points, not gospel — sweep them.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"  # graphs, plots, model outputs (gitignored)

for _d in (RAW_DIR, PROCESSED_DIR, ARTIFACTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Data window (spec: 2018-2025, spans the 2020 COVID shock on purpose)
# ---------------------------------------------------------------------------
START_DATE = "2018-01-01"
END_DATE = "2025-12-31"

# ---------------------------------------------------------------------------
# Statistical edge thresholds  (Section 4c: "figure it out, make it robust")
# ---------------------------------------------------------------------------
class EdgeThresholds:
    # An edge is only kept if it clears these. The AND across NLP + stats is
    # what keeps the graph from becoming a correlation hairball.
    COINTEGRATION_PVALUE = 0.05        # Engle-Granger: reject no-cointegration
    CORRELATION_MIN = 0.5              # baseline filter for candidate pairs
    TRANSFER_ENTROPY_PERCENTILE = 90   # keep TE above 90th pct of shuffled null
    MIN_OBSERVATIONS = 250             # ~1 trading year before we trust a pair
    TRANSFER_ENTROPY_ALPHA = 0.05   # keep a directed TE edge only if p <= this
    TE_SURROGATES = 200
# ---------------------------------------------------------------------------
# Rolling-window settings (Section 4b: rolling preferred over static)
# ---------------------------------------------------------------------------
class Windows:
    ROLLING_LOOKBACK = 252   # trading days (~1y) for rolling edge stats
    ROLLING_STEP = 21        # recompute edges ~monthly

# ---------------------------------------------------------------------------
# Shock / propagation definition (Section 1: what counts as a "shock")
# ---------------------------------------------------------------------------
class Shock:
    THRESHOLD_RETURN = 0.02   # a daily move >= 2% (abs) is a "shock" event
    HORIZON_DAYS = 5          # look for aftershocks within 5 trading days

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
RANDOM_SEED = 42
