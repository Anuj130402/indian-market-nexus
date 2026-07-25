"""
Market-data ingestion: End-of-Day OHLCV for the universe.

This is the *dynamic* data stream — the "electricity" that flows through the
graph. Everything statistical (cointegration, Kalman, transfer entropy) and the
shock/propagation logic is computed from the price series produced here.

v1 source: Yahoo Finance via `yfinance` (free EOD, no API key). Later you can
swap in a broker/paid feed behind the same `load_prices()` interface without
touching the rest of the codebase — that's the point of isolating I/O here.

STATUS: functional. Run `scripts/01_fetch_market_data.py` to populate the cache.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Make `config` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from config import settings, universe  # noqa: E402


_CACHE = settings.RAW_DIR / "prices_eod.parquet"


def fetch_prices(
    tickers: list[str] | None = None,
    start: str = settings.START_DATE,
    end: str = settings.END_DATE,
    field: str = "Close",
) -> pd.DataFrame:
    """
    Download adjusted EOD prices for `tickers` into a wide DataFrame
    (index = date, columns = ticker).

    Requires network + `yfinance`. On a machine without either, use
    `nexus.data.synthetic.make_synthetic_prices` to develop offline.
    """
    import yfinance as yf  # imported lazily so the package loads without it

    tickers = tickers or universe.tickers()
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
    )

    # yfinance returns a column MultiIndex when multiple tickers are requested.
    frames = {}
    for t in tickers:
        try:
            frames[t] = raw[t][field]
        except (KeyError, TypeError):
            # single-ticker shape, or ticker returned nothing
            if field in getattr(raw, "columns", []):
                frames[t] = raw[field]
    prices = pd.DataFrame(frames).sort_index()
    prices = prices.dropna(how="all")
    return prices


def save_prices(prices: pd.DataFrame, path: Path = _CACHE) -> Path:
    """Persist prices to Parquet (Section 5c: Parquet on disk for v1)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    prices.to_parquet(path)
    return path


def load_prices(path: Path = _CACHE) -> pd.DataFrame:
    """Load the cached price panel. Raises if it hasn't been fetched yet."""
    if not path.exists():
        raise FileNotFoundError(
            f"No cached prices at {path}. Run scripts/01_fetch_market_data.py first."
        )
    return pd.read_parquet(path)


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Daily log returns. Log returns are the sane default for stats work:
    they're additive over time and closer to normally distributed than
    simple returns.
    """
    import numpy as np

    return np.log(prices / prices.shift(1)).dropna(how="all")
