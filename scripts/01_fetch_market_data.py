#!/usr/bin/env python
"""
Fetch EOD OHLCV for the Nifty 50 universe and cache to Parquet.

    python scripts/01_fetch_market_data.py

Requires internet + yfinance. This is the ONLY step that needs the network;
everything downstream runs off the local cache.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nexus.data import market_data


def main() -> None:
    print("Downloading EOD prices (this hits Yahoo Finance)...")
    prices = market_data.fetch_prices()
    path = market_data.save_prices(prices)
    print(f"Saved {prices.shape[0]} rows x {prices.shape[1]} tickers -> {path}")
    missing = prices.columns[prices.isna().all()].tolist()
    if missing:
        print(f"WARNING: no data for {missing} (delisted / renamed / bad ticker?)")


if __name__ == "__main__":
    main()
