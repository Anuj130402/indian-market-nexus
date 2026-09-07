#!/usr/bin/env python
"""
Fetch BSE corporate announcements for the universe and cache them.
This is the first "Eyes" data-sourcing step: it produces a daily-ish text feed
(headline + subject + category) per stock, which becomes the raw material for
the NLP sentiment/event signal.

    pip install bse pyarrow
    python scripts/06_fetch_announcements.py

NOTE: this is the ONE step that hits the network (BSE). Be patient and polite:
it paginates per stock with a delay so BSE doesn't rate-limit you. Expect it to
take a while for 8 years x ~48 stocks — run it once and cache.
"""
import sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from config import settings, universe
from nexus.data.bse_source import BSEAnnouncements

FROM = datetime(2018, 1, 1)
TO = datetime(2025, 12, 31)
OUT = settings.RAW_DIR / "announcements.parquet"


def main():
    src = BSEAnnouncements(download_folder=str(settings.RAW_DIR / "bse_pdfs"))
    all_rows, failed = [], []
    tickers = universe.tickers()
    for i, ticker in enumerate(tickers, 1):
        try:
            rows = list(src.fetch_ticker(ticker, FROM, TO))
            all_rows.extend(rows)
            print(f"[{i:>2}/{len(tickers)}] {ticker:<14} {len(rows):>5} announcements")
        except Exception as e:
            failed.append(ticker)
            print(f"[{i:>2}/{len(tickers)}] {ticker:<14} FAILED ({type(e).__name__})")
    src.close()

    if not all_rows:
        print("\nNo announcements fetched. Check network / that `bse` resolves scrip codes.")
        return
    df = pd.DataFrame(all_rows).drop_duplicates("news_id").sort_values("date")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT)
    print(f"\nSaved {len(df):,} announcements across {df['ticker'].nunique()} stocks -> {OUT}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print("\nAnnouncements per year:")
    print(df.assign(year=pd.to_datetime(df['date']).dt.year).groupby('year').size().to_string())
    if failed:
        print(f"\nScrip-code lookup failed for {len(failed)}: {failed}")
        print("Fix by adding a manual {ticker: scrip_code} override for these names.")


if __name__ == "__main__":
    main()