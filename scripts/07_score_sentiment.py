#!/usr/bin/env python
"""
Turn the raw announcement corpus into a DAILY per-stock news signal.

    # 1) validate the whole pipeline on your real 69k rows in SECONDS (no download):
    python scripts/07_score_sentiment.py lexicon

    # 2) then the real signal with FinBERT (one-time ~440MB download; slow on CPU,
    #    fast on a GPU -> this is the Colab job):
    pip install transformers torch
    python scripts/07_score_sentiment.py finbert

Output: data/processed/news_signal.parquet  (one row per ticker-date)
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from config import settings
from nexus.nlp.news_signal import aggregate_daily

SCORER = sys.argv[1] if len(sys.argv) > 1 else "lexicon"
IN = settings.RAW_DIR / "announcements.parquet"
OUT = settings.PROCESSED_DIR / "news_signal.parquet"


def get_scorer(kind):
    if kind == "finbert":
        from nexus.nlp.sentiment import FinBERTScorer
        print("Loading FinBERT (first run downloads ~440MB)...")
        return FinBERTScorer()
    from nexus.nlp.sentiment import LexiconScorer
    return LexiconScorer()


def main():
    df = pd.read_parquet(IN)
    df["text"] = df["text"].fillna("").astype(str)
    print(f"Loaded {len(df):,} announcements. Scoring with: {SCORER}")

    scorer = get_scorer(SCORER)
    texts = df["text"].tolist()

    # score in chunks so long runs show progress instead of looking frozen
    import numpy as np
    chunk, pol = 2000, []
    for i in range(0, len(texts), chunk):
        pol.extend(scorer.score(texts[i:i + chunk]))
        print(f"  scored {min(i + chunk, len(texts)):>6,}/{len(texts):,}", end="\r")
    pol = np.asarray(pol)
    print()

    daily = aggregate_daily(df, pol)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    daily.to_parquet(OUT)
    print(f"\nSaved daily news signal: {len(daily):,} ticker-day rows -> {OUT}")
    print(f"Date range: {daily['date'].min().date()} to {daily['date'].max().date()}")
    print("\nMost negative news-days (sanity check — should look like real bad news):")
    print(daily.nsmallest(5, "news_sentiment")[
        ["ticker", "date", "news_count", "news_sentiment", "news_worst"]].to_string(index=False))
    print("\nSentiment distribution:")
    print(daily["news_sentiment"].describe()[["mean", "std", "min", "max"]].to_string())


if __name__ == "__main__":
    main()