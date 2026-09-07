"""
Aggregate scored announcements into a DAILY per-stock signal panel.

Input : announcements with columns [ticker, date, text, category, ...]
Output: one row per (ticker, date) with the day's news summarised into features
        that can be merged into the ML feature matrix:

  news_count      how many announcements that day (attention/activity)
  news_sentiment  mean polarity of the day's news
  news_neg_ratio  fraction of the day's news that was negative
  news_worst      the single most negative polarity that day (worst news)
  news_abs_max    the biggest-magnitude news that day (event intensity)

NOTE ON LOOK-AHEAD: this panel is keyed by the announcement's calendar date.
The look-ahead-safe alignment to trading days (and the after-market-hours shift)
is applied when this is MERGED into the feature matrix, where the trading
calendar is available. This module just summarises; it does not decide timing.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def aggregate_daily(df: pd.DataFrame, polarity: np.ndarray) -> pd.DataFrame:
    d = df.copy()
    d["polarity"] = polarity
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date"])                       # drop the null-date rows
    g = d.groupby(["ticker", d["date"].dt.normalize()])
    out = g.agg(
        news_count=("polarity", "size"),
        news_sentiment=("polarity", "mean"),
        news_neg_ratio=("polarity", lambda s: float((s < 0).mean())),
        news_worst=("polarity", "min"),
        news_abs_max=("polarity", lambda s: float(np.abs(s).max())),
    ).reset_index().rename(columns={"date": "date"})
    return out