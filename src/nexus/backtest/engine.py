"""
Backtest engine — Section 1C, your #1-ranked *end goal* ("Good Prediction =
Good Trades", measured by number of successful trades + Sharpe vs Nifty).

IMPORTANT SEQUENCING: this is intentionally the LAST thing we build. A backtest
that trades a signal you haven't validated will produce a beautiful, lying
equity curve. The order is: trust the graph -> measure the signal's hit-rate
(propagation.evaluate_predictions) -> only THEN turn signals into trades here.

STATUS: SKELETON (interface + metrics stubs). Deliberately not implemented yet.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class BacktestConfig:
    capital: float = 100_000.0
    cost_bps: float = 5.0          # round-trip transaction cost assumption
    max_positions: int = 10
    benchmark: str = "^NSEI"       # Nifty 50 index, for relative performance


def run_backtest(signals: pd.DataFrame, prices: pd.DataFrame, config: BacktestConfig) -> dict:
    """
    Turn a stream of directional signals into positions, simulate P&L with costs,
    and report metrics. STATUS: SKELETON.

    Guardrails to build in from day one (cheap now, painful to retrofit):
      - point-in-time data only (no using tomorrow's info today = look-ahead bias)
      - realistic transaction costs + slippage (config.cost_bps)
      - out-of-sample split: fit thresholds on 2018-2022, TEST on 2023-2025
    """
    raise NotImplementedError("run_backtest — v1 milestone 4, after the signal is validated.")


def performance_metrics(equity_curve: pd.Series, benchmark: pd.Series) -> dict:
    """Sharpe, max drawdown, hit rate, CAGR, and excess return vs benchmark."""
    raise NotImplementedError("performance_metrics — implement alongside run_backtest.")
