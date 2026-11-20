"""Computes risk and return statistics of daily return series with empyrical.

The statistics of a performance tear sheet, for each strategy of a parameter
sweep in turn: the loop a backtest ends with.
"""
from __future__ import annotations

import numpy as np

TITLE = "Risk and return statistics of daily return series"
SOURCE = "empyrical-reloaded, called as a library"
LICENCE = None
REQUIRES = ("empyrical", "pandas")

# By size: strategies in the sweep, and trading days of each. The small sweep
# is sixty series of twenty years; the large one is six series ten times as
# long, so the same number of returns comes in arrays ten times larger.
SWEEPS = {"small": (60, 5_040), "large": (6, 50_400)}
# Mean and standard deviation of a day's return.
DAILY_MEAN, DAILY_DEVIATION = 0.0004, 0.01
# Window of the rolling statistics, in days: one year.
WINDOW = 252
# The functions of one argument applied to every series.
STATISTICS = (
    "annual_return", "cum_returns_final", "annual_volatility", "sharpe_ratio", "calmar_ratio",
    "stability_of_timeseries", "max_drawdown", "omega_ratio", "sortino_ratio", "tail_ratio", "value_at_risk",
    "downside_risk",
)
# The first trading day. Fifty thousand business days from it end in 2193,
# before the last date pandas can index.
FIRST_DAY = "2000-01-03"
# Seed of the returns.
SEED = 0
# One line per size saying how large its data is.
SIZES = {size: f"{strategies} series of {days:,} daily returns" for size, (strategies, days) in SWEEPS.items()}


def make_data(size: str) -> tuple:
    """Build the return series.

    Args:
        size: One of SIZES.

    Returns:
        (list of pandas Series of daily returns indexed by business day, the
        benchmark's Series).
    """
    import pandas as pd

    strategies, length = SWEEPS[size]
    rng = np.random.default_rng(SEED)
    days = pd.bdate_range(FIRST_DAY, periods=length)
    benchmark = pd.Series(rng.normal(DAILY_MEAN, DAILY_DEVIATION, length), index=days)
    series = [
        pd.Series(0.5 * benchmark.to_numpy() + rng.normal(DAILY_MEAN, DAILY_DEVIATION, length), index=days)
        for _ in range(strategies)
    ]
    return series, benchmark


def run(series: list, benchmark) -> tuple:
    """Compute the statistics of every series.

    Args:
        series: The strategies' daily returns.
        benchmark: The benchmark's daily returns.

    Returns:
        (table of the scalar statistics, one row per strategy; alpha and beta
        of each; the cumulative returns, rolling drawdown and rolling Sharpe
        ratio of each, stacked).
    """
    import empyrical

    table = np.array([[getattr(empyrical, name)(returns) for name in STATISTICS] for returns in series])
    alpha_beta = np.array([empyrical.alpha_beta(returns, benchmark) for returns in series])
    cumulative = np.array([empyrical.cum_returns(returns).to_numpy() for returns in series])
    drawdown = np.array([empyrical.roll_max_drawdown(returns, WINDOW).to_numpy() for returns in series])
    sharpe = np.array([empyrical.roll_sharpe_ratio(returns, WINDOW).to_numpy() for returns in series])
    return table, alpha_beta, cumulative, drawdown, sharpe
