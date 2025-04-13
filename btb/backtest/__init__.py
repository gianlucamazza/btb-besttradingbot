"""Backtesting framework for the BestTradingBot."""

from btb.backtest.engine import Backtester
from btb.backtest.metrics import calculate_metrics
from btb.backtest.walk_forward import WalkForwardAnalyzer

# from btb.backtest.monte_carlo import MonteCarloSimulator  # Not implemented yet

__all__ = [
    "Backtester",
    "calculate_metrics",
    "WalkForwardAnalyzer",
    # "MonteCarloSimulator",
]
