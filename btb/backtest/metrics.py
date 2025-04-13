"""Performance metrics calculation for backtesting."""

from typing import Dict, List

import numpy as np


def calculate_metrics(equity_curve: List[float], trades: List[Dict], config: Dict = None) -> Dict:
    # If config is None, use an empty dict
    if config is None:
        config = {}
    """Calculate performance metrics from backtest results.

    Args:
        equity_curve: List of portfolio values over time
        trades: List of trade dictionaries
        config: Backtest configuration

    Returns:
        Dictionary with performance metrics
    """
    metrics = {}

    # Basic metrics
    initial_capital = equity_curve[0]
    final_capital = equity_curve[-1]
    total_return = (final_capital - initial_capital) / initial_capital
    metrics["initial_capital"] = initial_capital
    metrics["final_capital"] = final_capital
    metrics["total_return"] = total_return
    metrics["total_return_pct"] = total_return * 100

    # Convert equity curve to returns
    returns = []
    for i in range(1, len(equity_curve)):
        daily_return = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
        returns.append(daily_return)

    # Return metrics
    metrics["total_trades"] = len(trades)
    if len(returns) > 0:
        metrics["mean_return"] = np.mean(returns)
        metrics["std_return"] = np.std(returns)
    else:
        metrics["mean_return"] = 0
        metrics["std_return"] = 0

    # Annualized metrics (assume daily data)
    trading_days = 252  # Standard trading days in a year
    if len(returns) > 0:
        metrics["annualized_return"] = (1 + metrics["mean_return"]) ** trading_days - 1
        metrics["annualized_volatility"] = metrics["std_return"] * np.sqrt(trading_days)
    else:
        metrics["annualized_return"] = 0
        metrics["annualized_volatility"] = 0

    # Sharpe ratio (if requested)
    if config.get("metrics", {}).get("calculate_sharpe", True):
        risk_free_rate = 0.02  # Default annual risk-free rate (2%)
        (1 + risk_free_rate) ** (1 / trading_days) - 1

        if metrics["annualized_volatility"] > 0:
            metrics["sharpe_ratio"] = (metrics["annualized_return"] - risk_free_rate) / metrics["annualized_volatility"]
        else:
            metrics["sharpe_ratio"] = 0

    # Sortino ratio (if requested)
    if config.get("metrics", {}).get("calculate_sortino", True):
        negative_returns = [r for r in returns if r < 0]
        if len(negative_returns) > 0:
            downside_deviation = np.std(negative_returns) * np.sqrt(trading_days)
            if downside_deviation > 0:
                metrics["sortino_ratio"] = (metrics["annualized_return"] - risk_free_rate) / downside_deviation
            else:
                metrics["sortino_ratio"] = 0
        else:
            metrics["sortino_ratio"] = float("inf") if metrics["annualized_return"] > 0 else 0

    # Drawdown metrics (if requested)
    if config.get("metrics", {}).get("calculate_drawdown", True):
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        max_drawdown = abs(min(drawdown)) if len(drawdown) > 0 else 0
        metrics["max_drawdown"] = max_drawdown
        metrics["max_drawdown_pct"] = max_drawdown * 100

    # Win rate (if requested)
    if config.get("metrics", {}).get("calculate_win_rate", True):
        winning_trades = [t for t in trades if t["profit"] > 0]
        metrics["winning_trades"] = len(winning_trades)
        metrics["losing_trades"] = len(trades) - len(winning_trades)

        if len(trades) > 0:
            metrics["win_rate"] = len(winning_trades) / len(trades)
        else:
            metrics["win_rate"] = 0

        # Average win/loss
        if len(winning_trades) > 0:
            metrics["avg_profit"] = np.mean([t["profit"] for t in winning_trades])
            metrics["avg_profit_pct"] = np.mean([t["profit_pct"] for t in winning_trades])
        else:
            metrics["avg_profit"] = 0
            metrics["avg_profit_pct"] = 0

        losing_trades = [t for t in trades if t["profit"] <= 0]
        if len(losing_trades) > 0:
            metrics["avg_loss"] = np.mean([t["profit"] for t in losing_trades])
            metrics["avg_loss_pct"] = np.mean([t["profit_pct"] for t in losing_trades])
        else:
            metrics["avg_loss"] = 0
            metrics["avg_loss_pct"] = 0

        # Profit factor
        total_profit = sum([t["profit"] for t in winning_trades])
        total_loss = abs(sum([t["profit"] for t in losing_trades]))

        if total_loss > 0:
            metrics["profit_factor"] = total_profit / total_loss
        else:
            metrics["profit_factor"] = float("inf") if total_profit > 0 else 0

    # Consecutive trades
    if len(trades) > 0:
        trade_results = [1 if t["profit"] > 0 else 0 for t in trades]
        max_consecutive_wins = max_consecutive(trade_results, 1)
        max_consecutive_losses = max_consecutive(trade_results, 0)

        metrics["max_consecutive_wins"] = max_consecutive_wins
        metrics["max_consecutive_losses"] = max_consecutive_losses

    return metrics


def max_consecutive(arr, val):
    """Calculate maximum consecutive occurrences of val in arr."""
    max_count = count = 0
    for item in arr:
        if item == val:
            count += 1
            max_count = max(max_count, count)
        else:
            count = 0
    return max_count
