"""Backtesting engine for testing trading strategies."""

import json
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from btb.backtest.metrics import calculate_metrics
from btb.data.loader import DataLoader
from btb.strategies.base import BaseStrategy
from btb.strategies.factory import create_strategy


class Backtester:
    """Backtesting engine for trading strategies."""

    def __init__(self, config: Dict):
        """Initialize backtester with configuration.

        Args:
            config: Configuration dictionary
        """
        import logging

        self.logger = logging.getLogger("btb.backtest.engine")

        self.config = config
        self.logger.debug(f"Initializing strategy with config: {config}")
        self.strategy = self._init_strategy()
        self.logger.debug("Strategy initialized")
        self.data = self._load_data()
        self.logger.debug("Data loaded")
        self.results: Optional[Dict] = None

    def _init_strategy(self) -> BaseStrategy:
        """Initialize trading strategy."""
        try:
            strategy_name = self.config["backtest"]["strategy"]
            self.logger.debug(f"Strategy name: {strategy_name}")
            strategy_params = self.config.get("strategy_params", {})
            self.logger.debug(f"Strategy params: {strategy_params}")
            return create_strategy(strategy_name, strategy_params)
        except Exception as e:
            self.logger.error(f"Error initializing strategy: {e}")
            # Return a default TransformerStrategy if there's an error
            from btb.strategies.transformer_strategy import TransformerStrategy

            return TransformerStrategy({})

    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting."""
        try:
            # Get configuration
            backtest_config = self.config.get("backtest", {})
            if not backtest_config:
                self.logger.warning("No backtest configuration found, using default values")
                start_date = "2022-01-01"
                end_date = "2023-01-01"
                symbols = ["BTCUSDT"]
                timeframes = ["1h"]
            else:
                start_date = backtest_config.get("start_date", "2022-01-01")
                end_date = backtest_config.get("end_date", "2023-01-01")
                symbols = backtest_config.get("symbols", ["BTCUSDT"])
                timeframes = backtest_config.get("timeframes", ["1h"])

            self.logger.debug(f"Loading data for {symbols} {timeframes} from {start_date} to {end_date}")

            # Load data
            data_config = {"use_dummy": True}  # Always use dummy data for safety
            data_loader = DataLoader(data_config)
            data = data_loader.load_data(
                symbols=symbols, timeframes=timeframes, start_date=start_date, end_date=end_date
            )

            # Apply data processing if configured
            if "data_processing" in self.config:
                self.logger.debug("Applying data processing")
                from btb.data.preprocessing import DataPreprocessor

                preprocessor = DataPreprocessor()
                data = preprocessor.process(
                    data=data,
                    add_technical_indicators=self.config["data_processing"].get("feature_engineering", True),
                    normalize=self.config["data_processing"].get("normalization", None),
                    fill_missing=self.config["data_processing"].get("fill_missing", "ffill"),
                )

            return data
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            # Return dummy data
            return self._create_dummy_data()

    def _create_dummy_data(self) -> Dict[str, pd.DataFrame]:
        """Create dummy data for testing."""
        from datetime import timedelta

        import numpy as np
        import pandas as pd

        # Create date range
        start_date = datetime.strptime("2022-01-01", "%Y-%m-%d")
        end_date = datetime.strptime("2023-01-01", "%Y-%m-%d")
        days = (end_date - start_date).days
        dates = [start_date + timedelta(days=i) for i in range(days)]

        # Generate random price data
        n = len(dates)
        base_price = 10000

        # Add some randomness and trend
        price = float(base_price)
        prices = []
        for i in range(n):
            # Add some random walk component
            price = price * (1 + np.random.normal(0, 0.02))
            # Add cyclical component
            price = price * (1 + 0.01 * np.sin(i / 30))
            prices.append(price)

        # Create OHLC data
        opens = prices
        closes = [p * (1 + np.random.normal(0, 0.01)) for p in prices]
        highs = [max(o, c) * (1 + abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        lows = [min(o, c) * (1 - abs(np.random.normal(0, 0.01))) for o, c in zip(opens, closes)]
        volumes = [abs(np.random.normal(1000000, 500000)) for _ in range(n)]

        # Create DataFrame
        df = pd.DataFrame({"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates)

        return {"BTCUSDT_1h": df}

    def run(self) -> Dict:
        """Run the backtest.

        Returns:
            Dictionary with backtest results
        """
        results = {}
        portfolio_history: List[float] = []
        trades = []

        # Get backtest configuration
        initial_capital = self.config["backtest"]["initial_capital"]
        commission = self.config["backtest"].get("commission", 0.0007)  # Default to 0.07%
        slippage = self.config["backtest"].get("slippage", 0.0001)  # Default to 0.01%

        # Iterate through each symbol and timeframe
        for key, df in self.data.items():
            symbol, timeframe = key.split("_")

            # Generate signals
            df_with_signals = self.strategy.generate_signals(df)

            # Simulate trading
            capital = initial_capital
            position = None
            symbol_trades = []
            equity_curve = [capital]

            for i in range(1, len(df_with_signals)):
                current_row = df_with_signals.iloc[i]
                df_with_signals.iloc[i - 1]

                current_time = df_with_signals.index[i]
                current_price = current_row["close"]
                signal = current_row["signal"]

                # Check for stop loss / take profit
                if position is not None and position.get("is_open", False):
                    # Calculate profit/loss
                    entry_price = position["entry_price"]
                    position_size = position["size"]
                    position_type = position["type"]

                    # Check stop loss
                    stop_loss = position.get("stop_loss")
                    if stop_loss is not None:
                        if (position_type == "long" and current_price <= stop_loss) or (
                            position_type == "short" and current_price >= stop_loss
                        ):
                            # Stop loss triggered
                            exit_price = stop_loss
                            if position_type == "long":
                                profit = (exit_price - entry_price) * position_size
                            else:  # short
                                profit = (entry_price - exit_price) * position_size

                            # Apply commission
                            profit -= exit_price * position_size * commission

                            # Update capital
                            capital += profit + (position_size * entry_price)

                            # Record trade
                            trade = {
                                "symbol": symbol,
                                "entry_time": position["entry_time"],
                                "exit_time": current_time,
                                "entry_price": entry_price,
                                "exit_price": exit_price,
                                "size": position_size,
                                "type": position_type,
                                "profit": profit,
                                "profit_pct": (profit / (position_size * entry_price)) * 100
                                if position_size * entry_price != 0
                                else 0,
                                "exit_reason": "stop_loss",
                            }
                            symbol_trades.append(trade)

                            # Close position
                            position = None
                            continue  # Continue to next iteration since position is closed

                    # Only check take profit if position is still open
                    if position is not None:  # Check position again as it might have been closed by stop loss
                        take_profit = position.get("take_profit")
                        if take_profit is not None:
                            if (position_type == "long" and current_price >= take_profit) or (
                                position_type == "short" and current_price <= take_profit
                            ):
                                # Take profit triggered
                                exit_price = take_profit
                                if position_type == "long":
                                    profit = (exit_price - entry_price) * position_size
                                else:  # short
                                    profit = (entry_price - exit_price) * position_size

                                # Apply commission
                                profit -= exit_price * position_size * commission

                                # Update capital
                                capital += profit + (position_size * entry_price)

                                # Record trade
                                trade = {
                                    "symbol": symbol,
                                    "entry_time": position["entry_time"],
                                    "exit_time": current_time,
                                    "entry_price": entry_price,
                                    "exit_price": exit_price,
                                    "size": position_size,
                                    "type": position_type,
                                    "profit": profit,
                                    "profit_pct": (profit / (position_size * entry_price)) * 100
                                    if position_size * entry_price != 0
                                    else 0,
                                    "exit_reason": "take_profit",
                                }
                                symbol_trades.append(trade)

                                # Close position
                                position = None
                                continue  # Continue to next iteration since position is closed

                    # Check for trailing stop update
                    if position is not None:  # Check position again as it might have been closed
                        should_update, new_stop, new_take = self.strategy.should_update_stops(position, current_price)
                        if should_update:
                            position["stop_loss"] = new_stop
                            position["take_profit"] = new_take

                # Check for signal to open/close position
                if signal != 0:
                    if position is None:  # No open position
                        # Open new position
                        position_type = "long" if signal > 0 else "short"

                        # Calculate position size
                        price_with_slippage = (
                            current_price * (1 + slippage)
                            if position_type == "long"
                            else current_price * (1 - slippage)
                        )
                        position_size = self.strategy.calculate_position_size(capital, price_with_slippage)

                        # Calculate commission
                        commission_amount = price_with_slippage * position_size * commission

                        # Update available capital
                        available_capital = capital - (position_size * price_with_slippage) - commission_amount

                        # Set stop loss and take profit
                        stop_loss = None
                        take_profit = None

                        if position_type == "long":
                            if "stop_loss" in self.strategy.params:
                                stop_loss = price_with_slippage * (1 - self.strategy.params["stop_loss"])
                            if "take_profit" in self.strategy.params:
                                take_profit = price_with_slippage * (1 + self.strategy.params["take_profit"])
                        else:  # short
                            if "stop_loss" in self.strategy.params:
                                stop_loss = price_with_slippage * (1 + self.strategy.params["stop_loss"])
                            if "take_profit" in self.strategy.params:
                                take_profit = price_with_slippage * (1 - self.strategy.params["take_profit"])

                        # Create position
                        position = {
                            "is_open": True,
                            "type": position_type,
                            "size": position_size,
                            "entry_price": price_with_slippage,
                            "entry_time": current_time,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                        }

                        # Update capital
                        capital = available_capital

                    elif signal * (1 if position["type"] == "long" else -1) < 0:  # Signal opposite to current position
                        # Close position
                        exit_price = (
                            current_price * (1 - slippage)
                            if position["type"] == "long"
                            else current_price * (1 + slippage)
                        )
                        entry_price = position["entry_price"]
                        position_size = position["size"]
                        position_type = position["type"]

                        if position_type == "long":
                            profit = (exit_price - entry_price) * position_size
                        else:  # short
                            profit = (entry_price - exit_price) * position_size

                        # Apply commission
                        profit -= exit_price * position_size * commission

                        # Update capital
                        capital += profit + (position_size * entry_price)

                        # Record trade
                        trade = {
                            "symbol": symbol,
                            "entry_time": position["entry_time"],
                            "exit_time": current_time,
                            "entry_price": entry_price,
                            "exit_price": exit_price,
                            "size": position_size,
                            "type": position_type,
                            "profit": profit,
                            "profit_pct": (profit / (position_size * entry_price)) * 100
                            if position_size * entry_price != 0
                            else 0,
                            "exit_reason": "signal",
                        }
                        symbol_trades.append(trade)

                        # Open new position in opposite direction
                        position_type = "long" if signal > 0 else "short"

                        # Calculate position size
                        price_with_slippage = (
                            current_price * (1 + slippage)
                            if position_type == "long"
                            else current_price * (1 - slippage)
                        )
                        position_size = self.strategy.calculate_position_size(capital, price_with_slippage)

                        # Calculate commission
                        commission_amount = price_with_slippage * position_size * commission

                        # Update available capital
                        available_capital = capital - (position_size * price_with_slippage) - commission_amount

                        # Set stop loss and take profit
                        stop_loss = None
                        take_profit = None

                        if position_type == "long":
                            if "stop_loss" in self.strategy.params:
                                stop_loss = price_with_slippage * (1 - self.strategy.params["stop_loss"])
                            if "take_profit" in self.strategy.params:
                                take_profit = price_with_slippage * (1 + self.strategy.params["take_profit"])
                        else:  # short
                            if "stop_loss" in self.strategy.params:
                                stop_loss = price_with_slippage * (1 + self.strategy.params["stop_loss"])
                            if "take_profit" in self.strategy.params:
                                take_profit = price_with_slippage * (1 - self.strategy.params["take_profit"])

                        # Create position
                        position = {
                            "is_open": True,
                            "type": position_type,
                            "size": position_size,
                            "entry_price": price_with_slippage,
                            "entry_time": current_time,
                            "stop_loss": stop_loss,
                            "take_profit": take_profit,
                        }

                        # Update capital
                        capital = available_capital

                # Calculate portfolio value (capital + open position value)
                portfolio_value = capital
                if position is not None and position["is_open"]:
                    position_value = position["size"] * current_price
                    if position["type"] == "long":
                        unrealized_profit = (current_price - position["entry_price"]) * position["size"]
                    else:  # short
                        unrealized_profit = (position["entry_price"] - current_price) * position["size"]
                    portfolio_value = capital + position_value + unrealized_profit

                equity_curve.append(portfolio_value)

            # Close any open position at the end of the backtest
            if position is not None and position["is_open"]:
                exit_price = df_with_signals.iloc[-1]["close"]
                entry_price = position["entry_price"]
                position_size = position["size"]
                position_type = position["type"]

                if position_type == "long":
                    profit = (exit_price - entry_price) * position_size
                else:  # short
                    profit = (entry_price - exit_price) * position_size

                # Apply commission
                profit -= exit_price * position_size * commission

                # Update capital
                capital += profit + (position_size * entry_price)

                # Record trade
                trade = {
                    "symbol": symbol,
                    "entry_time": position["entry_time"],
                    "exit_time": df_with_signals.index[-1],
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "size": position_size,
                    "type": position_type,
                    "profit": profit,
                    "profit_pct": (profit / (position_size * entry_price)) * 100
                    if position_size * entry_price != 0
                    else 0,
                    "exit_reason": "end_of_backtest",
                }
                symbol_trades.append(trade)

                # Update final portfolio value
                equity_curve[-1] = capital

            # Store results for this symbol/timeframe
            results[key] = {
                "equity_curve": equity_curve,
                "trades": symbol_trades,
                "final_capital": equity_curve[-1],
                "return": (equity_curve[-1] - initial_capital) / initial_capital,
                "dates": df_with_signals.index.tolist(),
            }

            # Add trades to overall list
            trades.extend(symbol_trades)

            # Aggregate portfolio history across all symbols
            if len(portfolio_history) == 0:
                portfolio_history = equity_curve
            else:
                # Ensure both lists have the same length
                min_length = min(len(portfolio_history), len(equity_curve))
                portfolio_history = [
                    portfolio_history[i] + equity_curve[i] - initial_capital for i in range(min_length)
                ]

        # Calculate aggregate performance metrics
        try:
            metrics = calculate_metrics(portfolio_history, trades, self.config.get("backtest", {}))
        except Exception:
            # If there's an error, use simpler calculation
            metrics = calculate_metrics(portfolio_history, trades)

        # Store final results
        self.results = {
            "equity_curve": portfolio_history,
            "trades": trades,
            "metrics": metrics,
            "symbol_results": results,
            "config": self.config,
        }

        return dict(self.results)

    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics.

        Returns:
            Dictionary with performance metrics
        """
        if self.results is None:
            raise ValueError("Backtest results not available. Run the backtest first.")

        # If metrics not in results, calculate them
        if "metrics" not in self.results:
            self.results["metrics"] = calculate_metrics(self.results["equity_curve"], self.results["trades"])

        return dict(self.results["metrics"])

    def plot_results(self, filename: Optional[str] = None):
        """Plot backtest results.

        Args:
            filename: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Backtest results not available. Run the backtest first.")

        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [3, 1]})

        # Plot equity curve
        equity_curve = self.results["equity_curve"]
        dates = pd.to_datetime(self.results["symbol_results"][list(self.results["symbol_results"].keys())[0]]["dates"])
        if len(dates) != len(equity_curve):
            dates = dates[: len(equity_curve)]

        ax1.plot(dates, equity_curve)
        ax1.set_title("Equity Curve")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True)

        # Plot drawdown
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max * 100
        ax2.fill_between(dates, drawdown, 0, color="red", alpha=0.3)
        ax2.set_title("Drawdown (%)")
        ax2.set_ylabel("Drawdown %")
        ax2.set_xlabel("Date")
        ax2.grid(True)

        # Adjust layout
        plt.tight_layout()

        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def save_results(self, path: str):
        """Save backtest results to file.

        Args:
            path: Path to save results
        """
        if self.results is None:
            raise ValueError("Backtest results not available. Run the backtest first.")

        # Convert non-serializable objects
        results_to_save = self.results.copy()

        # Convert numpy arrays to lists
        for key in results_to_save["symbol_results"]:
            results_to_save["symbol_results"][key]["equity_curve"] = [
                float(x) for x in results_to_save["symbol_results"][key]["equity_curve"]
            ]
            results_to_save["symbol_results"][key]["dates"] = [
                str(x) for x in results_to_save["symbol_results"][key]["dates"]
            ]

        # Convert datetime objects in trades
        for trade in results_to_save["trades"]:
            trade["entry_time"] = str(trade["entry_time"])
            trade["exit_time"] = str(trade["exit_time"])

        # Convert equity curve
        results_to_save["equity_curve"] = [float(x) for x in results_to_save["equity_curve"]]

        # Save to file
        with open(path, "w") as f:
            json.dump(results_to_save, f, indent=4)

    def generate_report(self, path: str):
        """Generate a detailed HTML report.

        Args:
            path: Path to save the report
        """
        if self.results is None:
            raise ValueError("Backtest results not available. Run the backtest first.")

        # Generate report using a template
        from btb.backtest.report import generate_report

        generate_report(self.results, path)
