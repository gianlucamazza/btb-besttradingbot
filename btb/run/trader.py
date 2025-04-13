"""Live trading engine for the BestTradingBot."""

import signal
import threading
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from btb.data.loader import DataLoader
from btb.data.preprocessing import DataPreprocessor
from btb.exchange.base import BaseExchange
from btb.exchange.factory import create_exchange
from btb.strategies.base import BaseStrategy
from btb.strategies.factory import create_strategy
from btb.utils.logging import setup_logger


class Trader:
    """Live trading engine."""

    def __init__(self, config: Dict):
        """Initialize trader with configuration.

        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.logger = setup_logger("btb.trader")
        self.exchange = self._init_exchange()
        self.strategy = self._init_strategy()
        self.data_manager = self._init_data_manager()
        self.risk_manager = self._init_risk_manager()

        # Internal state
        self.running = False
        self.positions: Dict[str, Dict[str, Any]] = {}
        self.orders: Dict[str, Dict[str, Any]] = {}
        self.data_cache: Dict[str, Optional[pd.DataFrame]] = {}
        self.last_update_time: Dict[str, datetime] = {}

        # Thread control
        self.trading_thread = None
        self.stop_event = threading.Event()

    def _init_exchange(self) -> BaseExchange:
        """Initialize exchange connection."""
        self.logger.info("Initializing exchange connection")
        exchange_config = self.config["exchange"]
        exchange_name = exchange_config["name"]

        # Load API credentials from environment variables
        import os

        from dotenv import load_dotenv

        load_dotenv()

        api_key = os.getenv(f"{exchange_name.upper()}_API_KEY")
        api_secret = os.getenv(f"{exchange_name.upper()}_API_SECRET")

        if not api_key or not api_secret:
            self.logger.error("API credentials not found in environment variables")
            raise ValueError("API credentials not found")

        # Update exchange config with credentials
        exchange_config["api_key"] = api_key
        exchange_config["api_secret"] = api_secret

        return create_exchange(exchange_name, exchange_config)

    def _init_strategy(self) -> BaseStrategy:
        """Initialize trading strategy."""
        self.logger.info("Initializing trading strategy")
        strategy_name = self.config["trading"]["strategy"]
        strategy_params = self.config.get("strategy_params", {})

        return create_strategy(strategy_name, strategy_params)

    def _init_data_manager(self) -> DataLoader:
        """Initialize data manager."""
        self.logger.info("Initializing data manager")
        data_config = self.config["data"]
        return DataLoader(data_config)

    def _init_risk_manager(self) -> "RiskManager":
        """Initialize risk manager."""
        self.logger.info("Initializing risk manager")
        risk_config = self.config["risk"]
        return RiskManager(risk_config)

    def start(self):
        """Start the trading process."""
        if self.running:
            self.logger.warning("Trading already running")
            return

        self.logger.info("Starting trading process")
        self.running = True
        self.stop_event.clear()

        # Register signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Start trading thread
        self.trading_thread = threading.Thread(target=self._trading_loop)
        self.trading_thread.daemon = True
        self.trading_thread.start()

        self.logger.info("Trading process started")

    def stop(self):
        """Stop the trading process."""
        if not self.running:
            self.logger.warning("Trading already stopped")
            return

        self.logger.info("Stopping trading process")
        self.running = False
        self.stop_event.set()

        # Wait for trading thread to exit
        if self.trading_thread and self.trading_thread.is_alive():
            self.trading_thread.join(timeout=10)

        self.logger.info("Trading process stopped")

    def _signal_handler(self, sig, frame):
        """Handle system signals."""
        self.logger.info(f"Received signal {sig}, shutting down")
        self.stop()

    def _trading_loop(self):
        """Main trading loop."""
        self.logger.info("Entering trading loop")
        symbols = self.config["trading"]["symbols"]
        timeframes = self.config["trading"]["timeframes"]
        update_interval = self.config["data"].get("update_interval", 300)  # Default 5 minutes

        while self.running and not self.stop_event.is_set():
            try:
                # Update market data
                self._update_market_data(symbols, timeframes)

                # Process data and generate signals
                for symbol_tf, data in self.data_cache.items():
                    if data is not None and not data.empty:
                        self._process_symbol_data(symbol_tf, data)

                # Monitor open positions
                self._monitor_positions()

                # Sleep until next update
                self.logger.debug(f"Sleeping for {update_interval} seconds")
                self.stop_event.wait(update_interval)

            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                # Sleep for a short time to avoid rapid error loops
                time.sleep(5)

        self.logger.info("Exiting trading loop")

    def _update_market_data(self, symbols: List[str], timeframes: List[str]):
        """Update market data for all symbols and timeframes."""
        current_time = datetime.now()
        lookback_period = self.config["data"].get("lookback_period", 720)  # Default 720 hours (30 days)

        for symbol in symbols:
            for timeframe in timeframes:
                key = f"{symbol}_{timeframe}"
                last_update = self.last_update_time.get(key, None)

                # Check if update is needed
                if last_update is None or (current_time - last_update).total_seconds() > self._timeframe_to_seconds(
                    timeframe
                ):
                    # Unused variables - kept for documentation/reference
                    # start_date = (current_time - timedelta(hours=lookback_period)).strftime("%Y-%m-%d")
                    # end_date = current_time.strftime("%Y-%m-%d")

                    self.logger.debug(f"Updating market data for {key}")
                    try:
                        # Fetch data
                        data = self.exchange.get_market_data(symbol, timeframe, limit=lookback_period)

                        # Preprocess data
                        preprocessor = DataPreprocessor()
                        data = preprocessor.add_technical_indicators(data)

                        # Store in cache
                        self.data_cache[key] = data
                        self.last_update_time[key] = current_time

                    except Exception as e:
                        self.logger.error(f"Error updating market data for {key}: {e}")

    def _timeframe_to_seconds(self, timeframe: str) -> int:
        """Convert timeframe string to seconds."""
        # Parse the timeframe value and unit
        value = int(timeframe[:-1])
        unit = timeframe[-1]

        # Convert to seconds
        if unit == "m":
            return value * 60
        elif unit == "h":
            return value * 60 * 60
        elif unit == "d":
            return value * 24 * 60 * 60
        elif unit == "w":
            return value * 7 * 24 * 60 * 60
        elif unit == "M":
            return value * 30 * 24 * 60 * 60  # Approximate
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    def _process_symbol_data(self, symbol_timeframe: str, data: pd.DataFrame):
        """Process data for a symbol and generate signals."""
        symbol, timeframe = symbol_timeframe.split("_")
        self.logger.debug(f"Processing data for {symbol_timeframe}")

        # Generate signals
        data_with_signals = self.strategy.generate_signals(data)

        # Check the latest signal
        latest_signal = data_with_signals.iloc[-1]["signal"]

        if latest_signal != 0:
            self.logger.info(f"Generated {'BUY' if latest_signal > 0 else 'SELL'} signal for {symbol}")

            # Check if we should execute this signal
            if self._should_execute_signal(symbol, latest_signal):
                # Execute signal
                self._execute_signal(symbol, latest_signal, data_with_signals.iloc[-1]["close"])

    def _should_execute_signal(self, symbol: str, signal: int) -> bool:
        """Determine if a signal should be executed based on risk management."""
        # Check maximum open positions
        max_positions = self.config["trading"].get("max_open_positions", 1)
        if len(self.positions) >= max_positions and symbol not in self.positions:
            self.logger.info(f"Maximum open positions ({max_positions}) reached, skipping signal")
            return False

        # Check if we already have an open position for this symbol
        if symbol in self.positions:
            current_position = self.positions[symbol]

            # Check if the signal is in the opposite direction of the current position
            if (current_position["type"] == "long" and signal < 0) or (
                current_position["type"] == "short" and signal > 0
            ):
                self.logger.info("Signal is opposite to current position, will close existing position")
                return True
            else:
                self.logger.info("Signal is in same direction as current position, skipping")
                return False

        return True

    def _execute_signal(self, symbol: str, signal: int, price: float):
        """Execute a trading signal."""
        self.logger.info(f"Executing {'BUY' if signal > 0 else 'SELL'} signal for {symbol} at {price}")

        # Check if we need to close an existing position first
        if symbol in self.positions:
            self._close_position(symbol, price)

        # Open new position
        position_type = "long" if signal > 0 else "short"
        self._open_position(symbol, position_type, price)

    def _open_position(self, symbol: str, position_type: str, price: float):
        """Open a new position."""
        try:
            # Get available balance
            balance = self.exchange.get_balance()
            available_capital = balance.get("USDT", {}).get("free", 0)

            if available_capital <= 0:
                self.logger.error("Insufficient balance to open position")
                return

            # Calculate position size
            position_size = self.strategy.calculate_position_size(available_capital, price)

            # Calculate stop loss and take profit levels
            stop_loss, take_profit = self._calculate_stops(position_type, price)

            # Place order
            side = "buy" if position_type == "long" else "sell"
            order_type = self.config["execution"].get("order_type", "LIMIT").lower()

            if order_type == "limit":
                # Add slight offset for limit orders to increase chance of execution
                limit_price = price * 1.002 if position_type == "long" else price * 0.998
                order = self.exchange.place_order(symbol, "limit", side, position_size, limit_price)
            else:  # market order
                order = self.exchange.place_order(symbol, "market", side, position_size)

            # Store order information
            self.orders[order["id"]] = {
                "symbol": symbol,
                "type": position_type,
                "size": position_size,
                "price": price,
                "time": datetime.now(),
                "status": "open",
            }

            # Create position entry
            self.positions[symbol] = {
                "is_open": True,
                "type": position_type,
                "size": position_size,
                "entry_price": price,
                "entry_time": datetime.now(),
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "order_id": order["id"],
            }

            # Set stop loss and take profit orders if supported
            try:
                if stop_loss and hasattr(self.exchange, 'set_stop_loss'):
                    self.exchange.set_stop_loss(symbol, side, stop_loss)
                if take_profit and hasattr(self.exchange, 'set_take_profit'):
                    self.exchange.set_take_profit(symbol, side, take_profit)
            except Exception as e:
                self.logger.warning(f"Error setting stop loss/take profit: {e}")

            self.logger.info(f"Opened {position_type} position for {symbol} at {price} with size {position_size}")

        except Exception as e:
            self.logger.error(f"Error opening position: {e}")

    def _close_position(self, symbol: str, price: float):
        """Close an existing position."""
        try:
            position = self.positions.get(symbol)
            if not position or not position["is_open"]:
                self.logger.warning(f"No open position for {symbol} to close")
                return

            # Determine order details
            position_type = position["type"]
            position_size = position["size"]
            entry_price = position["entry_price"]

            # Place closing order
            side = "sell" if position_type == "long" else "buy"  # Opposite of position type
            order_type = self.config["execution"].get("order_type", "LIMIT").lower()

            if order_type == "limit":
                # Add slight offset for limit orders to increase chance of execution
                limit_price = price * 0.998 if position_type == "long" else price * 1.002
                order = self.exchange.place_order(symbol, "limit", side, position_size, limit_price)
            else:  # market order
                order = self.exchange.place_order(symbol, "market", side, position_size)

            # Calculate profit/loss
            if position_type == "long":
                profit = (price - entry_price) * position_size
            else:  # short
                profit = (entry_price - price) * position_size

            profit_pct = (profit / (position_size * entry_price)) * 100 if position_size * entry_price != 0 else 0

            # Mark position as closed
            position["is_open"] = False
            position["exit_price"] = price
            position["exit_time"] = datetime.now()
            position["profit"] = profit
            position["profit_pct"] = profit_pct
            position["exit_order_id"] = order["id"]

            self.logger.info(
                f"Closed {position_type} position for {symbol} at {price} with profit: {profit:.2f} ({profit_pct:.2f}%)"
            )

            # Remove from active positions
            del self.positions[symbol]

        except Exception as e:
            self.logger.error(f"Error closing position: {e}")

    def _calculate_stops(self, position_type: str, price: float) -> Tuple[Optional[float], Optional[float]]:
        """Calculate stop loss and take profit levels."""
        stop_loss = None
        take_profit = None

        # Get risk configuration
        stop_loss_pct = self.config["risk"].get("stop_loss")
        take_profit_pct = self.config["risk"].get("take_profit")

        if position_type == "long":
            if stop_loss_pct:
                stop_loss = price * (1 - stop_loss_pct)
            if take_profit_pct:
                take_profit = price * (1 + take_profit_pct)
        else:  # short
            if stop_loss_pct:
                stop_loss = price * (1 + stop_loss_pct)
            if take_profit_pct:
                take_profit = price * (1 - take_profit_pct)

        return stop_loss, take_profit

    def _monitor_positions(self):
        """Monitor open positions and update stop loss/take profit levels."""
        if not self.positions:
            return

        for symbol, position in list(self.positions.items()):
            if not position["is_open"]:
                continue

            try:
                # Get current price
                ticker = self.exchange.get_ticker(symbol)
                current_price = ticker["last"]

                # Check if we need to update stops
                should_update, new_stop, new_take = self.strategy.should_update_stops(position, current_price)

                if should_update:
                    self.logger.info(f"Updating stops for {symbol}: SL={new_stop}, TP={new_take}")

                    # Update local position info
                    position["stop_loss"] = new_stop
                    position["take_profit"] = new_take

                    # Update on exchange
                    try:
                        side = "buy" if position["type"] == "long" else "sell"
                        if new_stop and hasattr(self.exchange, 'set_stop_loss'):
                            self.exchange.set_stop_loss(symbol, side, new_stop)
                        if new_take and hasattr(self.exchange, 'set_take_profit'):
                            self.exchange.set_take_profit(symbol, side, new_take)
                    except Exception as e:
                        self.logger.warning(f"Error updating stop loss/take profit on exchange: {e}")

            except Exception as e:
                self.logger.error(f"Error monitoring position for {symbol}: {e}")


class RiskManager:
    """Risk management component."""

    def __init__(self, config: Dict):
        """Initialize risk manager with configuration.

        Args:
            config: Risk management configuration
        """
        self.config = config
        self.logger = setup_logger("btb.risk")
        self.max_drawdown = config.get("max_drawdown", 0.1)  # Default 10%
        self.portfolio_history: List[float] = []

    def check_drawdown(self, current_value: float) -> bool:
        """Check if maximum drawdown has been exceeded.

        Args:
            current_value: Current portfolio value

        Returns:
            True if drawdown is within limits, False if exceeded
        """
        self.portfolio_history.append(current_value)

        if len(self.portfolio_history) < 2:
            return True

        peak = max(self.portfolio_history)
        drawdown = (peak - current_value) / peak

        if drawdown > self.max_drawdown:
            self.logger.warning(f"Maximum drawdown exceeded: {drawdown:.2%} > {self.max_drawdown:.2%}")
            return False

        return True

    def calculate_position_size(
        self, symbol: str, available_capital: float, price: float, risk_per_trade: float = 0.01
    ) -> float:
        """Calculate position size based on risk per trade.

        Args:
            symbol: Trading symbol
            available_capital: Available capital in account
            price: Current price of the asset
            risk_per_trade: Percentage of capital to risk per trade (default 1%)

        Returns:
            Position size in base currency units
        """
        # Get stop loss percentage from config
        stop_loss_pct = self.config.get("stop_loss", 0.02)  # Default 2%

        # Calculate maximum capital to risk
        max_risk_amount = available_capital * risk_per_trade

        # Calculate position size based on risk and stop loss
        denominator = price * stop_loss_pct
        position_size = max_risk_amount / denominator if denominator != 0 else 0

        return float(position_size)
