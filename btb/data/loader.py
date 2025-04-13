"""Data loading module for market data."""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import ccxt
import numpy as np
import pandas as pd


class DataLoader:
    """Base class for data loading operations."""

    def __init__(self, config: Dict = None):
        """Initialize data loader with configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

    def load_data(
        self, symbols: List[str], timeframes: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Load market data for given symbols and timeframes.

        Args:
            symbols: List of market symbols (e.g., "BTCUSDT")
            timeframes: List of timeframes (e.g., "1h", "4h")
            start_date: Start date for data loading (YYYY-MM-DD format)
            end_date: End date for data loading (YYYY-MM-DD format)

        Returns:
            Dict mapping symbol_timeframe to DataFrames
        """
        # For testing purposes, generate dummy data instead of using real API
        use_dummy = self.config.get("use_dummy", True)
        if use_dummy:
            return self._generate_dummy_data(symbols, timeframes, start_date, end_date)

        result = {}

        # Convert string dates to timestamps
        start_ts = int(datetime.strptime(start_date, "%Y-%m-%d").timestamp() * 1000)
        end_ts = int(datetime.strptime(end_date, "%Y-%m-%d").timestamp() * 1000)

        # Initialize exchange
        exchange_name = self.config.get("exchange", "bybit")
        use_testnet = self.config.get("testnet", True)

        exchange = self._init_exchange(exchange_name, use_testnet)

        # Load data for each symbol and timeframe
        for symbol in symbols:
            for timeframe in timeframes:
                data = self._fetch_ohlcv(exchange, symbol, timeframe, start_ts, end_ts)
                if data is not None and not data.empty:
                    # Store with symbol_timeframe key
                    key = f"{symbol}_{timeframe}"
                    result[key] = data

        return result

    def _generate_dummy_data(
        self, symbols: List[str], timeframes: List[str], start_date: str, end_date: str
    ) -> Dict[str, pd.DataFrame]:
        """Generate dummy data for backtesting purposes."""
        result = {}

        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

        for symbol in symbols:
            for timeframe in timeframes:
                # Convert timeframe to timedelta
                td = self._timeframe_to_timedelta(timeframe)

                # Create date range
                dates = []
                current_date = start_date
                while current_date <= end_date:
                    dates.append(current_date)
                    current_date += td

                # Generate random price data with trend
                n = len(dates)
                base_price = 10000 if symbol.startswith("BTC") else 1000

                # Add some randomness and trend
                price = base_price
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
                df = pd.DataFrame(
                    {"open": opens, "high": highs, "low": lows, "close": closes, "volume": volumes}, index=dates
                )

                # Store with symbol_timeframe key
                key = f"{symbol}_{timeframe}"
                result[key] = df

        return result

    def _timeframe_to_timedelta(self, timeframe: str) -> timedelta:
        """Convert timeframe string to timedelta."""
        value = int(timeframe[:-1])
        unit = timeframe[-1]

        if unit == "m":
            return timedelta(minutes=value)
        elif unit == "h":
            return timedelta(hours=value)
        elif unit == "d":
            return timedelta(days=value)
        elif unit == "w":
            return timedelta(weeks=value)
        else:
            return timedelta(days=value * 30)  # Approximate for months

    def _init_exchange(self, exchange_name: str, use_testnet: bool) -> ccxt.Exchange:
        """Initialize exchange API client."""
        # Map of supported exchanges
        exchange_map = {
            "bybit": ccxt.bybit,
            "binance": ccxt.binance,
            "kucoin": ccxt.kucoin,
        }

        exchange_class = exchange_map.get(exchange_name.lower())
        if exchange_class is None:
            raise ValueError(f"Unsupported exchange: {exchange_name}")

        # Create exchange instance
        exchange = exchange_class(
            {
                "enableRateLimit": True,
            }
        )

        # Set testnet if required
        if use_testnet and hasattr(exchange, "urls") and "test" in exchange.urls:
            exchange.urls["api"] = exchange.urls["test"]

        return exchange

    def _fetch_ohlcv(
        self, exchange: ccxt.Exchange, symbol: str, timeframe: str, start_ts: int, end_ts: int
    ) -> Optional[pd.DataFrame]:
        """Fetch OHLCV data from exchange."""
        try:
            # Determine the limit per request
            limit = (
                exchange.rateLimit["ohlcv"]["limit"]
                if hasattr(exchange, "rateLimit") and "ohlcv" in exchange.rateLimit
                else 1000
            )

            # Calculate timeframe in milliseconds
            timeframe_ms = self._timeframe_to_milliseconds(timeframe)

            # Initialize result list
            all_candles = []
            current_ts = start_ts

            # Fetch data in chunks until we reach end_ts
            while current_ts < end_ts:
                candles = exchange.fetch_ohlcv(symbol, timeframe, current_ts, limit)
                if not candles:
                    break

                all_candles.extend(candles)

                # Update current_ts for the next iteration
                last_candle_ts = candles[-1][0]
                current_ts = last_candle_ts + timeframe_ms

                # Add rate limiting delay
                if hasattr(exchange, "sleep"):
                    exchange.sleep(exchange.rateLimit / 1000)  # Convert to seconds

            # Filter candles by end_ts
            all_candles = [candle for candle in all_candles if candle[0] <= end_ts]

            # Convert to DataFrame
            if all_candles:
                df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                df.set_index("timestamp", inplace=True)
                return df

            return None

        except Exception as e:
            print(f"Error fetching data for {symbol} {timeframe}: {e}")
            return None

    def _timeframe_to_milliseconds(self, timeframe: str) -> int:
        """Convert timeframe string to milliseconds."""
        # Parse the timeframe value and unit
        value = int(timeframe[:-1])
        unit = timeframe[-1]

        # Convert to milliseconds
        if unit == "m":
            return value * 60 * 1000
        elif unit == "h":
            return value * 60 * 60 * 1000
        elif unit == "d":
            return value * 24 * 60 * 60 * 1000
        elif unit == "w":
            return value * 7 * 24 * 60 * 60 * 1000
        elif unit == "M":
            return value * 30 * 24 * 60 * 60 * 1000  # Approximate
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")

    def load_local_data(self, file_path: str) -> pd.DataFrame:
        """Load market data from a local CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            DataFrame with market data
        """
        try:
            # Detect file format from extension
            if file_path.endswith(".csv"):
                df = pd.read_csv(file_path)
            elif file_path.endswith(".parquet"):
                df = pd.read_parquet(file_path)
            elif file_path.endswith(".feather"):
                df = pd.read_feather(file_path)
            elif file_path.endswith(".pickle") or file_path.endswith(".pkl"):
                df = pd.read_pickle(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

            # Convert timestamp column to datetime if it exists
            if "timestamp" in df.columns:
                # Try to infer timestamp format
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df.set_index("timestamp", inplace=True)

            return df

        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return pd.DataFrame()
