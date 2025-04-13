"""Data preprocessing module for market data."""

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


class DataPreprocessor:
    """Data preprocessing for market data."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize with optional configuration.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.scalers: Dict[str, Any] = {}  # Store scalers for each symbol/timeframe

    def process(
        self,
        data: Dict[str, pd.DataFrame],
        add_technical_indicators: bool = True,
        normalize: Optional[str] = None,
        fill_missing: Optional[str] = "ffill",
    ) -> Dict[str, pd.DataFrame]:
        """Process market data.

        Args:
            data: Dictionary mapping symbol_timeframe to DataFrames
            add_technical_indicators: Whether to add technical indicators
            normalize: Normalization method ("min_max", "z_score", "robust", or None)
            fill_missing: Method to fill missing values ("ffill", "bfill", "interpolate", or None)

        Returns:
            Processed data dictionary
        """
        processed_data = {}

        for key, df in data.items():
            # Make a copy to avoid modifying the original
            df_processed = df.copy()

            # Fill missing values if requested
            if fill_missing:
                df_processed = self._fill_missing_values(df_processed, method=fill_missing)

            # Add technical indicators if requested
            if add_technical_indicators:
                df_processed = self.add_technical_indicators(df_processed)

            # Remove rows with NaN values (usually at the beginning due to indicators)
            df_processed = df_processed.dropna()

            # Normalize data if requested
            if normalize:
                df_processed, scaler = self._normalize_data(df_processed, method=normalize)
                self.scalers[key] = scaler

            processed_data[key] = df_processed

        return processed_data

    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()

        # Calculate daily return
        df["daily_return"] = df["close"].pct_change(1)

        # Calculate weekly and monthly returns
        df["weekly_return"] = df["close"].pct_change(7)  # Approximate for 7 periods
        df["monthly_return"] = df["close"].pct_change(30)  # Approximate for 30 periods

        # Calculate volatility (standard deviation of returns over a window)
        df["volatility"] = df["daily_return"].rolling(window=21).std()

        # Moving Averages (more granular to match our feature list)
        for period in [7, 21, 50, 200]:
            df[f"ma_{period}"] = df["close"].rolling(window=period).mean()

        # Bollinger Bands (20-period, 2 standard deviations)
        bb_period = 20
        df["bb_middle"] = df["close"].rolling(window=bb_period).mean()
        df["bb_std"] = df["close"].rolling(window=bb_period).std()
        df["bb_upper"] = df["bb_middle"] + 2 * df["bb_std"]
        df["bb_lower"] = df["bb_middle"] - 2 * df["bb_std"]

        # RSI (Relative Strength Index)
        delta = df["close"].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)

        # Calculate average gain and average loss
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()

        # Calculate RS and RSI
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df["close"].ewm(span=12, adjust=False).mean()
        ema_26 = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = ema_12 - ema_26
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # ATR (Average True Range)
        tr1 = df["high"] - df["low"]
        tr2 = abs(df["high"] - df["close"].shift())
        tr3 = abs(df["low"] - df["close"].shift())
        df["tr"] = pd.DataFrame({"tr1": tr1, "tr2": tr2, "tr3": tr3}).max(axis=1)
        df["atr"] = df["tr"].rolling(window=14).mean()

        # Stochastic Oscillator
        n = 14
        df["stoch_k"] = (
            (df["close"] - df["low"].rolling(window=n).min())
            / (df["high"].rolling(window=n).max() - df["low"].rolling(window=n).min())
        ) * 100
        df["stoch_d"] = df["stoch_k"].rolling(window=3).mean()

        # On-Balance Volume (OBV)
        df["obv"] = 0.0  # Esplicitamente definito come float
        obv_values = ((df["close"] > df["close"].shift(1)).astype(int) * 2 - 1) * df["volume"]
        df.iloc[1:, df.columns.get_loc("obv")] = obv_values.iloc[1:].values
        df["obv"] = df["obv"].cumsum()

        # Average Directional Index (ADX)
        # Calculate +DI and -DI
        plus_dm = df["high"].diff()
        minus_dm = df["low"].diff().abs() * -1
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        # True Range
        tr = df["tr"]

        # Smoothed values
        smoothing_period = 14
        plus_di = 100 * (plus_dm.rolling(window=smoothing_period).sum() / tr.rolling(window=smoothing_period).sum())
        minus_di = 100 * (minus_dm.rolling(window=smoothing_period).sum() / tr.rolling(window=smoothing_period).sum())

        # Directional Movement Index
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))

        # Average Directional Index
        df["adx"] = dx.rolling(window=smoothing_period).mean()

        # Return adjusted data
        return df

    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features to the data.

        Args:
            data: DataFrame with datetime index

        Returns:
            DataFrame with added temporal features
        """
        df = data.copy()

        # Ensure index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index is not a DatetimeIndex")

        # Time-based features
        df["hour"] = df.index.hour
        df["day_of_week"] = df.index.dayofweek
        df["day_of_month"] = df.index.day
        df["month"] = df.index.month
        df["quarter"] = df.index.quarter
        df["year"] = df.index.year

        # Cyclical encoding of time features
        # Hour of day (0-23) -> cyclic encoding
        df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)

        # Day of week (0-6) -> cyclic encoding
        df["day_of_week_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_of_week_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        # Month (1-12) -> cyclic encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        return df

    def _fill_missing_values(self, data: pd.DataFrame, method: str = "ffill") -> pd.DataFrame:
        """Fill missing values in the data.

        Args:
            data: DataFrame with potentially missing values
            method: Method to fill missing values ("ffill", "bfill", "interpolate")

        Returns:
            DataFrame with filled missing values
        """
        df = data.copy()

        if method == "ffill":
            # Forward fill
            df = df.ffill()
            # In case there are NaNs at the beginning
            df = df.bfill()
        elif method == "bfill":
            # Backward fill
            df = df.bfill()
            # In case there are NaNs at the end
            df = df.ffill()
        elif method == "interpolate":
            # Linear interpolation
            df = df.interpolate(method="linear")
            # Fill remaining NaNs (at edges) with ffill and bfill
            df = df.ffill().bfill()
        else:
            raise ValueError(f"Unsupported fill method: {method}")

        return df

    def _normalize_data(self, data: pd.DataFrame, method: str = "min_max") -> Tuple[pd.DataFrame, Any]:
        """Normalize features in the data.

        Args:
            data: DataFrame with features
            method: Normalization method ("min_max", "z_score", "robust")

        Returns:
            Tuple of (normalized DataFrame, scaler object)
        """
        df = data.copy()

        # Select columns to normalize (exclude datetime-based columns)
        exclude_cols = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
        cols_to_normalize = [col for col in df.columns if col not in exclude_cols]

        # Select the appropriate scaler
        if method == "min_max":
            scaler = MinMaxScaler()
        elif method == "z_score":
            scaler = StandardScaler()
        elif method == "robust":
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unsupported normalization method: {method}")

        # Fit the scaler and transform the data
        normalized_data = scaler.fit_transform(df[cols_to_normalize])

        # Convert back to DataFrame
        normalized_df = pd.DataFrame(normalized_data, columns=cols_to_normalize, index=df.index)

        # Add back any columns that weren't normalized
        for col in exclude_cols:
            normalized_df[col] = df[col]

        return normalized_df, scaler

    def create_sequences(
        self,
        data: pd.DataFrame,
        sequence_length: int,
        target_column: str = "close",
        prediction_horizon: int = 1,
        classification: bool = False,
        threshold: float = 0.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create input sequences and targets for time series prediction.

        Args:
            data: DataFrame with features
            sequence_length: Length of input sequences
            target_column: Column to predict
            prediction_horizon: Number of steps ahead to predict
            classification: Whether to create binary classification targets
            threshold: Threshold for classification (if positive change > threshold, target = 1)

        Returns:
            Tuple of (input sequences array, target values array)
        """
        df = data.copy()

        # Prepare the target
        if classification:
            # Create classification target (1 if price goes up, 0 if it goes down)
            future_price = df[target_column].shift(-prediction_horizon)
            price_change = (future_price - df[target_column]) / df[target_column]
            df["target"] = (price_change > threshold).astype(int)
        else:
            # Regression target: future price or return
            future_price = df[target_column].shift(-prediction_horizon)
            df["target"] = (future_price - df[target_column]) / df[target_column]  # Return

        # Drop NaN values (from target calculation)
        df = df.dropna()

        # Extract features and target
        features = df.drop(["target"], axis=1).values
        targets = df["target"].values

        # Create sequences
        X, y = [], []
        for i in range(len(df) - sequence_length + 1):
            X.append(features[i : i + sequence_length])
            y.append(targets[i + sequence_length - 1])

        return np.array(X), np.array(y)
