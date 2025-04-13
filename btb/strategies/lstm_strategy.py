"""Trading strategy based on LSTM model predictions."""

import logging
import os
from typing import Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch

from btb.strategies.base import BaseStrategy


class LSTMAttentionStrategy(BaseStrategy):
    """Trading strategy based on LSTM with attention model predictions."""

    def __init__(self, params: Dict):
        """Initialize with strategy parameters."""
        super().__init__(params)

        # Configure logger
        self.logger = logging.getLogger("btb.strategies.lstm_strategy")

        # Store models_dir and model_naming_pattern
        self.models_dir = params.get("models_dir", "models/")
        self.model_naming_pattern = params.get("model_naming_pattern", "lstm_{symbol}_{timeframe}_model.pth")

        # Extract symbol and timeframe from params
        self.symbol = params.get("symbol", "")
        self.timeframe = params.get("timeframe", "")

        # Log parameters
        self.logger.info(
            f"Initializing LSTM strategy with models_dir: {self.models_dir}, pattern: {self.model_naming_pattern}"
        )

        # Scan models directory to identify available models
        self.available_models = {}
        if os.path.exists(self.models_dir):
            for filename in os.listdir(self.models_dir):
                if filename.endswith(".pth"):
                    for symbol in params.get("symbols", ["BTCUSDT", "ETHUSDT"]):
                        for timeframe in params.get("timeframes", ["1h", "4h"]):
                            model_name = self.model_naming_pattern.format(symbol=symbol, timeframe=timeframe)
                            if filename == model_name:
                                self.available_models[f"{symbol}_{timeframe}"] = os.path.join(self.models_dir, filename)
                                self.logger.info(f"Found model for {symbol}_{timeframe}: {filename}")

        # Store loaded models cache
        self.loaded_models: Dict[str, Any] = {}

        # Determine the model path if explicitly provided
        if "model_path" in params:
            # If model_path is explicitly provided, use it
            self.model_path = params["model_path"]
            # Pre-load the model
            self.model = self._load_model(self.model_path)
        else:
            # Models will be loaded dynamically based on symbol and timeframe
            self.model_path = None
            self.model = None
            self.logger.info(f"Will load models dynamically. Available: {list(self.available_models.keys())}")

        self.confidence_threshold = params.get("confidence_threshold", 0.6)
        self.sequence_length = params.get("sequence_length", 60)
        self.prediction_horizon = params.get("prediction_horizon", 24)
        self.stop_loss = params.get("stop_loss", 0.02)  # 2%
        self.take_profit = params.get("take_profit", 0.04)  # 4%
        self.trailing_stop = params.get("trailing_stop", False)

        # Uniformo i nomi dei parametri per allinearsi con la configurazione
        self.trailing_stop_activation = params.get("trailing_stop_activation", 0.01)  # 1%
        self.trailing_stop_distance = params.get("trailing_stop_distance", 0.005)  # 0.5%

    def _load_model(self, model_path: str):
        """Load the trained model from file."""
        from btb.models.lstm import LSTMModel

        # Check if model is already loaded
        if model_path in self.loaded_models:
            return self.loaded_models[model_path]

        try:
            self.logger.info(f"Loading model from {model_path}")
            model = LSTMModel.load(model_path)
            model.model.eval()  # Set to evaluation mode
            # Cache the loaded model
            self.loaded_models[model_path] = model
            return model
        except Exception as e:
            self.logger.error(f"Error loading model from {model_path}: {e}")
            return None

    def _extract_symbol_timeframe(self, data_key: str) -> Tuple[str, str]:
        """Extract symbol and timeframe from data key."""
        parts = data_key.split("_")
        if len(parts) >= 2:
            return parts[0], parts[1]
        return "", ""

    def _preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for model input.

        Args:
            data: DataFrame with market data

        Returns:
            Tensor with preprocessed data
        """
        # Extract features from configuration
        default_features = ["close", "volume", "ma_50", "rsi"]
        features = self.params.get("features", default_features)

        self.logger.info(f"Model expects {len(features)} features: {features}")
        self.logger.info(f"Available columns in data: {data.columns.tolist()}")

        # Check which features are available in the dataframe
        available_features = [f for f in features if f in data.columns]
        missing_features = [f for f in features if f not in data.columns]

        if missing_features:
            self.logger.warning(f"Missing {len(missing_features)} features: {missing_features}")

        if not available_features:
            self.logger.error("None of the required features are available in the data")
            return torch.tensor([])

        # Create a feature matrix with all required features
        # For missing features, use zeros or other default values
        feature_matrix = np.zeros((len(data), len(features)))

        # Fill in available features
        for i, feature in enumerate(features):
            if feature in data.columns:
                feature_matrix[:, i] = data[feature].values
            else:
                # For missing features, we have options:
                # 1. Fill with zeros (done by default)
                # 2. Fill with a constant value
                # 3. Fill with a derived value from existing features

                # Option 3 example: derive from close price if possible
                if feature.startswith("ma_") and "close" in data.columns:
                    # Try to generate simple moving average
                    window = int(feature.split("_")[1])
                    if len(data) >= window:
                        ma_values = data["close"].rolling(window=window).mean().bfill().values
                        feature_matrix[:, i] = ma_values
                elif feature == "daily_return" and "close" in data.columns:
                    daily_returns = data["close"].pct_change(1).fillna(0).values
                    feature_matrix[:, i] = daily_returns
                elif feature == "rsi" and "close" in data.columns:
                    # Simple RSI implementation if missing
                    try:
                        from ta.momentum import RSIIndicator  # type: ignore
                        
                        rsi = RSIIndicator(data["close"], window=14).rsi().fillna(50).values
                        feature_matrix[:, i] = rsi
                    except (ImportError, ModuleNotFoundError, Exception) as e:
                        # If ta library not available, fill with 50 (neutral)
                        self.logger.warning(f"Error calculating RSI: {e}")
                        feature_matrix[:, i] = 50

        self.logger.info(f"Created feature matrix with shape {feature_matrix.shape}")

        # Create sequences
        sequences = []
        for i in range(len(feature_matrix) - self.sequence_length + 1):
            seq = feature_matrix[i : i + self.sequence_length]
            sequences.append(seq)

        if not sequences:
            self.logger.warning("No valid sequences created for prediction")
            return torch.tensor([])

        self.logger.info(f"Created {len(sequences)} sequences with shape {sequences[0].shape}")

        # Convert to tensor
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on model predictions.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with trading signals
        """
        # Add signals column to data
        data = data.copy()
        data["signal"] = 0

        # Skip if not enough data
        if len(data) < self.sequence_length + self.prediction_horizon:
            self.logger.warning(
                f"Not enough data: {len(data)} rows, need at least {self.sequence_length + self.prediction_horizon}"
            )
            return data

        # Find which symbol_timeframe we're working with by checking against all known keys
        key = None
        for k in self.available_models.keys():
            # Use the first one as default
            if key is None:
                key = k

            # If we can extract exact symbol_timeframe from data, use that
            if hasattr(data, "name") and data.name == k:
                key = k
                break

            # Try other attributes
            if hasattr(data, "attrs") and "key" in data.attrs and data.attrs["key"] == k:
                key = k
                break

        if key:
            symbol, timeframe = self._extract_symbol_timeframe(key)
        else:
            # Use defaults
            symbol, timeframe = self.symbol, self.timeframe
            if not symbol or not timeframe:
                self.logger.error("Could not determine symbol/timeframe and no defaults provided")
                return data

        self.logger.info(f"Generating signals for {symbol}_{timeframe}")

        # Get the model from available models
        model_path = self.available_models.get(f"{symbol}_{timeframe}")
        if not model_path:
            self.logger.warning(f"No model available for {symbol}_{timeframe}")
            return data

        # Load the model
        model = self._load_model(model_path)
        if model is None:
            self.logger.error(f"Failed to load model for {symbol}_{timeframe}")
            return data

        # Prepare data for model
        X = self._preprocess_data(data)
        if len(X) == 0:
            self.logger.warning("No valid sequences created for prediction")
            return data

        # Generate predictions
        self.logger.info(f"Generating predictions with shape {X.shape}")
        predictions = model.predict(X)

        # Convert predictions to signals
        signal_count = 0
        for i in range(len(predictions)):
            if i + self.prediction_horizon >= len(data):
                break

            pred_return = predictions[i, 0]  # Predicted return

            # Use confidence if available, otherwise default to 1.0
            confidence = 1.0
            if predictions.shape[1] > 1:
                confidence = abs(predictions[i, 1])

            # Generate signal if confidence exceeds threshold
            if confidence > self.confidence_threshold:
                signal = 1 if pred_return > 0 else -1
                data.loc[data.index[i + self.prediction_horizon], "signal"] = signal
                signal_count += 1

        self.logger.info(f"Generated {signal_count} signals for {symbol}_{timeframe}")

        return data

    def should_update_stops(self, position, current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Update stop loss and take profit levels.

        Args:
            position: Current position information
            current_price: Current asset price

        Returns:
            Tuple of (update_required, new_stop_loss, new_take_profit)
        """
        if not position or not position.get("is_open", False):
            return False, None, None

        entry_price = position["entry_price"]
        position_type = position["type"]  # 'long' or 'short'
        current_stop = position.get("stop_loss")
        current_take = position.get("take_profit")

        # Calculate profit percentage
        if position_type == "long":
            profit_pct = (current_price - entry_price) / entry_price if entry_price != 0 else 0
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price if entry_price != 0 else 0

        # Initialize new stop loss and take profit
        new_stop = current_stop
        new_take = current_take

        # Update trailing stop if enabled and activated
        if self.trailing_stop and profit_pct >= self.trailing_stop_activation:
            if position_type == "long":
                trailing_level = current_price * (1 - self.trailing_stop_distance)
                if not current_stop or trailing_level > current_stop:
                    new_stop = trailing_level
            else:  # short
                trailing_level = current_price * (1 + self.trailing_stop_distance)
                if not current_stop or trailing_level < current_stop:
                    new_stop = trailing_level

        # Determine if update is needed
        update_needed = (new_stop != current_stop) or (new_take != current_take)
        return update_needed, new_stop, new_take
