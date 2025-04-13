"""Trading strategy based on Transformer model predictions."""

import os
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from btb.strategies.base import BaseStrategy


class TransformerStrategy(BaseStrategy):
    """Trading strategy based on Transformer model predictions."""

    def __init__(self, params: Dict):
        """Initialize with strategy parameters."""
        super().__init__(params)

        # Extract symbol and timeframe from params
        self.symbol = params.get("symbol", "")
        self.timeframe = params.get("timeframe", "")

        # Determine the model path based on symbol and timeframe
        if "model_path" in params:
            # If model_path is explicitly provided, use it
            self.model_path = params["model_path"]
        else:
            # Try to find a suitable model based on symbol and timeframe
            models_dir = params.get("models_dir", "models/")

            if self.symbol and self.timeframe:
                # First, try to find a specific model for this asset and timeframe
                specific_model = f"transformer_{self.symbol}_{self.timeframe}_model.pth"
                specific_path = os.path.join(models_dir, specific_model)

                if os.path.exists(specific_path):
                    self.model_path = specific_path
                else:
                    # Fallback to generic model
                    generic_model = "transformer_model.pth"
                    self.model_path = os.path.join(models_dir, generic_model)
            else:
                # If symbol or timeframe not provided, use generic model
                generic_model = "transformer_model.pth"
                self.model_path = os.path.join(models_dir, generic_model)

        self.model = self._load_model(self.model_path)
        self.confidence_threshold = params.get("confidence_threshold", 0.6)
        self.sequence_length = params.get("sequence_length", 60)
        self.prediction_horizon = params.get("prediction_horizon", 24)
        self.stop_loss = params.get("stop_loss", 0.02)  # 2%
        self.take_profit = params.get("take_profit", 0.04)  # 4%
        self.trailing_stop = params.get("trailing_stop", False)
        self.trailing_activation = params.get("trailing_stop_activation", 0.01)  # 1%
        self.trailing_distance = params.get("trailing_stop_distance", 0.005)  # 0.5%
        # Simple moving average periods
        self.short_period = params.get("short_period", 10)
        self.long_period = params.get("long_period", 30)

        # Add default position_size parameter if not provided
        if "position_size" not in params:
            params["position_size"] = 0.1  # Default to 10% of capital

    def _load_model(self, model_path: str):
        """Load the trained model from file."""
        from btb.models.transformer import TransformerModel

        model = TransformerModel.load(model_path)
        model.model.eval()  # Set to evaluation mode
        return model

    def _preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for model input.

        Args:
            data: DataFrame with market data

        Returns:
            Tensor with preprocessed data
        """
        # Extract features
        features = self.params.get("features", ["close", "volume", "ma_50", "rsi"])
        X = data[features].values

        # Create sequences
        sequences = []
        for i in range(len(X) - self.sequence_length + 1):
            seq = X[i : i + self.sequence_length]
            sequences.append(seq)

        if not sequences:
            return torch.tensor([])

        # Convert to tensor
        return torch.tensor(np.array(sequences), dtype=torch.float32)

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on moving average crossover.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with trading signals
        """
        # Add signals column to data
        data = data.copy()
        data["signal"] = 0

        # Skip if not enough data
        if len(data) < self.long_period + 1:
            return data

        # Calculate moving averages
        data["short_ma"] = data["close"].rolling(window=self.short_period).mean()
        data["long_ma"] = data["close"].rolling(window=self.long_period).mean()

        # Generate signals based on crossovers
        for i in range(1, len(data)):
            if pd.isna(data.iloc[i]["short_ma"]) or pd.isna(data.iloc[i]["long_ma"]):
                continue

            # Check for crossover
            prev_short = data.iloc[i - 1]["short_ma"]
            prev_long = data.iloc[i - 1]["long_ma"]
            curr_short = data.iloc[i]["short_ma"]
            curr_long = data.iloc[i]["long_ma"]

            # Generate buy signal
            if prev_short < prev_long and curr_short > curr_long:
                data.iloc[i, data.columns.get_loc("signal")] = 1

            # Generate sell signal
            elif prev_short > prev_long and curr_short < curr_long:
                data.iloc[i, data.columns.get_loc("signal")] = -1

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
        if self.trailing_stop and profit_pct >= self.trailing_activation:
            if position_type == "long":
                trailing_level = current_price * (1 - self.trailing_distance)
                if not current_stop or trailing_level > current_stop:
                    new_stop = trailing_level
            else:  # short
                trailing_level = current_price * (1 + self.trailing_distance)
                if not current_stop or trailing_level < current_stop:
                    new_stop = trailing_level

        # Determine if update is needed
        update_needed = (new_stop != current_stop) or (new_take != current_take)
        return update_needed, new_stop, new_take
