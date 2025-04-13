# Trading Strategies

## Overview

BestTradingBot offers a variety of trading strategies powered by machine learning models. This document describes the available strategies, their implementation, and how to customize them for your specific needs.

## Strategy Framework

All trading strategies in BTB follow a common interface, making it easy to create and test new strategies:

```python
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, params: Dict):
        """Initialize strategy with parameters."""
        self.params = params

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the data.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass

    def calculate_position_size(self, capital: float, price: float) -> float:
        """Calculate position size based on available capital.

        Args:
            capital: Available capital
            price: Current asset price

        Returns:
            Position size in base currency units
        """
        position_pct = self.params.get("position_size", 0.1)  # Default 10%
        return (capital * position_pct) / price

    @abstractmethod
    def should_update_stops(self, position, current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Determine if stop loss/take profit should be updated.

        Args:
            position: Current position information
            current_price: Current asset price

        Returns:
            Tuple of (update_required, new_stop_loss, new_take_profit)
        """
        pass
```

## Available Strategies

### 1. Transformer-Based Strategy

This strategy uses a Transformer neural network to predict price movements and generate trading signals.

```python
class TransformerStrategy(BaseStrategy):
    """Trading strategy based on Transformer model predictions."""

    def __init__(self, params: Dict):
        super().__init__(params)
        self.model = self._load_model(params["model_path"])
        self.confidence_threshold = params.get("confidence_threshold", 0.6)
        self.sequence_length = params.get("sequence_length", 60)
        self.prediction_horizon = params.get("prediction_horizon", 24)
        self.stop_loss = params.get("stop_loss", 0.02)  # 2%
        self.take_profit = params.get("take_profit", 0.04)  # 4%
        self.trailing_stop = params.get("trailing_stop", False)
        self.trailing_activation = params.get("trailing_stop_activation", 0.01)  # 1%
        self.trailing_distance = params.get("trailing_stop_distance", 0.005)  # 0.5%

    def _load_model(self, model_path: str):
        """Load the trained model from file."""
        import torch
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        model = self._build_model_from_config(checkpoint["config"])
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        return model

    def _build_model_from_config(self, config):
        """Build model architecture from config."""
        from btb.models.transformer import TransformerModel
        return TransformerModel(**config)

    def _preprocess_data(self, data: pd.DataFrame) -> torch.Tensor:
        """Preprocess data for model input."""
        # Implementation details omitted for brevity
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on model predictions."""
        import torch

        # Add signals column to data
        data = data.copy()
        data["signal"] = 0

        # Skip if not enough data
        if len(data) < self.sequence_length + self.prediction_horizon:
            return data

        # Prepare data for model
        X = self._preprocess_data(data)

        # Generate predictions
        with torch.no_grad():
            predictions = self.model(X).numpy()

        # Convert predictions to signals
        for i in range(len(data) - self.prediction_horizon):
            pred_return = predictions[i, 0]  # Predicted return
            confidence = abs(predictions[i, 1])  # Prediction confidence

            # Generate signal if confidence exceeds threshold
            if confidence > self.confidence_threshold:
                signal = 1 if pred_return > 0 else -1
                data.loc[data.index[i + self.prediction_horizon], "signal"] = signal

        return data

    def should_update_stops(self, position, current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Update stop loss and take profit levels."""
        if not position or not position["is_open"]:
            return False, None, None

        entry_price = position["entry_price"]
        position_type = position["type"]  # 'long' or 'short'
        current_stop = position.get("stop_loss")
        current_take = position.get("take_profit")

        # Calculate profit percentage
        if position_type == "long":
            profit_pct = (current_price - entry_price) / entry_price
        else:  # short
            profit_pct = (entry_price - current_price) / entry_price

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
```

### 2. LSTM with Attention Strategy

This strategy uses an LSTM network with an attention mechanism to capture long-term dependencies in price data.

```python
class LSTMAttentionStrategy(BaseStrategy):
    """Trading strategy based on LSTM with attention."""

    # Implementation details similar to TransformerStrategy
    pass
```

### 3. Ensemble Strategy

Combines predictions from multiple models to generate more robust trading signals.

```python
class EnsembleStrategy(BaseStrategy):
    """Trading strategy that combines signals from multiple models."""

    def __init__(self, params: Dict):
        super().__init__(params)
        self.models = self._load_models(params["model_paths"])
        self.weights = params.get("model_weights", None)  # Optional weighting
        # ... other initialization ...

    def _load_models(self, model_paths: List[str]):
        """Load multiple models."""
        models = []
        for path in model_paths:
            # Load model
            pass
        return models

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals by aggregating predictions from all models."""
        # Implementation details omitted for brevity
        pass
```

### 4. Reinforcement Learning Strategy

Uses reinforcement learning to learn optimal trading actions directly from market data.

```python
class RLStrategy(BaseStrategy):
    """Trading strategy based on reinforcement learning."""

    def __init__(self, params: Dict):
        super().__init__(params)
        self.agent = self._load_agent(params["agent_path"])
        # ... other initialization ...

    def _load_agent(self, agent_path: str):
        """Load trained RL agent."""
        # Implementation details omitted for brevity
        pass

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate signals using RL agent."""
        # Implementation details omitted for brevity
        pass
```

## Strategy Configuration

Each strategy can be configured via YAML files. Here's an example configuration for the Transformer strategy:

```yaml
# config/strategies/transformer_strategy.yaml
strategy:
  name: "transformer_strategy"
  model_path: "models/transformer_btcusdt_1h.pt"
  confidence_threshold: 0.65
  sequence_length: 60
  prediction_horizon: 24
  position_size: 0.1  # 10% of capital per trade
  max_open_positions: 3
  stop_loss: 0.02  # 2%
  take_profit: 0.04  # 4%
  trailing_stop: true
  trailing_stop_activation: 0.01  # 1%
  trailing_stop_distance: 0.005  # 0.5%
```

## Risk Management

All strategies incorporate risk management features:

1. **Position Sizing**: Control how much capital to allocate per trade
2. **Stop Loss**: Automatically exit losing trades at a predetermined level
3. **Take Profit**: Secure profits at a predetermined level
4. **Trailing Stop**: Dynamic stop loss that follows price in profitable trades
5. **Maximum Open Positions**: Limit the number of concurrent trades
6. **Confidence Thresholds**: Only trade when model predictions exceed a confidence level

## Creating Custom Strategies

To create a custom strategy:

1. Subclass `BaseStrategy` and implement the required methods
2. Register your strategy in the strategy factory
3. Create a configuration file for your strategy

Example of a custom strategy implementation:

```python
from btb.strategies.base import BaseStrategy
from btb.strategies.factory import register_strategy

@register_strategy("my_custom_strategy")
class MyCustomStrategy(BaseStrategy):
    """A custom trading strategy."""

    def __init__(self, params: Dict):
        super().__init__(params)
        # Custom initialization

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Implement your custom signal generation logic."""
        # Implementation
        return data_with_signals

    def should_update_stops(self, position, current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Implement custom stop loss/take profit logic."""
        # Implementation
        return update_needed, new_stop, new_take
```

## Strategy Evaluation

All strategies should be thoroughly tested using the backtesting framework before live deployment. Important metrics to evaluate include:

- Total return
- Risk-adjusted return (Sharpe ratio, Sortino ratio)
- Maximum drawdown
- Win rate
- Profit factor

See the [Backtesting Framework](BACKTESTING.html) documentation for details on strategy evaluation.
