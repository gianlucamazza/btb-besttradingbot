# Configuration Guide

This guide explains how to configure Best Trading Bot (BTB) for your specific needs.

## Configuration Files

BTB uses YAML files for configuration. The main configuration files are located in the `config/` directory:

- `backtest_config.yaml`: Configuration for backtesting
- `model_config.yaml`: Configuration for machine learning models
- `trading_config.yaml`: Configuration for live trading

## Backtest Configuration

The `backtest_config.yaml` file controls backtesting parameters:

```yaml
backtest:
  symbols: ["BTCUSDT", "ETHUSDT"]  # Trading pairs to backtest
  timeframe: "1h"  # Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
  start_date: "2022-01-01"  # Backtest start date
  end_date: "2022-12-31"  # Backtest end date
  initial_capital: 10000  # Starting capital in USD
  fee_rate: 0.001  # Trading fee (0.1%)
  slippage: 0.0005  # Slippage estimate (0.05%)

strategy:
  name: "transformer_strategy"  # Strategy to use
  model_path: "models/transformer_btcusdt_1h.pt"  # Path to trained model
  confidence_threshold: 0.65  # Minimum confidence for trade execution
  position_size: 0.1  # Portion of capital per trade (10%)
  stop_loss: 0.02  # Stop loss percentage (2%)
  take_profit: 0.04  # Take profit percentage (4%)
  trailing_stop: true  # Enable trailing stop
```

## Model Configuration

The `model_config.yaml` file defines parameters for machine learning models:

```yaml
model:
  type: "transformer"  # Model type (transformer, lstm)

  # Common parameters
  sequence_length: 60  # Input sequence length
  prediction_horizon: 24  # Future prediction horizon
  batch_size: 64  # Training batch size
  learning_rate: 0.0001  # Learning rate
  epochs: 100  # Maximum training epochs
  early_stopping: 10  # Early stopping patience

  # Transformer specific
  transformer:
    input_dim: 32  # Input dimension
    hidden_dim: 128  # Hidden dimension
    num_layers: 4  # Number of transformer layers
    nhead: 8  # Number of attention heads
    dropout: 0.1  # Dropout rate

  # LSTM specific
  lstm:
    input_dim: 32  # Input dimension
    hidden_dim: 128  # Hidden dimension
    num_layers: 3  # Number of LSTM layers
    dropout: 0.2  # Dropout rate
    bidirectional: true  # Whether to use bidirectional LSTM

# Feature engineering
features:
  include_raw_price: true  # Include raw price data
  include_returns: true  # Include price returns
  normalize: true  # Apply normalization
  technical_indicators:  # Technical indicators to include
    - "RSI"
    - "MACD"
    - "BB"
    - "ATR"
    - "SMA"
    - "EMA"
```

## Trading Configuration

The `trading_config.yaml` file configures live trading:

```yaml
exchange:
  name: "bybit"  # Exchange to use
  testnet: true  # Whether to use testnet (set to false for real trading)

trading:
  symbols: ["BTCUSDT"]  # Trading pairs
  timeframes: ["1h"]  # Timeframes to monitor
  position_size: 0.1  # Portion of capital per trade (10%)
  check_interval: 60  # Data update interval in seconds

strategy:
  name: "transformer_strategy"  # Strategy to use
  model_path: "models/transformer_btcusdt_1h.pt"  # Path to trained model
  confidence_threshold: 0.7  # Minimum confidence for trade execution

risk_management:
  max_open_positions: 3  # Maximum simultaneous positions
  max_daily_trades: 5  # Maximum trades per day
  stop_loss: 0.02  # Stop loss percentage (2%)
  take_profit: 0.04  # Take profit percentage (4%)
  trailing_stop: true  # Enable trailing stop
  trailing_stop_activation: 0.01  # Activate trailing stop after 1% profit
  trailing_stop_distance: 0.005  # Trailing stop follows price at 0.5% distance
```

## Environment Variables

Create a `.env` file in the project root to store sensitive information like API credentials:

```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=True  # Set to False for live trading
```

## Advanced Configuration

### Logging

Logging settings can be adjusted in `btb/utils/logging.py`:

```python
# Log level: DEBUG, INFO, WARNING, ERROR, CRITICAL
log_level = logging.INFO

# Log to console
console_logging = True

# Log to file
file_logging = True
log_file = "logs/btb.log"
```

### Custom Strategies

To add a custom strategy:

1. Create a new file in `btb/strategies/`
2. Subclass `BaseStrategy` and implement required methods
3. Register your strategy in the factory
4. Update the configuration to use your strategy

For more detailed information, see the [API Reference](API_REFERENCE.html).
