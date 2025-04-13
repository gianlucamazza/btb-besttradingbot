---
redirect: CONFIGURATION.md
---

# Redirecting...

If you are not redirected automatically, follow this [link to the configuration guide](CONFIGURATION.md).

# Configuration Guide

## Overview

BestTradingBot uses YAML configuration files to customize trading parameters, model settings, and backtesting options. This guide explains the available configuration options and how to set them up.

## Configuration Files

The main configuration files are located in the `config/` directory:

- `trading_config.yaml`: Configuration for live trading
- `backtest_config.yaml`: Configuration for backtesting
- `model_config.yaml`: Machine learning model parameters
- `logging_config.yaml`: Logging settings

## Trading Configuration

### Example `trading_config.yaml`

```yaml
# Exchange settings
exchange:
  name: bybit
  testnet: true  # Set to false for live trading
  rate_limit: true

# Trading parameters
trading:
  base_currency: BTC
  quote_currency: USDT
  symbols: ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
  timeframes: ["1h", "4h", "1d"]
  strategy: lstm_attention
  position_size: 0.1  # 10% of available balance per trade
  max_open_positions: 3
  
# Risk management
risk:
  max_drawdown: 0.15  # 15% maximum drawdown
  stop_loss: 0.02  # 2% stop loss
  take_profit: 0.04  # 4% take profit
  trailing_stop: true
  trailing_stop_activation: 0.01  # Activate at 1% profit
  trailing_stop_distance: 0.005  # 0.5% trailing distance

# Data collection
data:
  lookback_period: 720  # Hours of historical data to maintain
  features: ["close", "volume", "ma_50", "rsi", "bbands"]
  cache_data: true
  update_interval: 300  # Seconds between data updates

# Execution settings
execution:
  order_type: LIMIT  # MARKET or LIMIT
  limit_order_expiry: 60  # Seconds before a limit order is canceled
  max_retries: 3
  retry_delay: 5  # Seconds between retries
```

## Backtesting Configuration

### Example `backtest_config.yaml`

```yaml
# Backtest settings
backtest:
  start_date: "2022-01-01"
  end_date: "2023-01-01"
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["1h", "4h"]
  strategy: lstm_attention
  initial_capital: 10000  # USDT
  commission: 0.0007  # 0.07% trading fee
  slippage: 0.0001  # 0.01% slippage

# Data processing
data_processing:
  train_test_split: 0.8  # 80% training, 20% testing
  feature_engineering: true
  normalization: "min_max"  # Options: "min_max", "z_score", "robust"
  fill_missing: "ffill"  # Forward fill missing values

# Performance metrics
metrics:
  calculate_sharpe: true
  calculate_sortino: true
  calculate_drawdown: true
  calculate_win_rate: true
  benchmark: "BTCUSDT"  # Symbol to compare performance against
```

## Model Configuration

### Example `model_config.yaml`

```yaml
# Model architecture
model:
  type: transformer  # Options: "lstm", "transformer", "cnn", "ensemble"
  hidden_dim: 128
  num_layers: 3
  dropout: 0.2
  attention_heads: 8
  
# Training parameters
training:
  batch_size: 64
  learning_rate: 0.001
  epochs: 100
  early_stopping: true
  patience: 15
  optimizer: "adam"  # Options: "adam", "sgd", "rmsprop"
  scheduler: "cosine"  # Learning rate scheduler
  weight_decay: 1e-5
  gradient_clipping: 1.0

# Input features
features:
  sequence_length: 60  # Number of time steps to consider
  technical_indicators: true
  sentiment_analysis: true
  include_volume: true
  include_time_features: true
  
# Prediction settings
prediction:
  output_type: "regression"  # "regression" or "classification"
  prediction_horizon: 24  # Hours to predict ahead
  confidence_threshold: 0.7  # Minimum confidence for a trade signal
```

## Environment Variables

Sensitive information such as API keys should be stored in environment variables rather than configuration files. Create a `.env` file in the project root with the following variables:

```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=True  # Set to False for live trading

# Optional database configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_db
DB_USER=user
DB_PASSWORD=password

# Telegram notifications (optional)
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

## Configuration Validation

The system validates all configuration files at startup to ensure all required parameters are present and have valid values. If configuration validation fails, the system will exit with an error message.

For additional configuration validation options, see the main command-line interface documentation.