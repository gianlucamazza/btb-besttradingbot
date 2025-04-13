---
redirect: BACKTESTING.md
---

# Redirecting...

If you are not redirected automatically, follow this [link to the backtesting guide](BACKTESTING.md).

# Backtesting Framework

## Overview

The backtesting framework in BestTradingBot allows you to evaluate trading strategies against historical market data before deploying them in live environments. This document explains how to configure, run, and interpret backtest results.

## Key Features

- **Historical Data Processing**: Comprehensive data processing pipeline for multiple assets and timeframes
- **Strategy Testing**: Test multiple strategies with different parameters
- **Performance Metrics**: Calculate risk-adjusted returns and other key performance indicators
- **Visualization**: Generate detailed performance charts and reports
- **Walk-Forward Testing**: Advanced testing with time-based validation
- **Monte Carlo Simulations**: Estimate strategy robustness across different market conditions

## Backtesting Process

### Step 1: Data Collection and Preparation

Before running a backtest, historical market data needs to be collected and prepared:

```python
# Example of data preparation process
from btb.data.historical import HistoricalDataLoader
from btb.data.preprocessing import DataPreprocessor

# Load historical data
data_loader = HistoricalDataLoader()
raw_data = data_loader.load_data(
    symbols=["BTCUSDT", "ETHUSDT"],
    timeframes=["1h", "4h"],
    start_date="2021-01-01",
    end_date="2022-01-01"
)

# Preprocess data
preprocessor = DataPreprocessor()
processed_data = preprocessor.process(
    data=raw_data,
    add_technical_indicators=True,
    normalize=True,
    fill_missing="ffill"
)
```

### Step 2: Configuring the Backtest

Create a backtest configuration in YAML format or programmatically:

```yaml
# config/backtest_config.yaml
backtest:
  start_date: "2021-01-01"
  end_date: "2022-01-01"
  symbols: ["BTCUSDT", "ETHUSDT"]
  timeframes: ["1h", "4h"]
  strategy: "transformer_strategy"  # Strategy name
  initial_capital: 10000  # USDT
  position_size: 0.1  # 10% of capital per trade
  commission: 0.0007  # 0.07% trading fee
  slippage: 0.0001  # 0.01% slippage
  leverage: 1  # No leverage by default

strategy_params:
  model_path: "models/transformer_btcusdt_1h.pt"
  confidence_threshold: 0.65
  max_open_positions: 3
  stop_loss: 0.02  # 2%
  take_profit: 0.04  # 4%
  trailing_stop: true
  trailing_stop_activation: 0.01  # 1%
  trailing_stop_distance: 0.005  # 0.5%
```

### Step 3: Running the Backtest

Run the backtest using the command line or Python API:

```bash
# Command line
btb backtest --config config/backtest_config.yaml --output results/backtest_results.json --report results/backtest_report.html
```

Or programmatically:

```python
from btb.backtest import Backtester
from btb.utils.config import load_config

# Load configuration
config = load_config("config/backtest_config.yaml")

# Initialize and run backtester
backtester = Backtester(config)
results = backtester.run()

# Save results
backtester.save_results("results/backtest_results.json")

# Generate performance report
backtester.generate_report("results/backtest_report.html")
```

## Advanced Backtesting Techniques

### Walk-Forward Analysis

Walk-forward analysis helps validate strategy robustness by training on one time period and testing on another:

```python
from btb.backtest.walk_forward import WalkForwardAnalyzer

# Configure walk-forward analysis
wf_analyzer = WalkForwardAnalyzer(
    data=processed_data,
    strategy="transformer_strategy",
    train_size=180,  # 180 days training window
    test_size=30,    # 30 days testing window
    step_size=30,    # Step forward 30 days each iteration
    strategy_params=config["strategy_params"]
)

# Run analysis
wf_results = wf_analyzer.run()

# Generate report
wf_analyzer.generate_report("results/walk_forward_report.html")
```

### Monte Carlo Simulation

Monte Carlo simulation assesses strategy performance across different market scenarios:

```python
from btb.backtest.monte_carlo import MonteCarloSimulator

# Configure Monte Carlo simulation
mc_simulator = MonteCarloSimulator(
    backtest_results=results,
    num_simulations=1000,
    random_seed=42
)

# Run simulation
mc_results = mc_simulator.run()

# Generate report with confidence intervals
mc_simulator.generate_report("results/monte_carlo_report.html")
```

## Performance Metrics

The backtesting framework calculates numerous performance metrics:

### Return Metrics
- **Total Return**: Overall percentage return
- **Annualized Return**: Return normalized to a yearly rate
- **Daily/Monthly Returns**: Statistical distribution of periodic returns

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (return / volatility)
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Volatility**: Standard deviation of returns
- **Value at Risk (VaR)**: Maximum potential loss at a confidence level

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **Average Win/Loss**: Average profit/loss per trade
- **Maximum Consecutive Wins/Losses**: Longest streak of wins/losses
- **Average Holding Period**: Average time in a position

## Visualization

The backtesting framework generates various visualizations:

- **Equity Curve**: Shows the portfolio value over time
- **Drawdown Chart**: Visualizes drawdowns throughout the testing period
- **Return Distribution**: Histogram of daily/monthly returns
- **Trade Analysis**: Charts showing entry/exit points and position sizing
- **Performance Comparison**: Compare strategy performance against benchmarks

## Interpreting Results

When analyzing backtest results, consider:

1. **Statistical Significance**: Ensure enough trades to draw meaningful conclusions
2. **Overfitting**: Beware of strategies that perform well only on historical data
3. **Robustness**: Check performance across different market conditions
4. **Transaction Costs**: Verify that fees and slippage are realistically modeled
5. **Risk-Adjusted Returns**: Focus on risk-adjusted metrics rather than just total return

## Best Practices

1. **Use Realistic Assumptions**: Model transaction costs, slippage, and execution delays accurately
2. **Avoid Look-Ahead Bias**: Ensure the strategy only uses information available at the time of trading
3. **Test Across Multiple Markets**: Validate strategy performance on various assets
4. **Perform Sensitivity Analysis**: Test how strategy performs with different parameters
5. **Include Market Regimes**: Test through bull markets, bear markets, and sideways markets
6. **Compare to Benchmarks**: Evaluate performance against buy-and-hold or other strategies