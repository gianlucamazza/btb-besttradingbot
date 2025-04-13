# BestTradingBot - Quick Start Guide

This guide provides a step-by-step walkthrough to get you up and running with BestTradingBot quickly.

## 1. Installation

```bash
# Clone the repository (if you haven't already)
git clone https://github.com/gianlucamazza/btb-besttradingbot.git
cd btb-besttradingbot

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install the package in development mode
pip install -e .
```

## 2. Data Preparation

Before developing trading strategies or running backtests, you need data:

```bash
# Create necessary directories
mkdir -p data/processed

# Run data preparation command
btb prepare-data --config config/backtest_config.yaml --output-dir data/processed

# This will create processed data files in data/processed/ directory
# You should see files like BTCUSDT_1h_processed.csv
```

## 3. Train a Model

```bash
# Train an LSTM model
btb train --data data/processed/BTCUSDT_1h_processed.csv --model lstm --config config/model_config.yaml


# This will save a model to the models/ directory
```

## 4. Run a Backtest

```bash
# Run a backtest using the configuration file
btb backtest --config config/backtest_config.yaml

# Generate a backtest report
btb backtest --config config/backtest_config.yaml --report results/backtest_report.html
```

## 5. Analyze Results in Jupyter Notebooks

```bash
# Start Jupyter
jupyter notebook

# Open the following notebooks:
# 5_demo_run.ipynb - For a quick demonstration
# 3_strategy_testing.ipynb - For detailed strategy analysis
```

## 6. Live Trading Setup

For live trading with Bybit, you need to:

1. Create a `.env` file with your API credentials:

```bash
cp .env.example .env
# Edit .env and add your Bybit API key and secret
```

2. Configure trading parameters in `config/trading_config.yaml`

3. Start live trading:

```bash
btb run --config config/trading_config.yaml
```

## 7. Model and Strategy Development

To develop your own strategies:

1. Explore existing strategies in `btb/strategies/`
2. Create a new strategy file, e.g., `btb/strategies/my_strategy.py`
3. Implement the `BaseStrategy` interface with your logic
4. Register your strategy in `btb/strategies/factory.py`
5. Test your strategy through backtest

To develop your own models:

1. Explore existing models in `btb/models/`
2. Create a new model file, e.g., `btb/models/my_model.py`
3. Implement the `BaseModel` interface with your architecture
4. Train your model using the training utilities

## 8. Understanding Results

When running a backtest, pay attention to:

- **Total Return**: Overall percentage gain/loss
- **Sharpe Ratio**: Risk-adjusted return (higher is better, >1 is good)
- **Maximum Drawdown**: Largest percentage drop from peak to trough
- **Win Rate**: Percentage of profitable trades

The HTML report will provide visualizations of:
- Equity curve
- Drawdown periods
- Trade distributions
- Monthly returns

## 9. Command Line Interface (CLI) Reference

**Backtest:**
```
btb backtest --config CONFIG_FILE [--output OUTPUT_FILE] [--report REPORT_FILE] [--plot] [--verbose]
```

**Train:**
```
btb train --data DATA_FILE --model MODEL_TYPE [--config CONFIG_FILE] [--output OUTPUT_DIR] [--epochs EPOCHS]
```

**Train All Models:**
```
btb train-all --data-dir DATA_DIR --model MODEL_TYPE [--config CONFIG_FILE] [--output OUTPUT_DIR] [--symbols SYMBOLS] [--timeframes TIMEFRAMES] [--parallel]
```

**Data Preparation:**
```
btb prepare-data --config CONFIG_FILE [--output-dir OUTPUT_DIR] [--verbose]
```

**Live Trading:**
```
btb run --config CONFIG_FILE [--verbose]
```

**Cleanup:**
```
btb cleanup [--logs] [--cache] [--pycache] [--verbose]
```

## 10. Common Issues and Solutions

- **Exchange API issues**: Ensure your API keys have the correct permissions
- **Model training errors**: Check data dimensions and ensure sequence lengths match
- **Backtesting performance**: Optimize strategy parameters in config files
- **Memory usage concerns**: Reduce sequence lengths or batch sizes for larger datasets

## Next Steps

- Explore walk-forward analysis for robust strategy validation
- Try different combinations of technical indicators
- Experiment with ensemble strategies combining multiple models
- Check documentation for advanced usage scenarios

For full documentation, please refer to the main [Home page](index.md) and other documentation sections.
