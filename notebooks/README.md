# Notebooks

This directory contains Jupyter notebooks for data exploration, model development, strategy testing, and live trading monitoring. These notebooks provide an interactive way to understand and work with the BTB trading system.

## Overview

1. **Data Exploration (`1_data_exploration.ipynb`)**: 
   - Load and visualize cryptocurrency market data
   - Calculate and analyze technical indicators
   - Detect price anomalies
   - Perform correlation analysis
   - Analyze feature importance

2. **Model Development (`2_model_development.ipynb`)**: 
   - Prepare time series data for machine learning
   - Train and evaluate transformer and LSTM models
   - Compare model performance
   - Optimize hyperparameters
   - Generate trading signals from predictions

3. **Strategy Testing (`3_strategy_testing.ipynb`)**: 
   - Backtest trading strategies
   - Calculate performance metrics
   - Visualize equity curves and drawdowns
   - Perform walk-forward analysis
   - Run Monte Carlo simulations
   - Optimize strategy parameters and position sizing

4. **Live Trading Monitoring (`4_live_trading_monitoring.ipynb`)**: 
   - Connect to exchanges
   - Monitor active positions
   - Analyze strategy signals in real-time
   - Track and visualize trading performance
   - Adjust strategy parameters based on market conditions

## Usage

These notebooks are designed for:

1. **Research & Development**: Explore data, test hypotheses, and develop new models and strategies.
2. **Strategy Optimization**: Fine-tune trading parameters and risk management settings.
3. **Performance Analysis**: Evaluate historical and live trading performance.
4. **Education & Documentation**: Understand how the trading system works in an interactive format.

## Getting Started

1. Install required dependencies:
   ```
   pip install -r ../requirements.txt
   ```

2. Launch Jupyter:
   ```
   jupyter notebook
   ```

3. Open any notebook to begin exploration.

## Best Practices

1. **Version Control**: Make copies of notebooks before making significant changes.
2. **API Keys**: Never commit real API keys to version control. Use environment variables or config files.
3. **Resource Management**: Close notebooks when not in use to free up resources, especially when using GPU-accelerated models.
4. **Production Readiness**: These notebooks are for research and analysis. Use the main BTB modules for production deployment.