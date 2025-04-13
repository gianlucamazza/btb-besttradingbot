"""Walk-forward analysis for trading strategies."""

from datetime import timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from btb.backtest.engine import Backtester
from btb.strategies.factory import create_strategy


class WalkForwardAnalyzer:
    """Walk-forward analysis for trading strategies."""

    def __init__(
        self,
        data: Dict[str, pd.DataFrame],
        strategy: str,
        train_size: int,
        test_size: int,
        step_size: int,
        strategy_params: Dict,
    ):
        """Initialize walk-forward analyzer.

        Args:
            data: Dictionary mapping symbol_timeframe to DataFrames
            strategy: Strategy name
            train_size: Training window size in days
            test_size: Testing window size in days
            step_size: Step size in days
            strategy_params: Strategy parameters
        """
        self.data = data
        self.strategy_name = strategy
        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size
        self.strategy_params = strategy_params
        self.results = None

    def _train_model(self, data: Dict[str, pd.DataFrame], model_path: str, model_config: Dict):
        """Train a machine learning model on the training data.

        Args:
            data: Dictionary mapping symbol_timeframe to DataFrames
            model_path: Path to save the trained model
            model_config: Model configuration
        """
        # Import necessary modules for model training
        import numpy as np
        import torch

        from btb.models.lstm import LSTMModel
        from btb.models.transformer import TransformerModel

        # Extract model type from config
        model_type = model_config.get("type", "lstm")

        # Extract all data into one large DataFrame for training
        all_data = []
        for key, df in data.items():
            # Add symbol and timeframe as columns
            df_copy = df.copy()
            symbol, timeframe = key.split("_")
            df_copy["symbol"] = symbol
            df_copy["timeframe"] = timeframe
            all_data.append(df_copy)

        combined_data = pd.concat(all_data)

        # Prepare features and target
        target_col = model_config.get("target_column", "close")

        # Create target (future price change)
        prediction_horizon = model_config.get("prediction_horizon", 1)
        combined_data["target"] = combined_data[target_col].pct_change(prediction_horizon).shift(-prediction_horizon)

        # Drop NaN values
        combined_data = combined_data.dropna()

        # Create sequences
        seq_length = model_config.get("sequence_length", 60)

        def create_sequences(data, seq_length):
            """Create sequences for time series prediction."""
            data.drop(["target"], axis=1).values
            data["target"].values

            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data.iloc[i : (i + seq_length)].drop(["target"], axis=1).values
                y = data.iloc[i + seq_length]["target"]
                xs.append(x)
                ys.append(y)

            return np.array(xs), np.array(ys)

        X, y = create_sequences(combined_data, seq_length)

        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.FloatTensor(y).view(-1, 1)

        # Configure model based on type
        if model_type.lower() == "lstm":
            model_config = {
                "input_dim": X.shape[2],
                "hidden_dim": model_config.get("hidden_dim", 128),
                "num_layers": model_config.get("num_layers", 2),
                "output_dim": 1,
                "dropout": model_config.get("dropout", 0.2),
                "learning_rate": model_config.get("learning_rate", 0.001),
                "batch_size": model_config.get("batch_size", 64),
                "epochs": model_config.get("epochs", 50),
            }
            model = LSTMModel(model_config)

        elif model_type.lower() == "transformer":
            model_config = {
                "input_dim": X.shape[2],
                "d_model": model_config.get("hidden_dim", 128),
                "nhead": model_config.get("attention_heads", 8),
                "num_encoder_layers": model_config.get("num_layers", 2),
                "dim_feedforward": model_config.get("hidden_dim", 128) * 4,
                "dropout": model_config.get("dropout", 0.1),
                "output_dim": 1,
                "learning_rate": model_config.get("learning_rate", 0.001),
                "batch_size": model_config.get("batch_size", 64),
                "epochs": model_config.get("epochs", 50),
            }
            model = TransformerModel(model_config)

        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Train model (use 80% for training, 20% for validation)
        train_size = int(len(X_tensor) * 0.8)
        X_train = X_tensor[:train_size]
        y_train = y_tensor[:train_size]
        X_val = X_tensor[train_size:]
        y_val = y_tensor[train_size:]

        # Prepare data as expected by model.train()
        train_data = (X_train, y_train)
        val_data = (X_val, y_val)

        # Train model
        model.train(train_data, val_data)

        # Save trained model
        model.save(model_path)

    def run(self) -> Dict:
        """Run walk-forward analysis.

        Returns:
            Dictionary with walk-forward results
        """
        all_results = []
        equity_curves = []

        # Get the first DataFrame to determine date range
        first_key = list(self.data.keys())[0]
        df = self.data[first_key]

        # Convert index to datetime if not already
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # Calculate time windows
        start_date = df.index[0]
        end_date = df.index[-1]

        # Create windows
        current_date = start_date
        while current_date + timedelta(days=self.train_size + self.test_size) <= end_date:
            train_start = current_date
            train_end = train_start + timedelta(days=self.train_size)
            test_start = train_end
            test_end = test_start + timedelta(days=self.test_size)

            # Prepare training data
            train_data = {}
            for key, df in self.data.items():
                mask = (df.index >= train_start) & (df.index < train_end)
                train_data[key] = df.loc[mask].copy()

            # Prepare testing data
            test_data = {}
            for key, df in self.data.items():
                mask = (df.index >= test_start) & (df.index < test_end)
                test_data[key] = df.loc[mask].copy()

            # Train model if using a machine learning strategy
            if "model_path" in self.strategy_params:
                model_path = self.strategy_params["model_path"]
                model_config = self.strategy_params.get("model_config", {})
                self._train_model(train_data, model_path, model_config)

            # Create strategy
            strategy = create_strategy(self.strategy_name, self.strategy_params)

            # Run backtest on test data
            config = {
                "backtest": {
                    "start_date": test_start.strftime("%Y-%m-%d"),
                    "end_date": test_end.strftime("%Y-%m-%d"),
                    "symbols": [key.split("_")[0] for key in test_data.keys()],
                    "timeframes": [key.split("_")[1] for key in test_data.keys()],
                    "strategy": self.strategy_name,
                    "initial_capital": 10000,  # Default value
                    "commission": 0.0007,  # Default value
                    "slippage": 0.0001,  # Default value
                },
                "strategy_params": self.strategy_params,
            }

            # Create a custom backtester that uses the test data directly
            backtester = Backtester(config)
            backtester.data = test_data  # Override loaded data with test data
            backtester.strategy = strategy  # Use the strategy we created

            # Run backtest
            results = backtester.run()

            # Store results
            window_result = {
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "test_start": test_start.strftime("%Y-%m-%d"),
                "test_end": test_end.strftime("%Y-%m-%d"),
                "metrics": results["metrics"],
                "equity_curve": results["equity_curve"],
            }
            all_results.append(window_result)
            equity_curves.append(results["equity_curve"])

            # Move to next window
            current_date += timedelta(days=self.step_size)

        # Combine equity curves
        combined_equity = []
        for curve in equity_curves:
            # Normalize to start at 1.0
            if len(curve) > 0:
                normalized = [val / curve[0] for val in curve]
                combined_equity.append(normalized)

        # Calculate aggregate metrics
        agg_metrics = self._calculate_aggregate_metrics(all_results)

        # Store final results
        self.results = {
            "window_results": all_results,
            "aggregate_metrics": agg_metrics,
            "equity_curves": combined_equity,
        }

        return self.results

    def _calculate_aggregate_metrics(self, window_results: List[Dict]) -> Dict:
        """Calculate aggregate metrics across all windows."""
        if not window_results:
            return {}

        # Initialize with zeros
        agg_metrics = {"total_return": 0, "win_rate": 0, "sharpe_ratio": 0, "max_drawdown": 0, "profit_factor": 0}

        # Collect metrics across windows
        total_returns = []
        win_rates = []
        sharpe_ratios = []
        drawdowns = []
        profit_factors = []

        for result in window_results:
            metrics = result["metrics"]
            total_returns.append(metrics.get("total_return", 0))
            win_rates.append(metrics.get("win_rate", 0))
            sharpe_ratios.append(metrics.get("sharpe_ratio", 0))
            drawdowns.append(metrics.get("max_drawdown", 0))
            profit_factors.append(metrics.get("profit_factor", 0))

        # Calculate averages
        agg_metrics["total_return"] = np.mean(total_returns)
        agg_metrics["win_rate"] = np.mean(win_rates)
        agg_metrics["sharpe_ratio"] = np.mean(sharpe_ratios)
        agg_metrics["max_drawdown"] = np.mean(drawdowns)
        agg_metrics["profit_factor"] = np.mean(profit_factors)

        # Add standard deviations
        agg_metrics["total_return_std"] = np.std(total_returns)
        agg_metrics["win_rate_std"] = np.std(win_rates)
        agg_metrics["sharpe_ratio_std"] = np.std(sharpe_ratios)

        # Add min/max values
        agg_metrics["total_return_min"] = min(total_returns)
        agg_metrics["total_return_max"] = max(total_returns)
        agg_metrics["win_rate_min"] = min(win_rates)
        agg_metrics["win_rate_max"] = max(win_rates)

        return agg_metrics

    def generate_report(self, path: str):
        """Generate an HTML report with walk-forward results.

        Args:
            path: Path to save the report
        """
        if self.results is None:
            raise ValueError("Walk-forward analysis results not available. Run the analysis first.")

        # Implementation for generating HTML report
        # This would typically use a template engine like Jinja2
        # For simplicity, we'll just create a basic HTML report

        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Walk-Forward Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
                .header { background-color: #4CAF50; color: white; padding: 20px; margin-bottom: 20px; }
                .section { margin-bottom: 30px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Walk-Forward Analysis Report</h1>
            </div>

            <div class="section">
                <h2>Aggregate Metrics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """

        # Add aggregate metrics
        for key, value in self.results["aggregate_metrics"].items():
            html += f"<tr><td>{key}</td><td>{value:.4f}</td></tr>\n"

        html += """</table>
            </div>

            <div class="section">
                <h2>Window Results</h2>
                <table>
                    <tr>
                        <th>Training Period</th>
                        <th>Testing Period</th>
                        <th>Total Return</th>
                        <th>Win Rate</th>
                        <th>Sharpe Ratio</th>
                        <th>Max Drawdown</th>
                    </tr>
        """

        # Add window results
        for result in self.results["window_results"]:
            train_period = f"{result['train_start']} to {result['train_end']}"
            test_period = f"{result['test_start']} to {result['test_end']}"
            total_return = result["metrics"].get("total_return", 0) * 100
            win_rate = result["metrics"].get("win_rate", 0) * 100
            sharpe = result["metrics"].get("sharpe_ratio", 0)
            drawdown = result["metrics"].get("max_drawdown", 0) * 100

            html += f"""<tr>
                <td>{train_period}</td>
                <td>{test_period}</td>
                <td>{total_return:.2f}%</td>
                <td>{win_rate:.2f}%</td>
                <td>{sharpe:.2f}</td>
                <td>{drawdown:.2f}%</td>
            </tr>\n"""

        html += """</table>
            </div>
        </body>
        </html>"""

        # Save to file
        with open(path, "w") as f:
            f.write(html)

    def plot_results(self, filename: Optional[str] = None):
        """Plot walk-forward analysis results.

        Args:
            filename: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Walk-forward analysis results not available. Run the analysis first.")

        # Create figure
        plt.figure(figsize=(12, 8))

        # Plot equity curves for each window
        for i, curve in enumerate(self.results["equity_curves"]):
            window = self.results["window_results"][i]
            label = f"Window {i + 1}: {window['test_start']} to {window['test_end']}"
            plt.plot(curve, label=label if i < 5 else "")  # Only show first 5 labels to avoid clutter

        # Plot mean equity curve if we have multiple windows
        if len(self.results["equity_curves"]) > 1:
            # Find shortest curve length
            min_length = min(len(curve) for curve in self.results["equity_curves"])
            # Extract same length from all curves
            aligned_curves = [curve[:min_length] for curve in self.results["equity_curves"]]
            # Calculate mean curve
            mean_curve = np.mean(aligned_curves, axis=0)
            plt.plot(mean_curve, "k--", linewidth=2, label="Mean Performance")

        plt.title("Walk-Forward Analysis: Equity Curves")
        plt.xlabel("Trading Days")
        plt.ylabel("Normalized Equity (Starting at 1.0)")
        plt.grid(True)
        plt.legend(loc="best")

        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
