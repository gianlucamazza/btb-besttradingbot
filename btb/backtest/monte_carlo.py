"""Monte Carlo simulation for backtesting."""

from typing import Dict, List, Optional

import numpy as np
from matplotlib import pyplot as plt


class MonteCarloSimulator:
    """Monte Carlo simulator for trading strategies."""

    def __init__(self, backtest_results: Dict, num_simulations: int = 1000, random_seed: Optional[int] = None):
        """Initialize Monte Carlo simulator.

        Args:
            backtest_results: Results from Backtester.run()
            num_simulations: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility
        """
        self.backtest_results = backtest_results
        self.num_simulations = num_simulations
        self.random_seed = random_seed
        self.results = None

    def run(self) -> Dict:
        """Run Monte Carlo simulation.

        Returns:
            Dictionary with simulation results
        """
        # Set random seed if provided
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        # Extract trades from backtest results
        trades = self.backtest_results["trades"]
        if not trades:
            raise ValueError("No trades found in backtest results.")

        # Extract trade returns as percentages
        trade_returns = [trade["profit_pct"] / 100 for trade in trades]  # Convert to decimal

        # Run simulations
        initial_capital = self.backtest_results["equity_curve"][0]
        simulation_results = []

        for _ in range(self.num_simulations):
            # Shuffle trade returns
            shuffled_returns = np.random.choice(trade_returns, len(trade_returns))

            # Calculate equity curve
            equity = [initial_capital]
            current_capital = initial_capital

            for trade_return in shuffled_returns:
                # Apply trade return to current capital
                position_size = 0.1  # Assume 10% position size
                trade_profit = current_capital * position_size * trade_return
                current_capital += trade_profit
                equity.append(current_capital)

            simulation_results.append(
                {
                    "equity_curve": equity,
                    "final_capital": equity[-1],
                    "total_return": (equity[-1] - initial_capital) / initial_capital,
                    "max_drawdown": self._calculate_max_drawdown(equity),
                }
            )

        # Calculate statistics across all simulations
        final_capitals = [sim["final_capital"] for sim in simulation_results]
        total_returns = [sim["total_return"] for sim in simulation_results]
        max_drawdowns = [sim["max_drawdown"] for sim in simulation_results]

        stats = {
            "mean_final_capital": np.mean(final_capitals),
            "median_final_capital": np.median(final_capitals),
            "std_final_capital": np.std(final_capitals),
            "min_final_capital": min(final_capitals),
            "max_final_capital": max(final_capitals),
            "mean_return": np.mean(total_returns),
            "median_return": np.median(total_returns),
            "std_return": np.std(total_returns),
            "min_return": min(total_returns),
            "max_return": max(total_returns),
            "mean_max_drawdown": np.mean(max_drawdowns),
            "median_max_drawdown": np.median(max_drawdowns),
            "min_max_drawdown": min(max_drawdowns),
            "max_max_drawdown": max(max_drawdowns),
        }

        # Calculate percentiles
        percentiles = [5, 25, 50, 75, 95]
        for p in percentiles:
            stats[f"capital_{p}th_percentile"] = np.percentile(final_capitals, p)
            stats[f"return_{p}th_percentile"] = np.percentile(total_returns, p)
            stats[f"drawdown_{p}th_percentile"] = np.percentile(max_drawdowns, p)

        # Store results
        self.results = {
            "simulations": simulation_results,
            "stats": stats,
            "original_backtest": {
                "final_capital": self.backtest_results["equity_curve"][-1],
                "total_return": (self.backtest_results["equity_curve"][-1] - initial_capital) / initial_capital,
                "max_drawdown": self.backtest_results["metrics"].get("max_drawdown", 0),
            },
        }

        return self.results

    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown from equity curve."""
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - running_max) / running_max
        return abs(min(drawdown)) if len(drawdown) > 0 else 0

    def plot_results(self, filename: Optional[str] = None):
        """Plot Monte Carlo simulation results.

        Args:
            filename: Optional path to save the plot
        """
        if self.results is None:
            raise ValueError("Simulation results not available. Run the simulation first.")

        # Create subplots
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), gridspec_kw={"height_ratios": [3, 1, 1]})

        # Plot equity curves
        original_equity = self.backtest_results["equity_curve"]
        initial_capital = original_equity[0]

        # Plot random sample of simulation equity curves (50 max to avoid clutter)
        sample_size = min(50, len(self.results["simulations"]))
        sample_indices = np.random.choice(len(self.results["simulations"]), sample_size, replace=False)

        for idx in sample_indices:
            sim = self.results["simulations"][idx]
            ax1.plot(sim["equity_curve"], "b-", alpha=0.1)

        # Plot percentile curves if we have enough simulations
        if len(self.results["simulations"]) >= 20:
            # Find shortest equity curve length
            min_length = min(len(sim["equity_curve"]) for sim in self.results["simulations"])

            # Extract all equity curves with the same length
            all_curves = np.array([sim["equity_curve"][:min_length] for sim in self.results["simulations"]])

            # Calculate percentiles at each time step
            percentiles = [5, 25, 50, 75, 95]
            percentile_curves = {p: np.percentile(all_curves, p, axis=0) for p in percentiles}

            # Plot percentile curves
            ax1.plot(percentile_curves[50], "g-", linewidth=2, label="Median (50th percentile)")
            ax1.plot(percentile_curves[5], "r-", linewidth=1.5, label="5th percentile")
            ax1.plot(percentile_curves[95], "b-", linewidth=1.5, label="95th percentile")
            ax1.fill_between(
                range(min_length),
                percentile_curves[25],
                percentile_curves[75],
                alpha=0.2,
                color="gray",
                label="25th-75th percentile range",
            )

        # Plot original equity curve
        ax1.plot(original_equity, "k-", linewidth=2, label="Original Backtest")

        ax1.set_title("Monte Carlo Simulation: Equity Curves")
        ax1.set_ylabel("Portfolio Value")
        ax1.grid(True)
        ax1.legend(loc="best")

        # Plot return distribution
        total_returns = [sim["total_return"] * 100 for sim in self.results["simulations"]]  # Convert to percentages
        original_return = (self.backtest_results["equity_curve"][-1] - initial_capital) / initial_capital * 100

        ax2.hist(total_returns, bins=50, alpha=0.75, color="skyblue")
        ax2.axvline(
            original_return, color="r", linestyle="dashed", linewidth=2, label=f"Original: {original_return:.2f}%"
        )
        ax2.axvline(
            self.results["stats"]["mean_return"] * 100,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {self.results['stats']['mean_return'] * 100:.2f}%",
        )
        ax2.set_title("Distribution of Total Returns")
        ax2.set_xlabel("Total Return (%)")
        ax2.set_ylabel("Frequency")
        ax2.grid(True)
        ax2.legend(loc="best")

        # Plot drawdown distribution
        max_drawdowns = [sim["max_drawdown"] * 100 for sim in self.results["simulations"]]  # Convert to percentages
        original_drawdown = self.backtest_results["metrics"].get("max_drawdown", 0) * 100

        ax3.hist(max_drawdowns, bins=50, alpha=0.75, color="salmon")
        ax3.axvline(
            original_drawdown, color="r", linestyle="dashed", linewidth=2, label=f"Original: {original_drawdown:.2f}%"
        )
        ax3.axvline(
            self.results["stats"]["mean_max_drawdown"] * 100,
            color="g",
            linestyle="dashed",
            linewidth=2,
            label=f"Mean: {self.results['stats']['mean_max_drawdown'] * 100:.2f}%",
        )
        ax3.set_title("Distribution of Maximum Drawdowns")
        ax3.set_xlabel("Maximum Drawdown (%)")
        ax3.set_ylabel("Frequency")
        ax3.grid(True)
        ax3.legend(loc="best")

        plt.tight_layout()

        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

    def generate_report(self, path: str):
        """Generate an HTML report with Monte Carlo results.

        Args:
            path: Path to save the report
        """
        if self.results is None:
            raise ValueError("Simulation results not available. Run the simulation first.")

        # Implementation for generating HTML report
        # This would typically use a template engine like Jinja2
        # For simplicity, we'll just create a basic HTML report

        html = """<!DOCTYPE html>
        <html>
        <head>
            <title>Monte Carlo Simulation Report</title>
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
                <h1>Monte Carlo Simulation Report</h1>
                <p>Number of simulations: {self.num_simulations}</p>
            </div>

            <div class="section">
                <h2>Summary Statistics</h2>
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Original Backtest</th>
                        <th>Mean</th>
                        <th>Median</th>
                        <th>Std Dev</th>
                        <th>Min</th>
                        <th>Max</th>
                        <th>5th Percentile</th>
                        <th>95th Percentile</th>
                    </tr>
        """

        # Add return statistics
        original_return = self.results["original_backtest"]["total_return"] * 100
        mean_return = self.results["stats"]["mean_return"] * 100
        median_return = self.results["stats"]["median_return"] * 100
        std_return = self.results["stats"]["std_return"] * 100
        min_return = self.results["stats"]["min_return"] * 100
        max_return = self.results["stats"]["max_return"] * 100
        return_5th = self.results["stats"]["return_5th_percentile"] * 100
        return_95th = self.results["stats"]["return_95th_percentile"] * 100

        html += f"""<tr>
            <td>Total Return (%)</td>
            <td>{original_return:.2f}%</td>
            <td>{mean_return:.2f}%</td>
            <td>{median_return:.2f}%</td>
            <td>{std_return:.2f}%</td>
            <td>{min_return:.2f}%</td>
            <td>{max_return:.2f}%</td>
            <td>{return_5th:.2f}%</td>
            <td>{return_95th:.2f}%</td>
        </tr>\n"""

        # Add drawdown statistics
        original_dd = self.results["original_backtest"]["max_drawdown"] * 100
        mean_dd = self.results["stats"]["mean_max_drawdown"] * 100
        median_dd = self.results["stats"]["median_max_drawdown"] * 100
        min_dd = self.results["stats"]["min_max_drawdown"] * 100
        max_dd = self.results["stats"]["max_max_drawdown"] * 100
        dd_5th = self.results["stats"]["drawdown_5th_percentile"] * 100
        dd_95th = self.results["stats"]["drawdown_95th_percentile"] * 100

        html += f"""<tr>
            <td>Maximum Drawdown (%)</td>
            <td>{original_dd:.2f}%</td>
            <td>{mean_dd:.2f}%</td>
            <td>{median_dd:.2f}%</td>
            <td>N/A</td>
            <td>{min_dd:.2f}%</td>
            <td>{max_dd:.2f}%</td>
            <td>{dd_5th:.2f}%</td>
            <td>{dd_95th:.2f}%</td>
        </tr>\n"""

        # Add final capital statistics
        original_capital = self.results["original_backtest"]["final_capital"]
        mean_capital = self.results["stats"]["mean_final_capital"]
        median_capital = self.results["stats"]["median_final_capital"]
        std_capital = self.results["stats"]["std_final_capital"]
        min_capital = self.results["stats"]["min_final_capital"]
        max_capital = self.results["stats"]["max_final_capital"]
        capital_5th = self.results["stats"]["capital_5th_percentile"]
        capital_95th = self.results["stats"]["capital_95th_percentile"]

        html += f"""<tr>
            <td>Final Capital</td>
            <td>${original_capital:.2f}</td>
            <td>${mean_capital:.2f}</td>
            <td>${median_capital:.2f}</td>
            <td>${std_capital:.2f}</td>
            <td>${min_capital:.2f}</td>
            <td>${max_capital:.2f}</td>
            <td>${capital_5th:.2f}</td>
            <td>${capital_95th:.2f}</td>
        </tr>\n"""

        html += """</table>
            </div>

            <div class="section">
                <h2>Interpretation</h2>
                <p>
                    The Monte Carlo simulation results provide an estimate of the range of possible outcomes for the trading strategy.
                    By shuffling the order of trades, we can assess how robust the strategy is to the sequencing of trades.
                </p>
                <p>
                    <strong>Key findings:</strong>
                </p>
                <ul>
        """

        # Add interpretation based on results
        html += f"<li>There is a 90% probability that the total return will be between {return_5th:.2f}% and {return_95th:.2f}%.</li>\n"
        html += f"<li>The maximum drawdown is expected to be at least {dd_5th:.2f}% and could be as high as {dd_95th:.2f}% in 90% of scenarios.</li>\n"

        # Add risk assessment
        if return_5th < 0:
            html += "<li>CAUTION: There is a more than 5% chance of losing money with this strategy.</li>\n"
        if mean_return < original_return:
            html += "<li>The original backtest performed better than the average simulation, suggesting possible luck or selection bias.</li>\n"
        else:
            html += "<li>The original backtest performed similarly to the average simulation, suggesting the results are robust.</li>\n"

        html += """</ul>
            </div>
        </body>
        </html>"""

        # Save to file
        with open(path, "w") as f:
            f.write(html)
