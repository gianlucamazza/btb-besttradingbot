"""HTML Report generation module for backtest results."""

import base64
import io
import json
from datetime import datetime
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from jinja2 import Template


def generate_report(results: Dict, output_path: str):
    """Generate an HTML report from backtest results.

    Args:
        results: Backtest results dictionary
        output_path: Path to save the HTML report
    """
    # Create figures for the report
    figures = _create_figures(results)

    # Create report data
    report_data = _prepare_report_data(results, figures)

    # Generate HTML report
    html = _generate_html(report_data)

    # Save HTML report
    with open(output_path, "w") as f:
        f.write(html)


def _create_figures(results: Dict) -> Dict[str, str]:
    """Create figures for the report and convert to base64.

    Args:
        results: Backtest results dictionary

    Returns:
        Dictionary of base64-encoded figures
    """
    figures = {}

    # Equity curve figure
    fig_equity, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["equity_curve"])
    ax.set_title("Equity Curve")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True)
    figures["equity_curve"] = _fig_to_base64(fig_equity)
    plt.close(fig_equity)

    # Drawdown figure
    fig_drawdown, ax = plt.subplots(figsize=(10, 6))
    equity = np.array(results["equity_curve"])
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / running_max * 100
    ax.fill_between(range(len(drawdown)), drawdown, 0, color="red", alpha=0.3)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Trading Days")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True)
    figures["drawdown"] = _fig_to_base64(fig_drawdown)
    plt.close(fig_drawdown)

    # Profit distribution figure
    if results["trades"]:
        trades_df = pd.DataFrame(results["trades"])
        fig_profit, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(trades_df["profit_pct"], bins=30, kde=True, ax=ax)
        ax.axvline(x=0, color="red", linestyle="--")
        ax.set_title("Trade Profit Distribution (%)")
        ax.set_xlabel("Profit (%)")
        ax.set_ylabel("Frequency")
        ax.grid(True)
        figures["profit_distribution"] = _fig_to_base64(fig_profit)
        plt.close(fig_profit)

        # Monthly returns figure
        if "dates" in results["symbol_results"][list(results["symbol_results"].keys())[0]]:
            # Convert dates to datetime
            dates = [
                datetime.fromisoformat(d) if isinstance(d, str) else d
                for d in results["symbol_results"][list(results["symbol_results"].keys())[0]]["dates"]
            ]

            # Create equity curve with dates
            equity_df = pd.DataFrame({"equity": results["equity_curve"], "date": dates[: len(results["equity_curve"])]})
            equity_df.set_index("date", inplace=True)

            # Calculate monthly returns
            monthly_returns = equity_df["equity"].resample("ME").last().pct_change().dropna() * 100

            fig_monthly, ax = plt.subplots(figsize=(12, 6))
            monthly_returns.plot(kind="bar", ax=ax)
            ax.set_title("Monthly Returns (%)")
            ax.set_xlabel("Month")
            ax.set_ylabel("Return (%)")
            ax.grid(True, axis="y")
            figures["monthly_returns"] = _fig_to_base64(fig_monthly)
            plt.close(fig_monthly)

    return figures


def _prepare_report_data(results: Dict, figures: Dict[str, str]) -> Dict:
    """Prepare data for the report.

    Args:
        results: Backtest results dictionary
        figures: Dictionary of base64-encoded figures

    Returns:
        Dictionary of report data
    """
    # Extract metrics
    metrics = results["metrics"]

    # Create trades data
    trades_data = []
    if results["trades"]:
        for trade in results["trades"]:
            trades_data.append(
                {
                    "symbol": trade["symbol"],
                    "entry_time": trade["entry_time"],
                    "exit_time": trade["exit_time"],
                    "entry_price": f"${trade['entry_price']:.2f}",
                    "exit_price": f"${trade['exit_price']:.2f}",
                    "type": trade["type"],
                    "profit": f"${trade['profit']:.2f}",
                    "profit_pct": f"{trade['profit_pct']:.2f}%",
                    "exit_reason": trade["exit_reason"],
                }
            )

    # Create symbol results data
    symbol_results_data = []
    for key, data in results["symbol_results"].items():
        symbol_results_data.append(
            {
                "symbol": key,
                "final_capital": f"${data['final_capital']:.2f}",
                "return": f"{data['return'] * 100:.2f}%",
                "trades": len([t for t in results["trades"] if t["symbol"] == key.split("_")[0]]),
            }
        )

    # Create report data
    report_data = {
        "title": "Backtest Report",
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "figures": figures,
        "metrics": {
            "initial_capital": f"${metrics.get('initial_capital', 10000):.2f}",
            "final_capital": f"${metrics.get('final_capital', 0):.2f}",
            "total_return": f"{metrics.get('total_return_pct', 0):.2f}%",
            "annualized_return": f"{metrics.get('annualized_return_pct', 0):.2f}%",
            "sharpe_ratio": f"{metrics.get('sharpe_ratio', 0):.2f}",
            "sortino_ratio": f"{metrics.get('sortino_ratio', 0):.2f}",
            "max_drawdown": f"{metrics.get('max_drawdown_pct', 0):.2f}%",
            "win_rate": f"{metrics.get('win_rate', 0) * 100:.2f}%",
            "profit_factor": f"{metrics.get('profit_factor', 0):.2f}",
            "total_trades": len(results["trades"]),
        },
        "trades": trades_data[:100],  # Limit to first 100 trades for performance
        "symbols": symbol_results_data,
        "config": _pretty_print_config(results.get("config", {})),
    }

    return report_data


def _fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    buf.seek(0)
    img_str = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
    buf.close()
    return img_str


def _pretty_print_config(config: Dict) -> str:
    """Convert config dictionary to formatted JSON string."""
    return json.dumps(config, indent=2)


def _generate_html(data: Dict) -> str:
    """Generate HTML report from template and data.

    Args:
        data: Report data dictionary

    Returns:
        HTML report string
    """
    # HTML template
    template_str = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{{ title }}</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                line-height: 1.6;
                color: #333;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .header {
                text-align: center;
                margin-bottom: 30px;
                border-bottom: 1px solid #eee;
                padding-bottom: 20px;
            }
            .section {
                margin-bottom: 40px;
            }
            .metrics-grid {
                display: grid;
                grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                gap: 20px;
                margin-bottom: 20px;
            }
            .metric-card {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 15px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            }
            .metric-value {
                font-size: 24px;
                font-weight: bold;
                color: #2980b9;
            }
            .metric-name {
                font-size: 14px;
                color: #7f8c8d;
                margin-bottom: 5px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
                margin-bottom: 20px;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
            .figure {
                margin-bottom: 30px;
                text-align: center;
            }
            .figure img {
                max-width: 100%;
                height: auto;
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            .config {
                background-color: #f9f9f9;
                border-radius: 5px;
                padding: 15px;
                overflow-x: auto;
                font-family: monospace;
                white-space: pre-wrap;
            }
            .positive {
                color: green;
            }
            .negative {
                color: red;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>{{ title }}</h1>
            <p>Generated on {{ date }}</p>
        </div>

        <div class="section">
            <h2>Performance Summary</h2>
            <div class="metrics-grid">
                {% for name, value in metrics.items() %}
                <div class="metric-card">
                    <div class="metric-name">{{ name|replace('_', ' ')|title }}</div>
                    <div class="metric-value">{{ value }}</div>
                </div>
                {% endfor %}
            </div>
        </div>

        <div class="section">
            <h2>Performance Charts</h2>

            {% if figures.equity_curve %}
            <div class="figure">
                <h3>Equity Curve</h3>
                <img src="{{ figures.equity_curve }}" alt="Equity Curve">
            </div>
            {% endif %}

            {% if figures.drawdown %}
            <div class="figure">
                <h3>Drawdown</h3>
                <img src="{{ figures.drawdown }}" alt="Drawdown">
            </div>
            {% endif %}

            {% if figures.profit_distribution %}
            <div class="figure">
                <h3>Profit Distribution</h3>
                <img src="{{ figures.profit_distribution }}" alt="Profit Distribution">
            </div>
            {% endif %}

            {% if figures.monthly_returns %}
            <div class="figure">
                <h3>Monthly Returns</h3>
                <img src="{{ figures.monthly_returns }}" alt="Monthly Returns">
            </div>
            {% endif %}
        </div>

        {% if symbols %}
        <div class="section">
            <h2>Symbol Performance</h2>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Final Capital</th>
                        <th>Return</th>
                        <th>Trades</th>
                    </tr>
                </thead>
                <tbody>
                    {% for symbol in symbols %}
                    <tr>
                        <td>{{ symbol.symbol }}</td>
                        <td>{{ symbol.final_capital }}</td>
                        <td>{{ symbol.return }}</td>
                        <td>{{ symbol.trades }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        {% if trades %}
        <div class="section">
            <h2>Trade History</h2>
            <p>Showing first {{ trades|length }} trades</p>
            <table>
                <thead>
                    <tr>
                        <th>Symbol</th>
                        <th>Type</th>
                        <th>Entry Time</th>
                        <th>Exit Time</th>
                        <th>Entry Price</th>
                        <th>Exit Price</th>
                        <th>Profit</th>
                        <th>Profit %</th>
                        <th>Exit Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {% for trade in trades %}
                    <tr>
                        <td>{{ trade.symbol }}</td>
                        <td>{{ trade.type }}</td>
                        <td>{{ trade.entry_time }}</td>
                        <td>{{ trade.exit_time }}</td>
                        <td>{{ trade.entry_price }}</td>
                        <td>{{ trade.exit_price }}</td>
                        <td class="{{ 'positive' if trade.profit.startswith('$') and not trade.profit.startswith('-') else 'negative' }}">{{ trade.profit }}</td>
                        <td class="{{ 'positive' if not trade.profit_pct.startswith('-') else 'negative' }}">{{ trade.profit_pct }}</td>
                        <td>{{ trade.exit_reason }}</td>
                    </tr>
                    {% endfor %}
                </tbody>
            </table>
        </div>
        {% endif %}

        <div class="section">
            <h2>Configuration</h2>
            <div class="config">{{ config }}</div>
        </div>
    </body>
    </html>
    """

    # Render template
    template = Template(template_str)
    html = template.render(**data)

    return html
