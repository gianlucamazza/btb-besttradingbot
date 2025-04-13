# Installation Guide

This guide provides step-by-step instructions for installing and setting up the Best Trading Bot (BTB).

## Prerequisites

- Python 3.10 or higher
- Git
- pip (Python package manager)

## Basic Installation

1. Clone the repository:

```bash
git clone https://github.com/gianlucamazza/btb-besttradingbot.git
cd btb-besttradingbot
```

2. Create and activate a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Install the package in development mode:

```bash
pip install -e .
```

## Installation with Development Dependencies

If you plan to contribute to the project, you'll need development dependencies as well:

```bash
pip install -e ".[dev]"
```

This installs additional dependencies such as:
- pytest (for testing)
- black (for code formatting)
- ruff (for linting)
- mypy (for type checking)

## Setting Up API Access

To use the trading bot with real or test exchanges, you need to set up API credentials:

1. Create a `.env` file in the project root directory:

```bash
cp .env.example .env
```

2. Open the `.env` file and add your API credentials:

```
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
BYBIT_TESTNET=True  # Set to False for live trading
```

## Verifying Installation

To verify that everything is installed correctly, run:

```bash
# Check CLI commands
btb --help

# Run a simple test
pytest tests/
```

## Updating

To update to the latest version:

```bash
git pull
pip install -e .
```

## Troubleshooting

If you encounter any issues during installation:

- Make sure you have the correct Python version
- Check that all dependencies are properly installed
- Verify your API credentials if connecting to exchanges

For more detailed help, check the [Configuration Guide](CONFIGURATION.md) or open an issue on GitHub.
