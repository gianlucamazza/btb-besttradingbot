"""Command-line interface for the BestTradingBot."""

import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Optional, Tuple, Union

import click

from btb.backtest.engine import Backtester
from btb.models.lstm import LSTMModel
from btb.models.transformer import TransformerModel
from btb.run.trader import Trader
from btb.utils.config import (
    BACKTEST_CONFIG_SCHEMA,
    MODEL_CONFIG_SCHEMA,
    TRADING_CONFIG_SCHEMA,
    load_config,
    validate_config,
)
from btb.utils.logging import get_log_file_path, setup_logger


@click.group()
def cli():
    """BestTradingBot command line interface."""


@cli.command()
@click.option("--config", "-c", required=True, help="Path to configuration file")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", "-l", help="Path to log file")
def run(config: str, verbose: bool, log_file: Optional[str]):
    """Run the trading bot with the specified configuration."""
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_file or get_log_file_path()
    logger = setup_logger("btb", log_level, log_file)

    # Load configuration
    try:
        logger.info(f"Loading configuration from {config}")
        cfg = load_config(config)
        validate_config(cfg, TRADING_CONFIG_SCHEMA)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Run trader
    try:
        logger.info("Starting trading bot")
        trader = Trader(cfg)
        trader.start()
    except KeyboardInterrupt:
        logger.info("Trading bot stopped by user")
        trader.stop()
    except Exception as e:
        logger.error(f"Error running trading bot: {e}")
        sys.exit(1)


@cli.command()
@click.option("--config", "-c", required=True, help="Path to backtest configuration")
@click.option("--output", "-o", default=None, help="Path to save backtest results")
@click.option("--report", "-r", default=None, help="Path to save backtest report")
@click.option("--plot", "-p", is_flag=True, help="Show backtest results plot")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", "-l", help="Path to log file")
def backtest(
    config: str,
    output: Optional[str],
    report: Optional[str],
    plot: bool,
    verbose: bool,
    log_file: Optional[str],
):
    """Run backtest with the specified configuration."""
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_file or get_log_file_path(prefix="backtest_")
    logger = setup_logger("btb.backtest", log_level, log_file)

    # Load configuration
    try:
        logger.info(f"Loading configuration from {config}")
        cfg = load_config(config)
        validate_config(cfg, BACKTEST_CONFIG_SCHEMA)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Run backtest
    try:
        logger.info("Starting backtest")
        logger.debug(f"Config: {cfg}")
        backtester = Backtester(cfg)
        logger.debug("Backtester initialized")
        backtester.run()
        logger.debug("Backtest run completed")

        # Calculate metrics
        metrics = backtester.calculate_metrics()

        # Display summary
        logger.info("Backtest completed successfully")
        logger.info(f"Initial capital: ${metrics['initial_capital']:.2f}")
        logger.info(f"Final capital: ${metrics['final_capital']:.2f}")
        logger.info(f"Total return: {metrics['total_return_pct']:.2f}%")
        if "sharpe_ratio" in metrics:
            logger.info(f"Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        if "max_drawdown_pct" in metrics:
            logger.info(f"Maximum drawdown: {metrics['max_drawdown_pct']:.2f}%")
        if "win_rate" in metrics:
            logger.info(f"Win rate: {metrics['win_rate'] * 100:.2f}%")

        # Save results if output path is specified
        if output:
            logger.info(f"Saving results to {output}")
            backtester.save_results(output)

        # Generate report if requested
        if report:
            logger.info(f"Generating report to {report}")
            backtester.generate_report(report)

        # Show plot if requested
        if plot:
            logger.info("Displaying backtest results plot")
            backtester.plot_results()

    except Exception as e:
        logger.error(f"Error running backtest: {e}")
        sys.exit(1)


@cli.command()
@click.option("--data", "-d", required=True, help="Path to training data")
@click.option("--model", "-m", required=True, help="Model type to train")
@click.option("--config", "-c", default=None, help="Path to model configuration")
@click.option("--output", "-o", default="models/", help="Directory to save trained model")
@click.option("--epochs", "-e", type=int, default=None, help="Number of training epochs")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", "-l", help="Path to log file")
def train(
    data: str,
    model: str,
    config: Optional[str],
    output: str,
    epochs: Optional[int],
    verbose: bool,
    log_file: Optional[str],
):
    """Train a model with the specified configuration."""
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_file or get_log_file_path(prefix="train_")
    logger = setup_logger("btb.train", log_level, log_file)

    # Load configuration if provided
    cfg = {}
    if config:
        try:
            logger.info(f"Loading configuration from {config}")
            cfg = load_config(config)
            validate_config(cfg, MODEL_CONFIG_SCHEMA)
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            sys.exit(1)

    # Override epochs if specified
    if epochs is not None and "training" in cfg:
        cfg["training"]["epochs"] = epochs

    # Create output directory if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)

    # Extract asset symbol and timeframe from data path
    data_filename = os.path.basename(data)
    if "_processed" in data_filename:
        # Extract symbol and timeframe from filename (e.g., "BTCUSDT_1h_processed.csv")
        parts = data_filename.split("_")
        if len(parts) >= 3:
            symbol = parts[0]
            timeframe = parts[1]
            model_filename = f"{model}_{symbol}_{timeframe}_model.pth"
        else:
            model_filename = f"{model}_model.pth"
    else:
        model_filename = f"{model}_model.pth"

    model_path = os.path.join(output, model_filename)
    logger.info(f"Model will be saved as: {model_path}")

    # Train model
    try:
        logger.info(f"Loading data from {data}")

        # Import necessary modules - use 'type: ignore' to suppress mypy warnings about missing stubs
        import numpy as np  # type: ignore
        import pandas as pd  # type: ignore
        import torch

        # Models already imported at the top level
        # Load data
        logger.info("Loading and processing data...")
        df = pd.read_csv(data)

        # If there's an unnamed index column, drop it
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        df_clean = df.dropna()
        logger.info(f"Data loaded, shape: {df_clean.shape}")

        # Extract features and target
        features = df_clean.drop(["target"], axis=1).values
        target = df_clean["target"].values

        # Get sequence length from config
        seq_length = 60  # Default
        if cfg and "features" in cfg and "sequence_length" in cfg["features"]:
            seq_length = cfg["features"]["sequence_length"]

        # Create sequences
        logger.info(f"Creating sequences with length {seq_length}...")

        # Function to create sequences
        def create_sequences(data, seq_length):
            xs, ys = [], []
            for i in range(len(data) - seq_length):
                x = data[i : (i + seq_length)]
                y = data[i + seq_length, -1]  # Target is the last column
                xs.append(x)
                ys.append(y)
            return np.array(xs), np.array(ys)

        X, y = create_sequences(np.column_stack((features, target)), seq_length)
        logger.info(f"Created sequences: X shape: {X.shape}, y shape: {y.shape}")

        # Split into train, validation, test sets
        train_ratio = 0.7
        val_ratio = 0.15
        train_size = int(len(X) * train_ratio)
        val_size = int(len(X) * val_ratio)

        X_train, y_train = X[:train_size], y[:train_size]
        X_val, y_val = (
            X[train_size : train_size + val_size],
            y[train_size : train_size + val_size],
        )

        # Convert to PyTorch tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val).view(-1, 1)

        # The last column in each sequence is the target, so we need to use (:, :, :-1)
        # to get just the features for the model input
        X_train_features = X_train_tensor[:, :, :-1]
        X_val_features = X_val_tensor[:, :, :-1]

        # Configure model
        logger.info(f"Training {model} model")

        # Set default parameters if not in config
        hidden_dim = 128
        num_layers = 2
        dropout = 0.2
        learning_rate = 0.001
        batch_size = 64
        training_epochs = 50
        optimizer_name = "adam"
        scheduler_name = "cosine"

        # Update from config if available
        if cfg:
            if "model" in cfg:
                hidden_dim = cfg["model"].get("hidden_dim", hidden_dim)
                num_layers = cfg["model"].get("num_layers", num_layers)
                dropout = cfg["model"].get("dropout", dropout)
            if "training" in cfg:
                learning_rate = cfg["training"].get("learning_rate", learning_rate)
                training_epochs = cfg["training"].get("epochs", training_epochs)
                batch_size = cfg["training"].get("batch_size", batch_size)
                optimizer_name = cfg["training"].get("optimizer", optimizer_name)
                scheduler_name = cfg["training"].get("scheduler", scheduler_name)

        # Override epochs if specified on command line
        if epochs:
            training_epochs = epochs

        # Initialize model based on type
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        # Declare variable to hold the model instance with proper typing
        model_instance: Union[LSTMModel, TransformerModel]

        if model == "lstm":
            # Configure LSTM model
            model_config = {
                "input_dim": X_train_features.shape[2],
                "hidden_dim": hidden_dim,
                "num_layers": num_layers,
                "output_dim": 1,
                "dropout": dropout,
                "learning_rate": float(learning_rate),
                "batch_size": batch_size,
                "epochs": training_epochs,
                "optimizer": optimizer_name,
                "scheduler": scheduler_name,
            }

            # Initialize model
            model_instance = LSTMModel(model_config)

        elif model == "transformer":
            # Configure Transformer model
            attention_heads = 8
            if cfg and "model" in cfg and "attention_heads" in cfg["model"]:
                attention_heads = cfg["model"]["attention_heads"]

            model_config = {
                "input_dim": X_train_features.shape[2],
                "d_model": hidden_dim,
                "nhead": attention_heads,
                "num_encoder_layers": num_layers,
                "dim_feedforward": hidden_dim * 4,
                "dropout": dropout,
                "output_dim": 1,
                "learning_rate": float(learning_rate),
                "batch_size": batch_size,
                "epochs": training_epochs,
            }

            # Initialize model
            model_instance = TransformerModel(model_config)
        else:
            raise ValueError(f"Unsupported model type: {model}")

        # Train model
        logger.info(f"Training model for {training_epochs} epochs...")
        train_metrics = model_instance.train((X_train_features, y_train_tensor), (X_val_features, y_val_tensor))

        # Log training results
        final_train_loss = train_metrics["final_train_loss"]
        final_val_loss = train_metrics.get("final_val_loss", "N/A")
        logger.info(f"Training complete. Final train loss: {final_train_loss:.6f}, Validation loss: {final_val_loss}")

        # Save model
        model_instance.save(model_path)
        logger.info(f"Model saved to {model_path}")

    except Exception as e:
        logger.error(f"Error training model: {e}")
        sys.exit(1)


@cli.command()
@click.option("--data-dir", "-d", default="data/processed/", help="Directory containing processed data files")
@click.option("--model", "-m", required=True, help="Model type to train (lstm or transformer)")
@click.option("--config", "-c", default="config/model_config.yaml", help="Path to model configuration")
@click.option("--output", "-o", default="models/", help="Directory to save trained models")
@click.option("--symbols", "-s", multiple=True, help="Specific symbols to train (default: all available)")
@click.option("--timeframes", "-t", multiple=True, help="Specific timeframes to train (default: all available)")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", "-l", help="Path to log file")
@click.option("--parallel", "-p", is_flag=True, help="Train models in parallel using multiple processes")
@click.option(
    "--max-workers",
    "-w",
    type=int,
    default=None,
    help="Maximum number of worker processes for parallel training (default: number of CPU cores)",
)
def train_all(
    data_dir: str,
    model: str,
    config: str,
    output: str,
    symbols: Tuple[str],
    timeframes: Tuple[str],
    verbose: bool,
    log_file: Optional[str],
    parallel: bool,
    max_workers: Optional[int],
):
    """Train models for all available assets and timeframes."""
    import multiprocessing
    import time
    from concurrent.futures import ProcessPoolExecutor

    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_file or get_log_file_path(prefix="train_batch_")
    logger = setup_logger("btb.train_batch", log_level, log_file)

    # Check model type
    if model.lower() not in ["lstm", "transformer"]:
        logger.error(f"Unsupported model type: {model}. Use 'lstm' or 'transformer'")
        sys.exit(1)

    # Load configuration
    try:
        logger.info(f"Loading configuration from {config}")
        cfg = load_config(config)
        validate_config(cfg, MODEL_CONFIG_SCHEMA)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Create output directory if it doesn't exist
    if not os.path.exists(output):
        os.makedirs(output)

    # Find all processed data files
    data_files = []
    for file in os.listdir(data_dir):
        if file.endswith("_processed.csv"):
            parts = file.split("_")
            if len(parts) >= 3:
                symbol = parts[0]
                timeframe = parts[1]

                # Filter by symbols and timeframes if specified
                if symbols and symbol not in symbols:
                    continue
                if timeframes and timeframe not in timeframes:
                    continue

                data_files.append((file, symbol, timeframe, data_dir, output, model, cfg, verbose))

    if not data_files:
        logger.error(f"No suitable data files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(data_files)} data files to process")

    # Set the number of worker processes
    if max_workers is None:
        max_workers = min(multiprocessing.cpu_count(), len(data_files))

    start_time = time.time()

    if parallel:
        logger.info(f"Training models in parallel using {max_workers} worker processes")

        # Train models in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all training tasks to the executor
            futures = [executor.submit(train_single_model, *params) for params in data_files]

            # Process results as they complete
            for i, future in enumerate(futures):
                try:
                    result = future.result()
                    logger.info(f"Completed training ({i + 1}/{len(data_files)}): {result}")
                except Exception as e:
                    logger.error(f"Error in parallel training: {e}")
    else:
        logger.info("Training models sequentially")
        # Train models sequentially
        for i, params in enumerate(data_files):
            try:
                result = train_single_model(*params)
                logger.info(f"Completed training ({i + 1}/{len(data_files)}): {result}")
            except Exception as e:
                logger.error(f"Error training model: {e}")
                continue

    end_time = time.time()
    logger.info(f"All model training completed in {end_time - start_time:.2f} seconds")


def train_single_model(file, symbol, timeframe, data_dir, output, model_type, cfg, verbose):
    """Train a single model with the given parameters."""
    import time
    from datetime import datetime

    import numpy as np
    import pandas as pd
    import torch

    # Set up a separate logger for this process
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    process_id = os.getpid()
    log_file = f"logs/train_{symbol}_{timeframe}_{timestamp}_{process_id}.log"
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger(f"btb.train.{symbol}_{timeframe}", log_level, log_file)

    # Prepare paths
    data_path = os.path.join(data_dir, file)
    model_filename = f"{model_type}_{symbol}_{timeframe}_model.pth"
    model_path = os.path.join(output, model_filename)

    start_time = time.time()

    logger.info(f"Training {model_type} model for {symbol} {timeframe}")
    logger.info(f"Using data from {data_path}")
    logger.info(f"Model will be saved as {model_path}")

    try:
        # Load data
        df = pd.read_csv(data_path)

        # If there's an unnamed index column, drop it
        if "Unnamed: 0" in df.columns:
            df = df.drop("Unnamed: 0", axis=1)

        df_clean = df.dropna()
        logger.info(f"Data loaded, shape: {df_clean.shape}")

        # Extract features and target
        features = df_clean.drop(["target"], axis=1).values
        target = df_clean["target"].values

        # Create sequences
        def create_sequences(data, seq_length):
            sequences = []
            targets = []
            for i in range(len(data) - seq_length):
                seq = data[i : i + seq_length]
                target_val = target[i + seq_length]
                sequences.append(seq)
                targets.append(target_val)
            return np.array(sequences), np.array(targets)

        # Use sequence length from config or default to 60
        seq_length = cfg.get("features", {}).get("sequence_length", 60)
        X, y = create_sequences(features, seq_length)
        logger.info(f"Created {len(X)} sequences of length {seq_length}")

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y.reshape(-1, 1), dtype=torch.float32)

        # Split data into train and validation sets
        train_size = int(0.8 * len(X_tensor))
        X_train, X_val = X_tensor[:train_size], X_tensor[train_size:]
        y_train, y_val = y_tensor[:train_size], y_tensor[train_size:]

        logger.info(f"Training set: {X_train.shape}, Validation set: {X_val.shape}")

        # Prepare model config
        model_config = {
            "input_dim": X.shape[2],
            "hidden_dim": int(cfg.get("model", {}).get("hidden_dim", 128)),
            "output_dim": 1,
            "num_layers": int(cfg.get("model", {}).get("num_layers", 2)),
            "dropout": float(cfg.get("model", {}).get("dropout", 0.1)),
            "optimizer": str(cfg.get("training", {}).get("optimizer", "adam")),
            "learning_rate": float(cfg.get("training", {}).get("learning_rate", 0.001)),
            "epochs": int(cfg.get("training", {}).get("epochs", 100)),
            "early_stopping": bool(cfg.get("training", {}).get("early_stopping", True)),
            "patience": int(cfg.get("training", {}).get("patience", 15)),
            "scheduler": str(cfg.get("training", {}).get("scheduler", "cosine")),
            "weight_decay": float(cfg.get("training", {}).get("weight_decay", 1e-5)),
            "gradient_clipping": float(cfg.get("training", {}).get("gradient_clipping", 1.0)),
        }

        # Initialize and train model
        # Declare variable to hold the model instance with proper typing
        model_instance: Union[LSTMModel, TransformerModel]

        if model_type.lower() == "lstm":
            model_instance = LSTMModel(model_config)
        else:  # transformer
            model_instance = TransformerModel(model_config)

        logger.info("Starting model training...")
        train_metrics = model_instance.train(
            (X_train, y_train),
            validation_data=(X_val, y_val),
        )

        # Log training results
        logger.info("Training completed")
        logger.info(f"Final training loss: {train_metrics['final_train_loss']:.6f}")
        if "final_val_loss" in train_metrics:
            logger.info(f"Final validation loss: {train_metrics['final_val_loss']:.6f}")
        if "best_val_loss" in train_metrics:
            logger.info(f"Best validation loss: {train_metrics['best_val_loss']:.6f}")

        # Save model
        model_instance.save(model_path)
        logger.info(f"Model saved to {model_path}")

        end_time = time.time()
        logger.info(f"Training completed in {end_time - start_time:.2f} seconds")

        return f"{symbol}_{timeframe} ({os.path.basename(model_path)})"

    except Exception as e:
        logger.error(f"Error training model for {symbol} {timeframe}: {str(e)}", exc_info=True)
        raise e


@cli.command()
@click.option(
    "--config",
    "-c",
    default="config/backtest_config.yaml",
    help="Path to backtest configuration for data parameters",
)
@click.option(
    "--output-dir",
    "-o",
    default="data/processed",
    help="Directory to save processed data",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
@click.option("--log-file", "-l", help="Path to log file")
def prepare_data(config: str, output_dir: str, verbose: bool, log_file: Optional[str]):
    """Prepare data for model training."""
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file = log_file or get_log_file_path(prefix="data_prep_")
    logger = setup_logger("btb.data", log_level, log_file)

    # Load configuration
    try:
        logger.info(f"Loading configuration from {config}")
        cfg = load_config(config)
        validate_config(cfg, BACKTEST_CONFIG_SCHEMA)
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        sys.exit(1)

    # Process data
    try:
        logger.info("Starting data preparation...")

        # Import necessary modules
        from btb.data.loader import DataLoader
        from btb.data.preprocessing import DataPreprocessor

        # Initialize data loader
        data_loader = DataLoader({"use_dummy": True})

        # Load historical market data
        start_date = cfg["backtest"]["start_date"]
        end_date = cfg["backtest"]["end_date"]
        symbols = cfg["backtest"]["symbols"]
        timeframes = cfg["backtest"]["timeframes"]

        logger.info(f"Loading data for {symbols} from {start_date} to {end_date}")

        # Load data
        data = data_loader.load_data(
            symbols=symbols,
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
        )

        # Initialize data preprocessor
        preprocessor = DataPreprocessor()

        # Process each symbol/timeframe
        for key, df in data.items():
            symbol, timeframe = key.split("_")
            logger.info(f"Processing {symbol} {timeframe} data...")

            # Add technical indicators
            df_with_features = preprocessor.add_technical_indicators(df)

            # Add target variable (next day's return)
            df_with_features["target"] = df_with_features["close"].pct_change(1).shift(-1)

            # Drop rows with NaN values
            df_clean = df_with_features.dropna()

            # Select features (exclude price data and target)
            feature_cols = [
                col
                for col in df_clean.columns
                if col
                not in [
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                    "target",
                    "daily_return",
                ]
            ]

            # Normalize data for ML models
            df_normalized = preprocessor._normalize_data(df_clean[feature_cols + ["close"]])[0]
            df_normalized["target"] = df_clean["target"]

            # Save processed data
            os.makedirs(output_dir, exist_ok=True)
            processed_data_path = f"{output_dir}/{symbol}_{timeframe}_processed.csv"
            df_normalized.to_csv(processed_data_path)
            logger.info(f"Saved processed data to {processed_data_path}")

        logger.info("Data preparation completed successfully")

    except Exception as e:
        logger.error(f"Error preparing data: {e}")
        sys.exit(1)


@cli.command()
@click.option("--logs", is_flag=True, help="Clean log files only")
@click.option("--cache", is_flag=True, help="Clean cache directories only")
@click.option("--pycache", is_flag=True, help="Clean __pycache__ directories only")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose logging")
def cleanup(logs: bool, cache: bool, pycache: bool, verbose: bool):
    """Clean up temporary files and directories."""
    # Set up logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logger = setup_logger("btb.cleanup", log_level)

    # If no specific options are given, clean everything
    clean_all = not (logs or cache or pycache)

    if clean_all or logs:
        logger.info("Cleaning log files...")
        # Root log files
        log_files = list(Path(".").glob("*.log"))
        # Log directory
        log_dir = Path("logs")

        if log_dir.exists() and log_dir.is_dir():
            log_dir_files = list(log_dir.glob("*.log"))
            logger.info(f"Found {len(log_dir_files)} log files in logs/ directory")

            for log_file in log_dir_files:
                logger.debug(f"Removing {log_file}")
                log_file.unlink()

        # Root directory log files
        logger.info(f"Found {len(log_files)} log files in root directory")
        for log_file in log_files:
            logger.debug(f"Removing {log_file}")
            log_file.unlink()

    if clean_all or pycache:
        logger.info("Cleaning __pycache__ directories...")
        pycache_dirs = []
        for root, dirs, _ in os.walk("."):
            if "__pycache__" in dirs:
                pycache_path = os.path.join(root, "__pycache__")
                pycache_dirs.append(pycache_path)

        logger.info(f"Found {len(pycache_dirs)} __pycache__ directories")
        for pycache_dir in pycache_dirs:
            if os.path.isdir(pycache_dir):
                logger.debug(f"Removing {pycache_dir}")
                shutil.rmtree(pycache_dir)

    if clean_all or cache:
        logger.info("Cleaning cache directories...")
        cache_dirs = [".mypy_cache", ".ruff_cache", ".pytest_cache"]
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
                logger.info(f"Removing {cache_dir}/")
                shutil.rmtree(cache_dir)

    logger.info("Cleanup completed!")


if __name__ == "__main__":
    cli()
