"""Logging utility functions."""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "btb", log_level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Set up a logger with the specified name and level.

    Args:
        name: Logger name
        log_level: Logging level
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)

    # Add console handler to logger
    logger.addHandler(console_handler)

    # Create file handler if log_file is specified
    if log_file:
        # Create logs directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)

        # Create file handler and set level
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)

        # Add file handler to logger
        logger.addHandler(file_handler)

    return logger


def get_log_file_path(base_dir: str = "logs", prefix: str = "btb_") -> str:
    """Generate a log file path with timestamp.

    Args:
        base_dir: Base directory for log files
        prefix: Prefix for log file name

    Returns:
        Path to log file
    """
    # Create logs directory if it doesn't exist
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Generate timestamped filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{prefix}{timestamp}.log"

    return os.path.join(base_dir, log_file)
