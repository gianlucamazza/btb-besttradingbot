"""Configuration loading and validation utilities."""

import os
from typing import Any, Dict

import yaml
from pydantic import ValidationError, create_model


def load_config(path: str) -> Dict[str, Any]:
    """Load configuration from YAML file.

    Args:
        path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r") as f:
        config = yaml.safe_load(f)

    return dict(config)


def validate_config(config: Dict, schema: Dict) -> bool:
    """Validate configuration against schema.

    Args:
        config: Configuration dictionary
        schema: Validation schema

    Returns:
        True if valid, otherwise raises ValidationError
    """
    # Generate dynamic model from schema
    field_definitions: Dict[str, Any] = {}
    for field_name, field_schema in schema["properties"].items():
        # Convert schema type to Python type
        field_type = _schema_type_to_python_type(field_schema["type"])

        # Check if field is required
        if "required" in schema and field_name in schema["required"]:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (field_type, None)

    # Create dynamic model
    ConfigModel = create_model("ConfigModel", **field_definitions)

    # Validate config against model
    try:
        ConfigModel(**config)
        return True
    except ValidationError as e:
        raise e


def _schema_type_to_python_type(schema_type: str) -> Any:
    """Convert JSON Schema type to Python type."""
    type_map = {"string": str, "integer": int, "number": float, "boolean": bool, "array": list, "object": dict}

    return type_map.get(schema_type, Any)


# Validation schemas for different configuration files
TRADING_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["exchange", "trading", "risk", "data"],
    "properties": {
        "exchange": {
            "type": "object",
            "required": ["name", "testnet"],
            "properties": {
                "name": {"type": "string"},
                "testnet": {"type": "boolean"},
                "rate_limit": {"type": "boolean"},
            },
        },
        "trading": {
            "type": "object",
            "required": ["symbols", "timeframes", "strategy"],
            "properties": {
                "symbols": {"type": "array"},
                "timeframes": {"type": "array"},
                "strategy": {"type": "string"},
                "position_size": {"type": "number"},
                "max_open_positions": {"type": "integer"},
            },
        },
        "risk": {
            "type": "object",
            "required": ["max_drawdown", "stop_loss", "take_profit"],
            "properties": {
                "max_drawdown": {"type": "number"},
                "stop_loss": {"type": "number"},
                "take_profit": {"type": "number"},
                "trailing_stop": {"type": "boolean"},
                "trailing_stop_activation": {"type": "number"},
                "trailing_stop_distance": {"type": "number"},
            },
        },
        "data": {
            "type": "object",
            "required": ["lookback_period", "features"],
            "properties": {
                "lookback_period": {"type": "integer"},
                "features": {"type": "array"},
                "cache_data": {"type": "boolean"},
                "update_interval": {"type": "integer"},
            },
        },
        "execution": {
            "type": "object",
            "properties": {
                "order_type": {"type": "string"},
                "limit_order_expiry": {"type": "integer"},
                "max_retries": {"type": "integer"},
                "retry_delay": {"type": "integer"},
            },
        },
    },
}

BACKTEST_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["backtest"],
    "properties": {
        "backtest": {
            "type": "object",
            "required": ["start_date", "end_date", "symbols", "timeframes", "strategy"],
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
                "symbols": {"type": "array"},
                "timeframes": {"type": "array"},
                "strategy": {"type": "string"},
                "initial_capital": {"type": "number"},
                "commission": {"type": "number"},
                "slippage": {"type": "number"},
            },
        },
        "data_processing": {
            "type": "object",
            "properties": {
                "train_test_split": {"type": "number"},
                "feature_engineering": {"type": "boolean"},
                "normalization": {"type": "string"},
                "fill_missing": {"type": "string"},
            },
        },
        "metrics": {
            "type": "object",
            "properties": {
                "calculate_sharpe": {"type": "boolean"},
                "calculate_sortino": {"type": "boolean"},
                "calculate_drawdown": {"type": "boolean"},
                "calculate_win_rate": {"type": "boolean"},
                "benchmark": {"type": "string"},
            },
        },
        "strategy_params": {"type": "object"},
    },
}

MODEL_CONFIG_SCHEMA = {
    "type": "object",
    "required": ["model", "training", "features"],
    "properties": {
        "model": {
            "type": "object",
            "required": ["type"],
            "properties": {
                "type": {"type": "string"},
                "hidden_dim": {"type": "integer"},
                "num_layers": {"type": "integer"},
                "dropout": {"type": "number"},
                "attention_heads": {"type": "integer"},
            },
        },
        "training": {
            "type": "object",
            "properties": {
                "batch_size": {"type": "integer"},
                "learning_rate": {"type": "number"},
                "epochs": {"type": "integer"},
                "early_stopping": {"type": "boolean"},
                "patience": {"type": "integer"},
                "optimizer": {"type": "string"},
                "scheduler": {"type": "string"},
                "weight_decay": {"type": "number"},
                "gradient_clipping": {"type": "number"},
            },
        },
        "features": {
            "type": "object",
            "properties": {
                "sequence_length": {"type": "integer"},
                "technical_indicators": {"type": "boolean"},
                "sentiment_analysis": {"type": "boolean"},
                "include_volume": {"type": "boolean"},
                "include_time_features": {"type": "boolean"},
            },
        },
        "prediction": {
            "type": "object",
            "properties": {
                "output_type": {"type": "string"},
                "prediction_horizon": {"type": "integer"},
                "confidence_threshold": {"type": "number"},
            },
        },
    },
}
