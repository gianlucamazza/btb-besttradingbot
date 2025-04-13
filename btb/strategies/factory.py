"""Factory function for creating strategy instances."""

from typing import Dict

from btb.strategies.base import BaseStrategy
from btb.strategies.lstm_strategy import LSTMAttentionStrategy
from btb.strategies.transformer_strategy import TransformerStrategy

# Registry of available strategy classes
STRATEGY_REGISTRY = {"transformer_strategy": TransformerStrategy, "lstm_strategy": LSTMAttentionStrategy}


def register_strategy(strategy_name: str):
    """Decorator to register a strategy class."""

    def decorator(cls):
        STRATEGY_REGISTRY[strategy_name] = cls
        return cls

    return decorator


def create_strategy(strategy_name: str, params: Dict) -> BaseStrategy:
    """Create a strategy instance by name.

    Args:
        strategy_name: Name of the strategy to create
        params: Strategy parameters

    Returns:
        Strategy instance

    Raises:
        ValueError: If the strategy name is not recognized
    """
    strategy_class = STRATEGY_REGISTRY.get(strategy_name.lower())
    if strategy_class is None:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    return strategy_class(params)
