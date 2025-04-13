"""Factory function for creating exchange instances."""

from typing import Dict

from btb.exchange.base import BaseExchange
from btb.exchange.bybit import BybitExchange

# Registry of available exchange classes
EXCHANGE_REGISTRY = {
    "bybit": BybitExchange,
}


def create_exchange(exchange_name: str, config: Dict) -> BaseExchange:
    """Create an exchange instance.

    Args:
        exchange_name: Name of the exchange to create
        config: Exchange configuration

    Returns:
        Exchange instance

    Raises:
        ValueError: If the exchange name is not recognized
    """
    exchange_class = EXCHANGE_REGISTRY.get(exchange_name.lower())
    if exchange_class is None:
        raise ValueError(f"Unsupported exchange: {exchange_name}")

    return exchange_class(config)
