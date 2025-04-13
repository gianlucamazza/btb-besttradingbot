"""Base strategy class for all trading strategies."""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple

import pandas as pd


class BaseStrategy(ABC):
    """Base class for all trading strategies."""

    def __init__(self, params: Optional[Dict] = None):
        """Initialize strategy with parameters."""
        self.params = params if params is not None else {}

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on the data.

        Args:
            data: DataFrame with market data

        Returns:
            DataFrame with signals (1 for buy, -1 for sell, 0 for hold)
        """
        pass

    def calculate_position_size(self, capital: float, price: float) -> float:
        """Calculate position size based on available capital.

        Args:
            capital: Available capital
            price: Current asset price

        Returns:
            Position size in base currency units
        """
        position_pct = self.params.get("position_size", 0.1)  # Default 10%
        return (capital * position_pct) / price if price != 0 else 0

    @abstractmethod
    def should_update_stops(self, position, current_price: float) -> Tuple[bool, Optional[float], Optional[float]]:
        """Determine if stop loss/take profit should be updated.

        Args:
            position: Current position information
            current_price: Current asset price

        Returns:
            Tuple of (update_required, new_stop_loss, new_take_profit)
        """
        pass
