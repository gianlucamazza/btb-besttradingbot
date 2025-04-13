"""Base exchange class for all exchange integrations."""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import pandas as pd


class BaseExchange(ABC):
    """Base class for all exchange integrations."""

    def __init__(self, config: Dict):
        """Initialize exchange with configuration."""
        self.config = config

    @abstractmethod
    def get_market_data(
        self, symbol: str, timeframe: str, since: Optional[int] = None, limit: int = 100
    ) -> pd.DataFrame:
        """Get market data from exchange.

        Args:
            symbol: Market symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1h", "4h")
            since: Start time in milliseconds
            limit: Number of candles to fetch

        Returns:
            DataFrame with market data
        """
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        order_type: str,
        side: str,
        amount: float,
        price: Optional[float] = None,
        params: Optional[Dict] = None,
    ) -> Dict:
        """Place an order on the exchange.

        Args:
            symbol: Market symbol (e.g., "BTCUSDT")
            order_type: Order type (e.g., "limit", "market")
            side: Order side ("buy" or "sell")
            amount: Order amount in base currency
            price: Order price (required for limit orders)
            params: Additional parameters

        Returns:
            Order information
        """
        pass

    @abstractmethod
    def get_balance(self) -> Dict:
        """Get account balance.

        Returns:
            Dictionary with balance information
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """Cancel an order.

        Args:
            order_id: Order ID
            symbol: Market symbol
            params: Additional parameters

        Returns:
            Order information
        """
        pass

    @abstractmethod
    def get_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """Get information about an order.

        Args:
            order_id: Order ID
            symbol: Market symbol
            params: Additional parameters

        Returns:
            Order information
        """
        pass

    @abstractmethod
    def get_open_orders(
        self,
        symbol: Optional[str] = None,
        since: Optional[int] = None,
        limit: Optional[int] = None,
        params: Optional[Dict] = None,
    ) -> List[Dict]:
        """Get open orders.

        Args:
            symbol: Market symbol
            since: Start time in milliseconds
            limit: Number of orders to fetch
            params: Additional parameters

        Returns:
            List of open orders
        """
        pass
