"""Bybit exchange integration."""

from typing import Dict, List, Optional

import ccxt
import pandas as pd

from btb.exchange.base import BaseExchange


class BybitExchange(BaseExchange):
    """Bybit exchange integration."""

    def __init__(self, config: Dict):
        """Initialize Bybit exchange connection.

        Args:
            config: Dictionary with configuration including:
                - api_key: Bybit API key
                - api_secret: Bybit API secret
                - testnet: Whether to use testnet (bool)
        """
        super().__init__(config)
        self.client = self._init_client()

    def _init_client(self) -> ccxt.bybit:
        """Initialize exchange client."""
        api_key = self.config.get("api_key")
        api_secret = self.config.get("api_secret")
        use_testnet = self.config.get("testnet", True)

        # Options for the CCXT client
        options = {
            "adjustForTimeDifference": True,
            "recvWindow": 5000,
        }

        # Create CCXT client
        client = ccxt.bybit(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": self.config.get("rate_limit", True),
                "options": options,
            }
        )

        # Set testnet if required
        if use_testnet:
            client.urls["api"] = client.urls["test"]

        return client

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
        # Fetch OHLCV data
        ohlcv = self.client.fetch_ohlcv(symbol, timeframe, since, limit)

        # Convert to DataFrame
        df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)

        return df

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
        # Normalize order type
        order_type = order_type.lower()

        # Additional parameters
        params = params or {}

        # Place order
        if order_type == "market":
            return self.client.create_market_order(symbol, side, amount, params=params)
        elif order_type == "limit":
            if price is None:
                raise ValueError("Price is required for limit orders")
            return self.client.create_limit_order(symbol, side, amount, price, params=params)
        else:
            raise ValueError(f"Unsupported order type: {order_type}")

    def get_balance(self) -> Dict:
        """Get account balance.

        Returns:
            Dictionary with balance information
        """
        return self.client.fetch_balance()

    def cancel_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """Cancel an order.

        Args:
            order_id: Order ID
            symbol: Market symbol
            params: Additional parameters

        Returns:
            Order information
        """
        params = params or {}
        return self.client.cancel_order(order_id, symbol, params=params)

    def get_order(self, order_id: str, symbol: str, params: Optional[Dict] = None) -> Dict:
        """Get information about an order.

        Args:
            order_id: Order ID
            symbol: Market symbol
            params: Additional parameters

        Returns:
            Order information
        """
        params = params or {}
        return self.client.fetch_order(order_id, symbol, params=params)

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
        params = params or {}
        return self.client.fetch_open_orders(symbol, since, limit, params=params)

    def get_positions(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get current positions.

        Args:
            symbol: Market symbol (optional)

        Returns:
            List of positions
        """
        params = {}
        if symbol:
            params["symbol"] = symbol
        return self.client.fetch_positions(params=params)

    def set_leverage(self, symbol: str, leverage: int) -> Dict:
        """Set leverage for a symbol.

        Args:
            symbol: Market symbol
            leverage: Leverage value

        Returns:
            Response from the exchange
        """
        return self.client.set_leverage(leverage, symbol)

    def set_stop_loss(
        self, symbol: str, position_side: str, stop_loss_price: float, params: Optional[Dict] = None
    ) -> Dict:
        """Set stop loss for a position.

        Args:
            symbol: Market symbol
            position_side: Position side ("buy" or "sell")
            stop_loss_price: Stop loss price
            params: Additional parameters

        Returns:
            Response from the exchange
        """
        params = params or {}
        params["stopLoss"] = stop_loss_price
        return self.client.edit_position(symbol, position_side, params=params)

    def set_take_profit(
        self, symbol: str, position_side: str, take_profit_price: float, params: Optional[Dict] = None
    ) -> Dict:
        """Set take profit for a position.

        Args:
            symbol: Market symbol
            position_side: Position side ("buy" or "sell")
            take_profit_price: Take profit price
            params: Additional parameters

        Returns:
            Response from the exchange
        """
        params = params or {}
        params["takeProfit"] = take_profit_price
        return self.client.edit_position(symbol, position_side, params=params)

    def get_ticker(self, symbol: str) -> Dict:
        """Get ticker information for a symbol.

        Args:
            symbol: Market symbol

        Returns:
            Ticker information
        """
        return self.client.fetch_ticker(symbol)
