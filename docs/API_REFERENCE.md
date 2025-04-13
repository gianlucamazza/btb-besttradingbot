# API Reference

## Overview

This document provides a comprehensive reference for the BestTradingBot API, including modules, classes, and functions that make up the system. It serves as a technical reference for developers looking to understand or extend the functionality of the trading bot.

## Core Modules

### btb.models

The `models` module contains all machine learning model implementations.

#### btb.models.base

```python
class BaseModel(ABC):
    """Abstract base class for all models."""
    
    @abstractmethod
    def __init__(self, config: Dict):
        """Initialize model with configuration."""
        pass
    
    @abstractmethod
    def train(self, train_data: Any, validation_data: Optional[Any] = None) -> Dict:
        """Train the model and return training metrics."""
        pass
    
    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Generate predictions for the input data."""
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk."""
        pass
    
    @classmethod
    @abstractmethod
    def load(cls, path: str) -> 'BaseModel':
        """Load model from disk."""
        pass
```

#### btb.models.transformer

```python
class TransformerModel(BaseModel):
    """Transformer-based model for time series prediction."""
    
    def __init__(self, config: Dict):
        """Initialize transformer model.
        
        Args:
            config: Dict containing model parameters including:
                - input_dim: Dimension of input features
                - hidden_dim: Dimension of hidden layers
                - num_layers: Number of transformer layers
                - nhead: Number of attention heads
                - dropout: Dropout rate
                - output_dim: Dimension of output
        """
        self.config = config
        self.model = self._build_model()
        
    def _build_model(self) -> nn.Module:
        """Build and return PyTorch model."""
        # Implementation details
        
    def train(self, train_data, validation_data=None) -> Dict:
        """Train the model.
        
        Args:
            train_data: Training data loader or dataset
            validation_data: Validation data loader or dataset
            
        Returns:
            Dict of training metrics
        """
        # Implementation details
        
    def predict(self, data) -> np.ndarray:
        """Generate predictions.
        
        Args:
            data: Input data for prediction
            
        Returns:
            Numpy array of predictions
        """
        # Implementation details
        
    def save(self, path: str) -> None:
        """Save model to specified path."""
        # Implementation details
        
    @classmethod
    def load(cls, path: str) -> 'TransformerModel':
        """Load model from specified path."""
        # Implementation details
```

### btb.data

The `data` module handles data loading, processing, and feature engineering.

#### btb.data.loader

```python
class DataLoader:
    """Base class for data loading operations."""
    
    def __init__(self, config: Dict):
        """Initialize data loader with configuration."""
        self.config = config
        
    def load_data(self, symbols: List[str], timeframes: List[str], 
                  start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """Load market data for given symbols and timeframes.
        
        Args:
            symbols: List of market symbols (e.g., "BTCUSDT")
            timeframes: List of timeframes (e.g., "1h", "4h")
            start_date: Start date for data loading
            end_date: End date for data loading
            
        Returns:
            Dict mapping symbol_timeframe to DataFrames
        """
        # Implementation details
```

#### btb.data.features

```python
class FeatureEngineering:
    """Feature engineering for market data."""
    
    def __init__(self, config: Dict = None):
        """Initialize feature engineering with optional configuration."""
        self.config = config or {}
        
    def add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators to the data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with added technical indicators
        """
        # Implementation details
        
    def add_temporal_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features like hour of day, day of week, etc.
        
        Args:
            data: DataFrame with time index
            
        Returns:
            DataFrame with added temporal features
        """
        # Implementation details
        
    def normalize_features(self, data: pd.DataFrame, method: str = "min_max") -> Tuple[pd.DataFrame, Any]:
        """Normalize features in the data.
        
        Args:
            data: DataFrame with features
            method: Normalization method ("min_max", "z_score", "robust")
            
        Returns:
            Tuple of (normalized DataFrame, scaler object)
        """
        # Implementation details
```

### btb.strategies

The `strategies` module contains trading strategy implementations.

#### btb.strategies.factory

```python
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
        Instantiated strategy object
    
    Raises:
        ValueError: If strategy_name is not registered
    """
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_name}")
    return STRATEGY_REGISTRY[strategy_name](params)
```

### btb.exchange

The `exchange` module handles interaction with cryptocurrency exchanges.

#### btb.exchange.bybit

```python
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
        
    def _init_client(self):
        """Initialize exchange client."""
        import ccxt
        # Client initialization details
        
    def get_market_data(self, symbol: str, timeframe: str, since: int = None, limit: int = 100) -> pd.DataFrame:
        """Get market data from exchange.
        
        Args:
            symbol: Market symbol (e.g., "BTCUSDT")
            timeframe: Timeframe (e.g., "1h", "4h")
            since: Start time in milliseconds
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with market data
        """
        # Implementation details
        
    def place_order(self, symbol: str, order_type: str, side: str, amount: float, price: float = None, params: Dict = None) -> Dict:
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
        # Implementation details
        
    def get_balance(self) -> Dict:
        """Get account balance.
        
        Returns:
            Dictionary with balance information
        """
        # Implementation details
```

### btb.backtest

The `backtest` module provides backtesting functionality.

#### btb.backtest.engine

```python
class Backtester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, config: Dict):
        """Initialize backtester with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.strategy = self._init_strategy()
        self.data = self._load_data()
        self.results = None
        
    def _init_strategy(self) -> BaseStrategy:
        """Initialize trading strategy."""
        # Implementation details
        
    def _load_data(self) -> Dict[str, pd.DataFrame]:
        """Load historical data for backtesting."""
        # Implementation details
        
    def run(self) -> Dict:
        """Run the backtest.
        
        Returns:
            Dictionary with backtest results
        """
        # Implementation details
        
    def calculate_metrics(self) -> Dict:
        """Calculate performance metrics.
        
        Returns:
            Dictionary with performance metrics
        """
        # Implementation details
        
    def plot_results(self, filename: str = None):
        """Plot backtest results.
        
        Args:
            filename: Optional path to save the plot
        """
        # Implementation details
        
    def save_results(self, path: str):
        """Save backtest results to file.
        
        Args:
            path: Path to save results
        """
        # Implementation details
```

### btb.run

The `run` module handles live trading operations.

#### btb.run.trader

```python
class Trader:
    """Live trading engine."""
    
    def __init__(self, config: Dict):
        """Initialize trader with configuration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.exchange = self._init_exchange()
        self.strategy = self._init_strategy()
        self.data_manager = self._init_data_manager()
        self.risk_manager = self._init_risk_manager()
        
    def _init_exchange(self) -> BaseExchange:
        """Initialize exchange connection."""
        # Implementation details
        
    def _init_strategy(self) -> BaseStrategy:
        """Initialize trading strategy."""
        # Implementation details
        
    def _init_data_manager(self) -> DataManager:
        """Initialize data manager."""
        # Implementation details
        
    def _init_risk_manager(self) -> RiskManager:
        """Initialize risk manager."""
        # Implementation details
        
    def start(self):
        """Start the trading process."""
        # Implementation details
        
    def stop(self):
        """Stop the trading process."""
        # Implementation details
        
    def update(self):
        """Update market data and process trading logic."""
        # Implementation details
```

## Utility Modules

### btb.utils.config

```python
def load_config(path: str) -> Dict:
    """Load configuration from YAML file.
    
    Args:
        path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Implementation details
    
def validate_config(config: Dict, schema: Dict) -> bool:
    """Validate configuration against schema.
    
    Args:
        config: Configuration dictionary
        schema: Validation schema
        
    Returns:
        True if valid, otherwise raises ValidationError
    """
    # Implementation details
```

### btb.utils.logging

```python
def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the specified name and level.
    
    Args:
        name: Logger name
        level: Logging level
        
    Returns:
        Configured logger
    """
    # Implementation details
```

## Command Line Interface

### btb.cli.main

```python
@click.group()
def cli():
    """BestTradingBot command line interface."""
    pass

@cli.command()
@click.option('--config', '-c', required=True, help='Path to configuration file')
def run(config):
    """Run the trading bot with the specified configuration."""
    # Implementation details

@cli.command()
@click.option('--config', '-c', required=True, help='Path to backtest configuration')
@click.option('--output', '-o', default=None, help='Path to save backtest results')
def backtest(config, output):
    """Run backtest with the specified configuration."""
    # Implementation details

@cli.command()
@click.option('--data', '-d', required=True, help='Path to training data')
@click.option('--model', '-m', required=True, help='Model type to train')
@click.option('--config', '-c', default=None, help='Path to model configuration')
@click.option('--output', '-o', default='models/', help='Directory to save trained model')
def train(data, model, config, output):
    """Train a model with the specified configuration."""
    # Implementation details
```

## Extension Points

The BestTradingBot is designed with extensibility in mind. Here are the main extension points:

1. **Custom Models**: Create new model architectures by subclassing `BaseModel`
2. **Custom Strategies**: Implement new trading strategies by subclassing `BaseStrategy`
3. **Custom Indicators**: Add new technical indicators to `FeatureEngineering`
4. **Exchange Integrations**: Add support for new exchanges by subclassing `BaseExchange`
5. **Risk Management**: Extend `RiskManager` with custom risk management algorithms

## Error Handling

The system uses custom exceptions for different error types:

```python
class BTBError(Exception):
    """Base class for all BTB exceptions."""
    pass

class ConfigError(BTBError):
    """Configuration-related errors."""
    pass

class ExchangeError(BTBError):
    """Exchange-related errors."""
    pass

class DataError(BTBError):
    """Data-related errors."""
    pass

class ModelError(BTBError):
    """Model-related errors."""
    pass
```

## Configuration Schema

Configuration validation uses JSON Schema. Here's an example of the trading configuration schema:

```python
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
                "rate_limit": {"type": "boolean"}
            }
        },
        "trading": {
            "type": "object",
            "required": ["symbols", "timeframes", "strategy"],
            "properties": {
                "symbols": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "timeframes": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "strategy": {"type": "string"},
                "position_size": {"type": "number"},
                "max_open_positions": {"type": "integer"}
            }
        },
        # Other schema sections...
    }
}
```

## API Client Example

Here's an example of using the BestTradingBot API programmatically:

```python
from btb.utils.config import load_config
from btb.exchange.factory import create_exchange
from btb.strategies.factory import create_strategy
from btb.data.loader import DataLoader
from btb.run.trader import Trader

# Load configuration
config = load_config("config/trading_config.yaml")

# Create exchange instance
exchange = create_exchange(config["exchange"]["name"], config["exchange"])

# Create strategy instance
strategy = create_strategy(config["trading"]["strategy"], config["strategy_params"])

# Create data loader
data_loader = DataLoader(config["data"])

# Create and run trader
trader = Trader(config)
trader.start()

# Stop trader when done
try:
    # Trading loop
    while True:
        time.sleep(60)
 except KeyboardInterrupt:
    trader.stop()
```