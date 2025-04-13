"""Base model class for all machine learning models."""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import numpy as np


class BaseModel(ABC):
    """Abstract base class for all models."""

    @abstractmethod
    def __init__(self, config: Dict):
        """Initialize model with configuration."""
        self.config = config

    @abstractmethod
    def train(self, train_data: Any, validation_data: Optional[Any] = None) -> Dict:
        """Train the model and return training metrics.

        Args:
            train_data: Training data
            validation_data: Optional validation data

        Returns:
            Dictionary with training metrics
        """
        pass

    @abstractmethod
    def predict(self, data: Any) -> np.ndarray:
        """Generate predictions for the input data.

        Args:
            data: Input data for prediction

        Returns:
            NumPy array of predictions
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Args:
            path: Path to save the model
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":
        """Load model from disk.

        Args:
            path: Path to load the model from

        Returns:
            Loaded model instance
        """
        pass
