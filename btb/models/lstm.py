"""LSTM-based model with attention mechanism for time series prediction."""

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from btb.models.base import BaseModel


class LSTMModel(BaseModel):
    """LSTM-based model with attention mechanism for time series prediction."""

    def __init__(self, config: Dict):
        """Initialize LSTM model with attention.

        Args:
            config: Dict containing model parameters including:
                - input_dim: Dimension of input features
                - hidden_dim: Dimension of hidden layers
                - num_layers: Number of LSTM layers
                - dropout: Dropout rate
                - output_dim: Dimension of output
        """
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Module:
        """Build and return PyTorch model."""

        class _LSTMAttentionModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
                super(_LSTMAttentionModel, self).__init__()
                self.hidden_dim = hidden_dim
                self.num_layers = num_layers

                self.lstm = nn.LSTM(
                    input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0
                )

                # Multi-head attention layer
                self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)

                # Output layer
                self.fc = nn.Linear(hidden_dim, output_dim)
                self.dropout = nn.Dropout(dropout)

                # Initialize weights
                self.init_weights()

            def init_weights(self):
                for name, param in self.lstm.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_normal_(param)
                    elif "bias" in name:
                        nn.init.constant_(param, 0.0)

                nn.init.xavier_normal_(self.fc.weight)
                nn.init.constant_(self.fc.bias, 0.0)

            def forward(self, x):
                # LSTM forward pass
                lstm_out, _ = self.lstm(x)  # shape: [batch, seq_len, hidden_dim]

                # Apply attention mechanism
                # Reshape for attention: [seq_len, batch, hidden_dim]
                lstm_out_permuted = lstm_out.permute(1, 0, 2)
                attn_output, _ = self.attention(lstm_out_permuted, lstm_out_permuted, lstm_out_permuted)

                # Reshape back: [batch, seq_len, hidden_dim]
                attn_output = attn_output.permute(1, 0, 2)

                # Apply dropout
                attn_output = self.dropout(attn_output)

                # Final output layer
                # We'll use the last timestep's output for sequence prediction
                output = self.fc(attn_output[:, -1, :])

                return output

        return _LSTMAttentionModel(
            input_dim=self.config["input_dim"],
            hidden_dim=self.config["hidden_dim"],
            output_dim=self.config["output_dim"],
            num_layers=self.config["num_layers"],
            dropout=self.config.get("dropout", 0.1),
        )

    def _get_optimizer(self) -> torch.optim.Optimizer:
        """Get optimizer based on configuration."""
        optimizer_name = self.config.get("optimizer", "adam").lower()
        lr = self.config.get("learning_rate", 0.001)
        weight_decay = self.config.get("weight_decay", 1e-5)

        if optimizer_name == "adam":
            return Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "sgd":
            return SGD(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        elif optimizer_name == "rmsprop":
            return RMSprop(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    def _get_scheduler(self, optimizer: torch.optim.Optimizer) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Get learning rate scheduler based on configuration."""
        scheduler_name = self.config.get("scheduler", None)
        if scheduler_name is None:
            return None

        scheduler_name = scheduler_name.lower()
        if scheduler_name == "cosine":
            epochs = self.config.get("epochs", 100)
            return CosineAnnealingLR(optimizer, T_max=epochs)
        else:
            raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    def train(
        self,
        train_data: Tuple[torch.Tensor, torch.Tensor],
        validation_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Dict:
        """Train the model.

        Args:
            train_data: Tuple of (inputs, targets) tensors
            validation_data: Optional tuple of (inputs, targets) tensors

        Returns:
            Dict of training metrics
        """
        X_train, y_train = train_data
        X_train = X_train.to(self.device)
        y_train = y_train.to(self.device)

        if validation_data is not None:
            X_val, y_val = validation_data
            X_val = X_val.to(self.device)
            y_val = y_val.to(self.device)

        # Get training parameters
        epochs = self.config.get("epochs", 100)
        batch_size = self.config.get("batch_size", 64)
        early_stopping = self.config.get("early_stopping", True)
        patience = self.config.get("patience", 15)
        gradient_clipping = self.config.get("gradient_clipping", None)

        # Set up optimizer and scheduler
        optimizer = self._get_optimizer()
        scheduler = self._get_scheduler(optimizer)

        # Training loop
        best_val_loss = float("inf")
        patience_counter = 0
        train_losses = []
        val_losses = []

        for epoch in range(epochs):
            # Train mode
            self.model.train()
            train_loss = 0.0
            n_batches = 0

            # Create batches
            indices = torch.randperm(len(X_train))
            for start_idx in range(0, len(X_train), batch_size):
                batch_indices = indices[start_idx : start_idx + batch_size]
                batch_X = X_train[batch_indices]
                batch_y = y_train[batch_indices]

                # Forward pass
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = F.mse_loss(outputs, batch_y)

                # Backward pass and optimize
                loss.backward()
                if gradient_clipping is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clipping)
                optimizer.step()

                train_loss += loss.item()
                n_batches += 1

            train_loss /= n_batches
            train_losses.append(train_loss)

            # Validation
            if validation_data is not None:
                val_loss = self._evaluate(X_val, y_val, batch_size)
                val_losses.append(val_loss)

                # Learning rate scheduling
                if scheduler is not None:
                    scheduler.step()

                # Early stopping
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        # Save best model
                        best_model_state = self.model.state_dict().copy()
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            print(f"Early stopping at epoch {epoch + 1}")
                            # Restore best model
                            self.model.load_state_dict(best_model_state)
                            break

        # Return training metrics
        metrics = {
            "train_losses": train_losses,
            "final_train_loss": train_losses[-1],
        }

        if validation_data is not None:
            metrics["val_losses"] = val_losses
            metrics["final_val_loss"] = val_losses[-1]
            metrics["best_val_loss"] = best_val_loss

        return metrics

    def _evaluate(self, X: torch.Tensor, y: torch.Tensor, batch_size: int) -> float:
        """Evaluate model on validation data."""
        self.model.eval()
        val_loss = 0.0
        n_batches = 0

        with torch.no_grad():
            for start_idx in range(0, len(X), batch_size):
                batch_X = X[start_idx : start_idx + batch_size]
                batch_y = y[start_idx : start_idx + batch_size]

                outputs = self.model(batch_X)
                loss = F.mse_loss(outputs, batch_y)

                val_loss += loss.item()
                n_batches += 1

        return val_loss / n_batches

    def predict(self, data: torch.Tensor) -> np.ndarray:
        """Generate predictions.

        Args:
            data: Input tensor

        Returns:
            NumPy array of predictions
        """
        self.model.eval()
        with torch.no_grad():
            data = data.to(self.device)
            predictions = self.model(data)
        return predictions.cpu().numpy()

    def save(self, path: str) -> None:
        """Save model to specified path."""
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> "LSTMModel":
        """Load model from specified path."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model = cls(checkpoint["config"])
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model
