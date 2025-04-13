"""Transformer-based model for time series prediction."""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, Adam, RMSprop
from torch.optim.lr_scheduler import CosineAnnealingLR

from btb.models.base import BaseModel


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
        super().__init__(config)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._build_model().to(self.device)

    def _build_model(self) -> nn.Module:
        """Build and return PyTorch model."""

        class _TransformerModel(nn.Module):
            def __init__(
                self, input_dim, d_model, output_dim, num_encoder_layers, nhead, dim_feedforward=None, dropout=0.1
            ):
                super(_TransformerModel, self).__init__()
                self.d_model = d_model
                self.input_dim = input_dim

                if dim_feedforward is None:
                    dim_feedforward = d_model * 4

                # Feature embedding
                self.embedding = nn.Linear(input_dim, d_model)

                # Positional encoding
                self.pos_encoder = PositionalEncoding(d_model, dropout)

                # Transformer encoder
                encoder_layers = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
                )
                self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)

                # Output layers
                self.decoder = nn.Linear(d_model, output_dim)

                # Initialize weights
                self.init_weights()

            def init_weights(self):
                initrange = 0.1
                nn.init.uniform_(self.embedding.weight, -initrange, initrange)
                nn.init.zeros_(self.decoder.bias)
                nn.init.uniform_(self.decoder.weight, -initrange, initrange)

            def forward(self, src, src_mask=None):
                # Input has shape [batch, seq_len, features]

                # Apply feature embedding
                src = self.embedding(src)

                # Apply positional encoding
                src = self.pos_encoder(src)

                # Apply transformer encoder
                # For nn.TransformerEncoder with batch_first=True:
                # src shape: [batch, seq_len, d_model]
                output = self.transformer_encoder(src, src_mask)

                # We'll use the last timestep for sequence prediction
                output = self.decoder(output[:, -1, :])

                return output

        # Positional encoding to provide sequence position information
        class PositionalEncoding(nn.Module):
            def __init__(self, d_model, dropout=0.1, max_len=5000):
                super(PositionalEncoding, self).__init__()
                self.dropout = nn.Dropout(p=dropout)

                # Create positional encodings
                pe = torch.zeros(max_len, d_model)
                position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
                div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
                pe[:, 0::2] = torch.sin(position * div_term)
                pe[:, 1::2] = torch.cos(position * div_term)
                pe = pe.unsqueeze(0).transpose(0, 1)
                self.register_buffer("pe", pe)

            def forward(self, x):
                # x shape: [batch, seq_len, d_model]
                pos_encoding = self.pe[: x.size(1), :].transpose(0, 1)
                x = x + pos_encoding.unsqueeze(0)
                return self.dropout(x)

        # Get dimension names based on config keys
        d_model = self.config.get("d_model", self.config.get("hidden_dim", 128))
        dim_feedforward = self.config.get("dim_feedforward", d_model * 4)
        num_encoder_layers = self.config.get("num_encoder_layers", self.config.get("num_layers", 2))

        return _TransformerModel(
            input_dim=self.config["input_dim"],
            d_model=d_model,
            output_dim=self.config["output_dim"],
            num_encoder_layers=num_encoder_layers,
            nhead=self.config["nhead"],
            dim_feedforward=dim_feedforward,
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
            for i in range(0, len(X_train), batch_size):
                batch_X = X_train[i : i + batch_size]
                batch_y = y_train[i : i + batch_size]

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
            for i in range(0, len(X), batch_size):
                batch_X = X[i : i + batch_size]
                batch_y = y[i : i + batch_size]

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
    def load(cls, path: str) -> "TransformerModel":
        """Load model from specified path."""
        checkpoint = torch.load(path, map_location=torch.device("cpu"))
        model = cls(checkpoint["config"])
        model.model.load_state_dict(checkpoint["model_state_dict"])
        return model
