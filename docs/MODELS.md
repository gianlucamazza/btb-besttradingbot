# Model Architecture

## Overview

BestTradingBot employs advanced deep learning models implemented in PyTorch to predict market movements and generate trading signals. This document describes the architecture of these models, their implementation, and customization options.

## Model Types

### Transformer Model

The primary model architecture is based on the Transformer, which has demonstrated superior performance in sequence modeling tasks.

#### Architecture Details

```
Input → Embedding → Positional Encoding → Transformer Encoder → MLP Head → Output
```

- **Embedding Layer**: Converts numerical features into a high-dimensional representation
- **Positional Encoding**: Adds information about the sequence position
- **Transformer Encoder**: Self-attention mechanism to capture temporal relationships
- **MLP Head**: Final layers that output predictions

#### Implementation

```python
class TransformerModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, nhead, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layers = nn.TransformerEncoderLayer(hidden_dim, nhead, hidden_dim*4, dropout)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.decoder = nn.Linear(hidden_dim, output_dim)
        self.init_weights()
        
    def init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)
        
    def forward(self, src, src_mask=None):
        src = self.embedding(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output
```

### LSTM Model

Long Short-Term Memory networks are used for capturing long-term dependencies in time series data.

#### Architecture Details

```
Input → LSTM Layers → Attention Layer → MLP Head → Output
```

#### Implementation

```python
class LSTMAttentionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, dropout=0.1):
        super(LSTMAttentionModel, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, dropout=dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        output = self.fc(attn_output)
        return output
```

### CNN Model

Convolutional Neural Networks are effective at capturing local patterns in the data.

#### Architecture Details

```
Input → Conv1D Layers → Global Pooling → MLP Head → Output
```

#### Implementation

```python
class CNN1DModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, kernel_sizes=[3, 5, 7], dropout=0.1):
        super(CNN1DModel, self).__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=input_dim, out_channels=hidden_dim, kernel_size=k)
            for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * len(kernel_sizes), output_dim)
        
    def forward(self, x):
        x = x.permute(0, 2, 1)  # [batch, features, sequence]
        conv_results = []
        for conv in self.convs:
            conv_out = F.relu(conv(x))
            pool_out = F.adaptive_max_pool1d(conv_out, 1).squeeze(-1)
            conv_results.append(pool_out)
        concat = torch.cat(conv_results, dim=1)
        dropout = self.dropout(concat)
        output = self.fc(dropout)
        return output
```

### Ensemble Model

Combines multiple model types to improve prediction robustness.

#### Architecture Details

```
Input → [Transformer, LSTM, CNN] → Aggregation Layer → MLP Head → Output
```

#### Implementation

```python
class EnsembleModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, config):
        super(EnsembleModel, self).__init__()
        self.transformer = TransformerModel(input_dim, hidden_dim, hidden_dim, 
                                         config["transformer"]["num_layers"], 
                                         config["transformer"]["nhead"])
        self.lstm = LSTMAttentionModel(input_dim, hidden_dim, hidden_dim, 
                                     config["lstm"]["num_layers"])
        self.cnn = CNN1DModel(input_dim, hidden_dim, hidden_dim)
        self.aggregation = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        transformer_out = self.transformer(x)
        lstm_out = self.lstm(x)
        cnn_out = self.cnn(x)
        # Use the output from the last time step for each model
        transformer_out = transformer_out[:, -1, :]
        lstm_out = lstm_out[:, -1, :]
        # Concatenate model outputs
        combined = torch.cat([transformer_out, lstm_out, cnn_out], dim=1)
        aggregated = F.relu(self.aggregation(combined))
        output = self.output_layer(aggregated)
        return output
```

## Feature Engineering

A crucial aspect of model performance is feature engineering. The system processes raw market data to extract meaningful features:

1. **Technical Indicators**:
   - Moving averages (SMA, EMA, WMA)
   - RSI (Relative Strength Index)
   - MACD (Moving Average Convergence Divergence)
   - Bollinger Bands
   - Stochastic Oscillator
   - ATR (Average True Range)

2. **Price Data Transformations**:
   - Log returns
   - Normalized prices
   - Price momentum
   - Volatility measures

3. **Temporal Features**:
   - Time of day
   - Day of week
   - Month
   - Holiday indicators

4. **Market Sentiment Features** (optional):
   - Social media sentiment
   - News sentiment
   - Market fear/greed indicators

## Training Process

The model training pipeline includes:

1. **Data Preprocessing**:
   - Splitting data into training, validation, and test sets
   - Feature normalization
   - Sequence creation (sliding window approach)

2. **Model Training**:
   - Mini-batch gradient descent
   - Configurable optimizers (Adam, SGD, RMSprop)
   - Learning rate schedulers
   - Early stopping based on validation loss

3. **Hyperparameter Optimization**:
   - Bayesian optimization for hyperparameter tuning
   - Cross-validation to ensure robustness

## Model Evaluation

Models are evaluated using both traditional ML metrics and financial metrics:

1. **ML Metrics**:
   - Mean Squared Error (MSE)
   - Mean Absolute Error (MAE)
   - R-squared
   - Directional Accuracy

2. **Financial Metrics**:
   - Backtested returns
   - Sharpe ratio
   - Sortino ratio
   - Maximum drawdown
   - Win rate

## Deployment

Trained models are serialized and saved for deployment:

```python
# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': model_config,
    'feature_names': feature_names,
    'scaler': scaler,  # For feature normalization
    'performance_metrics': metrics
}, 'models/transformer_btcusdt_1h.pt')

# Load model for inference
checkpoint = torch.load('models/transformer_btcusdt_1h.pt')
model = TransformerModel(**checkpoint['config'])
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()  # Set to evaluation mode
```

## Hardware Acceleration

The system supports GPU acceleration for faster training and inference:

```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)
input_data = input_data.to(device)
```

## Customization

The model architecture can be customized through the configuration files. Key parameters include:

- Number of layers
- Hidden dimension size
- Number of attention heads
- Dropout rate
- Optimizer choice and learning rate
- Sequence length
- Feature selection

Refer to the `model_config.yaml` file for all available options.