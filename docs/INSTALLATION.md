---
redirect: INSTALLATION.md
---

# Redirecting...

If you are not redirected automatically, follow this [link to the installation guide](INSTALLATION.md).

# Installation Guide

## Prerequisites

- Python 3.10 or higher
- Git
- Access to Bybit API (for live trading)

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/gianlucamazza/btb-besttradingbot.git
cd btb-besttradingbot
```

### 2. Set Up a Virtual Environment

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

Create a `.env` file in the project root directory with your Bybit API credentials:

```
BYBIT_API_KEY=your_api_key
BYBIT_API_SECRET=your_api_secret
BYBIT_TESTNET=True  # Set to False for live trading
```

### 5. Verify Installation

Run the test suite to ensure everything is set up correctly:

```bash
python -m pytest
```

## Docker Installation (Alternative)

If you prefer using Docker:

```bash
# Build the Docker image
docker build -t besttradingbot .

# Run the container
docker run -d --name btb-container \
  --env-file .env \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/logs:/app/logs \
  besttradingbot
```

## GPU Support (Optional)

For accelerated model training and inference:

```bash
# Install PyTorch with CUDA support
pip install torch==2.0.0+cu117 torchvision==0.15.0+cu117 -f https://download.pytorch.org/whl/cu117/torch_stable.html
```

Verify GPU availability:

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
```

## Troubleshooting

### Common Issues

1. **Missing dependencies**: Ensure you have all required system libraries installed.

2. **API connection failures**: Check your network connection and API key permissions.

3. **CUDA errors**: Verify that your NVIDIA drivers match the CUDA version required by PyTorch.

For further assistance, please open an issue on the GitHub repository.