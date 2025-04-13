# Contributing Guidelines

## Overview

Thank you for your interest in contributing to BestTradingBot! This document provides guidelines and instructions for contributing to the project. We appreciate all contributions, whether they're bug reports, feature requests, documentation improvements, or code changes.

## Code of Conduct

Please read and follow our [Code of Conduct](CODE_OF_CONDUCT.md) to keep our community respectful and inclusive.

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Git
- Basic understanding of PyTorch, trading systems, and machine learning

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork locally:

```bash
git clone https://github.com/yourusername/btb-besttradingbot.git
cd btb-besttradingbot
```

3. Set up a virtual environment:

```bash
# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
# .venv\Scripts\activate
```

4. Install development dependencies:

```bash
pip install -e ".[dev]"
pip install -r requirements.txt
```

5. Set up pre-commit hooks:

```bash
pre-commit install
```

## Development Process

### Branching Strategy

We follow a simplified Git flow model:

- `main`: Stable release branch
- `develop`: Active development branch
- Feature branches: Created from `develop` for specific features or fixes

Name your feature branches following this convention: `feature/descriptive-name` or `fix/issue-description`.

### Pull Request Process

1. Create a branch from `develop` for your feature or fix
2. Make your changes, following our coding standards
3. Add or update tests to cover your changes
4. Update documentation if necessary
5. Run tests and static analysis to ensure everything passes
6. Submit a pull request to the `develop` branch
7. Address any feedback during code review
8. Once approved, your PR will be merged

## Coding Standards

### Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with a few exceptions:

- Line length: 100 characters maximum
- Import order: standard library, third-party packages, local modules

We use a combination of [Black](https://black.readthedocs.io/), [isort](https://pycqa.github.io/isort/), and [ruff](https://beta.ruff.rs/) for code formatting and linting.

### Docstrings

We use [Google-style docstrings](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) for all public functions, classes, and methods:

```python
def function_with_types_in_docstring(param1: int, param2: str) -> bool:
    """Example function with types documented in the docstring.
    
    Args:
        param1: The first parameter.
        param2: The second parameter.
        
    Returns:
        The return value. True for success, False otherwise.
        
    Raises:
        ValueError: If param1 is less than 0.
    """
    if param1 < 0:
        raise ValueError("param1 must be >= 0")
    return param1 > 0
```

### Type Hints

We use Python type hints for all function arguments and return values:

```python
from typing import Dict, List, Optional, Tuple, Union

def process_data(data: pd.DataFrame, config: Dict[str, Any]) -> Tuple[pd.DataFrame, Dict[str, float]]:
    # Implementation
    return processed_data, metrics
```

## Testing

### Writing Tests

We use [pytest](https://docs.pytest.org/) for our testing framework. All tests should be placed in the `tests/` directory, following the same structure as the module they test.

Each test file should start with `test_` and each test function should also start with `test_`.

```python
# tests/models/test_transformer.py
import pytest
import torch
from btb.models.transformer import TransformerModel

def test_transformer_initialization():
    config = {
        "input_dim": 10,
        "hidden_dim": 64,
        "output_dim": 2,
        "num_layers": 3,
        "nhead": 8
    }
    model = TransformerModel(config)
    assert isinstance(model, TransformerModel)
    # More assertions...

def test_transformer_forward_pass():
    # Test the forward pass logic
    # ...
```

### Running Tests

To run the tests:

```bash
pytest
```

To run tests with coverage:

```bash
pytest --cov=btb tests/
```

## Documentation

### Writing Documentation

We use Markdown for documentation. All documentation files should be placed in the `docs/` directory.

When adding new features, please update relevant documentation files or create new ones if needed. Documentation should be clear, concise, and include examples when appropriate.

### Building Documentation

To build the documentation locally:

```bash
mkdocs build
```

To serve the documentation site locally for preview:

```bash
mkdocs serve
```

## Submitting Changes

### Issue Tracking

Before starting work on a new feature or fix, check the issue tracker to see if it's already being discussed. If not, create a new issue to discuss the proposed changes before investing significant time in implementation.

### Pull Requests

When submitting a pull request:

1. Reference any related issues in the PR description
2. Provide a clear description of the changes made
3. Include any necessary testing instructions
4. Update relevant documentation
5. Ensure all CI checks pass

## Release Process

Our release process follows these steps:

1. Feature freeze on `develop` branch
2. Comprehensive testing and bug fixing
3. Version bump according to [Semantic Versioning](https://semver.org/)
4. Release candidate testing
5. Merge to `main` branch
6. Tag release with version number
7. Build and publish package

## Additional Resources

- [Project Home](index.md)
- [Installation Guide](INSTALLATION.md)
- [Configuration Guide](CONFIGURATION.md)
- [Discord Community](https://discord.gg/yourproject)

## Recognition

All contributors will be recognized in our [CONTRIBUTORS.md](CONTRIBUTORS.md) file. We value and appreciate your contributions to making BestTradingBot better!

## Questions and Support

If you have questions or need support with contributing, please:

- Open an issue with the "question" label
- Join our [Discord community](https://discord.gg/yourproject)
- Contact the maintainers directly

Thank you for contributing to BestTradingBot!