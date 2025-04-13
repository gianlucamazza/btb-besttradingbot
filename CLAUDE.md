# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test Commands
- Install: `pip install -e .` or `pip install -e ".[dev]"` (with dev tools)
- Run linting: `ruff .`
- Run type checking: `mypy .`
- Run formatting: `black .` and `isort .`
- Run tests: `pytest`
- Run single test: `pytest tests/path/to/test_file.py::test_function -v`
- Run with coverage: `pytest --cov=btb`

## Code Style Guidelines
- **Formatting**: Black with 120 max line length
- **Imports**: Sorted with isort (black profile), grouped (stdlib, third-party, local)
- **Types**: Use type hints for all function parameters and return values
- **Documentation**: Google-style docstrings with triple double quotes
- **Naming**: snake_case for functions/variables, CamelCase for classes
- **Error handling**: Use try/except with specific exceptions, log errors
- **Project structure**: Follow existing module organization patterns
- **Testing**: Write pytest-compatible tests with appropriate fixtures

Follow pre-commit hooks configuration (black, isort, ruff, mypy) for code quality.