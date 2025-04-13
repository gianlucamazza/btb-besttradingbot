"""Setup script for BestTradingBot."""

from setuptools import find_packages, setup

# Read version from package __init__.py
with open("btb/__init__.py", "r") as f:
    for line in f:
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip("'\"")
            break

# Read long description from README
with open("README.md", "r") as f:
    long_description = f.read()

# Core dependencies
requires = [
    "numpy>=1.22.0",
    "pandas>=1.4.0",
    "python-dotenv>=0.19.0",
    "pydantic>=1.9.0",
    "PyYAML>=6.0",
    "click>=8.0.0",
    "torch>=2.0.0",
    "torchvision>=0.15.0",
    "scikit-learn>=1.0.0",
    "ccxt>=2.0.0",
    "ta>=0.10.0",
    "pandas-ta>=0.3.14b0",
    "requests>=2.27.0",
    "matplotlib>=3.5.0",
]

# Development dependencies
dev_requires = [
    "pytest>=7.0.0",
    "pytest-cov>=3.0.0",
    "black>=23.11.0",
    "isort>=5.12.0",
    "ruff>=0.1.5",
    "mypy>=1.7.0",
    "pre-commit>=3.5.0",
    "bandit>=1.7.5",
    "types-requests>=2.31.0",
    "types-PyYAML>=6.0.12",
    "types-setuptools>=69.0.0",
    "pydocstyle>=6.3.0",
    "nbqa>=1.7.0",
    "build>=1.0.3",
    "twine>=4.0.2",
    "mkdocs>=1.5.3",
    "mkdocs-material>=9.4.6",
]

setup(
    name="BestTradingBot",
    version=version,
    description="A fully automated cryptocurrency trading bot with machine learning capabilities",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/btb-besttradingbot",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requires,
    extras_require={
        "dev": dev_requires,
    },
    entry_points={
        "console_scripts": [
            "btb=btb.cli.main:cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
    ],
    python_requires=">=3.10",
)
