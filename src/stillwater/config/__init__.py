"""Configuration package: frozen schema plus YAML loading."""

from stillwater.config.loader import load_config
from stillwater.config.params import BacktestParams
from stillwater.config.schema import (
    Config,
    Data,
    Features,
    Reporting,
    Splits,
    Trading,
    Train,
    Workers,
)

__all__ = [
    "BacktestParams",
    "Config",
    "Data",
    "Features",
    "Reporting",
    "Splits",
    "Trading",
    "Train",
    "Workers",
    "load_config",
]
