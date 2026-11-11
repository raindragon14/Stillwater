"""Configuration package: frozen schema plus YAML loading."""

from stillwar.config.loader import load_config
from stillwar.config.params import BacktestParams
from stillwar.config.schema import (
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
