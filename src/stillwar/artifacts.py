"""Compatibility shim for the old ``stillwar.artifacts`` import path.

New code should import from :mod:`stillwar.provenance.artifacts`.
"""

from stillwar.provenance.artifacts import (
    BASELINE_KEYS,
    MODEL_KEYS,
    load_all,
    load_baselines,
    load_forecasts,
    save_baselines,
    save_forecasts,
)

__all__ = [
    "BASELINE_KEYS",
    "MODEL_KEYS",
    "load_all",
    "load_baselines",
    "load_forecasts",
    "save_baselines",
    "save_forecasts",
]
