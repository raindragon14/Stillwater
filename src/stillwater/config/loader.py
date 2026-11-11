"""YAML loading for the frozen configuration schema."""

from __future__ import annotations

from pathlib import Path

import yaml

from stillwater.config.schema import Config

__all__ = ["load_config"]


def load_config(path: str | Path = "configs/base.yaml") -> Config:
    """Load and validate the YAML configuration.

    Parameters
    ----------
    path
        Path to the YAML file. Defaults to ``configs/base.yaml``.

    Returns
    -------
    Config
        Frozen, fully validated configuration.

    Raises
    ------
    pydantic.ValidationError
        If a required field is missing or an unknown field is present.
    """
    with open(path, encoding="utf-8") as handle:
        return Config(**yaml.safe_load(handle))
