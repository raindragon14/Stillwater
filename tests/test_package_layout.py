"""Package-boundary smoke tests.

These tests fail if a refactor introduces an import cycle, removes a public package,
or breaks the compatibility shims that external users still import.
"""

from __future__ import annotations

import importlib

import pytest

PACKAGES = (
    "stillwater.config",
    "stillwater.data",
    "stillwater.features",
    "stillwater.models",
    "stillwater.optimization",
    "stillwater.backtest",
    "stillwater.experiments",
    "stillwater.reporting",
    "stillwater.provenance",
    "stillwater.cli",
)


@pytest.mark.parametrize("name", PACKAGES)
def test_package_imports(name: str) -> None:
    """Every public subpackage imports without a cycle."""
    assert importlib.import_module(name) is not None


def test_cli_wrapper_points_at_cli_module() -> None:
    """``python -m stillwater`` and the console script share one implementation."""
    from stillwater import __main__, cli

    assert __main__.main is cli.main
    assert __main__.STAGES is cli.STAGES
    assert __main__.ALL_STAGES is cli.ALL_STAGES
