"""Shared fixtures for the test suite."""

from pathlib import Path

import pytest

from stillwar.config import Config, load_config
from tests.helpers import REPO_ROOT


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root, so tests do not depend on the CWD."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def cfg() -> Config:
    """The repository's own configuration, so tests exercise the real parameters."""
    return load_config(REPO_ROOT / "configs/base.yaml")
