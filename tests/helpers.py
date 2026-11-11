"""Helpers shared by the test modules."""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def needs(relative: str) -> pytest.MarkDecorator:
    """Skip a test when a generated artifact is absent.

    Artifact-dependent tests must not fail on a bare clone; they run after
    ``make all`` and are skipped otherwise.
    """
    return pytest.mark.skipif(
        not (REPO_ROOT / relative).exists(),
        reason=f"missing artifact {relative}; run `make all` first",
    )
