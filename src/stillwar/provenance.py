"""Provenance blocks that make every artifact attributable to code, config and data.

Without this module a result JSON cannot answer "which configuration and which
commit produced me?". That gap is how a manuscript table drifted from its own
artifact. Every writer in the pipeline now stamps :func:`artifact_metadata` into its
output, and ``tests/test_provenance.py`` refuses to accept an artifact without one.
"""

import hashlib
import platform
import subprocess
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

from stillwar.config import Config

_PKG_DIR = Path(__file__).resolve().parent


@cache
def _git(*args: str) -> str:
    """Run a git command in the repository root; empty string if git is unavailable."""
    try:
        out = subprocess.run(
            ["git", *args],
            cwd=_PKG_DIR.parent.parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def code_digest(depends_on: Sequence[str]) -> str:
    """SHA-256 over the source files of the modules a stage depends on.

    Hashing only the relevant modules means editing ``figures.py`` does not
    invalidate a cached training run, while editing ``portfolio.py`` does.

    Parameters
    ----------
    depends_on
        Module names inside ``stillwar``, e.g. ``("portfolio", "params")``.

    Returns
    -------
    str
        First 16 hex characters of the combined digest.
    """
    h = hashlib.sha256()
    for name in sorted(depends_on):
        path = _PKG_DIR / f"{name}.py"
        if not path.exists():
            raise FileNotFoundError(f"unknown module in depends_on: {name}")
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()[:16]


def library_versions() -> dict[str, str]:
    """Versions of the interpreter and every library that can change a result."""
    import numpy
    import scipy
    import torch

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }


def artifact_metadata(cfg: Config, depends_on: Sequence[str]) -> dict[str, Any]:
    """Build the provenance block stamped into an artifact.

    Parameters
    ----------
    cfg
        Configuration that produced the artifact.
    depends_on
        Modules whose source determines the result (see :func:`code_digest`).

    Returns
    -------
    dict
        ``config_digest``, ``code_digest``, ``git_sha``, ``git_dirty``, ``versions``
        and ``created_utc``.
    """
    status = _git("status", "--porcelain")
    return {
        "config_digest": cfg.digest,
        "code_digest": code_digest(depends_on),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(status),
        "versions": library_versions(),
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def write_json(path: str | Path, metadata: dict[str, Any], results: Any) -> None:
    """Write ``{"metadata": ..., "results": ...}`` with stable formatting.

    Parameters
    ----------
    path
        Destination; parent directories are created.
    metadata
        Output of :func:`artifact_metadata`.
    results
        JSON-serialisable payload. Must contain floats, never pre-formatted strings:
        formatting belongs to ``stillwar.tables``.
    """
    import json

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"metadata": metadata, "results": results}, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> tuple[dict[str, Any], Any]:
    """Read an artifact written by :func:`write_json` and return ``(metadata, results)``.

    Raises
    ------
    KeyError
        If the file lacks a provenance block, which means it predates this module or
        was produced by an unregistered path.
    """
    import json

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "metadata" not in payload or "results" not in payload:
        raise KeyError(f"{path} has no provenance block; regenerate it with the pipeline")
    return payload["metadata"], payload["results"]
