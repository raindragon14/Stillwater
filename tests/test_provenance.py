"""Provenance, manuscript-table synchronisation, and artifact reproducibility."""

import hashlib
import json
from pathlib import Path

import pytest

from stillwar.provenance.metadata import artifact_metadata, code_digest, read_json, write_json
from stillwar.reporting.tables import (
    load_blocks,
    render_table1,
    render_table2,
    render_table3,
    sync_manuscript,
)
from tests.helpers import needs

PROCESSED = "data/processed"
ARTIFACTS = (
    "table1.json",
    "table2.json",
    "table3.json",
    "phase4.json",
    "kkt_log.json",
    "gate_weights.json",
    "exp_unconstrained_tolerance.json",
    "exp_shrinkage.json",
    "exp_volatility_penalty.json",
    "trials.json",
)


def test_code_digest_is_stable_and_sensitive() -> None:
    """The code digest is deterministic and changes with the module set."""
    assert code_digest(("config", "params")) == code_digest(("params", "config"))
    assert code_digest(("config",)) != code_digest(("config", "params"))


def test_code_digest_rejects_unknown_module() -> None:
    """An unknown module name fails loudly rather than silently hashing nothing."""
    with pytest.raises(FileNotFoundError):
        code_digest(("not_a_module",))


def test_metadata_block_is_complete(cfg) -> None:
    """Every provenance block carries the fields a reviewer needs."""
    meta = artifact_metadata(cfg, ("config", "params"))
    for key in ("config_digest", "code_digest", "git_sha", "git_dirty", "versions", "created_utc"):
        assert key in meta
    for lib in ("python", "numpy", "scipy", "torch"):
        assert lib in meta["versions"]


def test_write_read_roundtrip(tmp_path, cfg) -> None:
    """An artifact written by ``write_json`` reads back with its provenance intact."""
    path = tmp_path / "demo.json"
    write_json(path, artifact_metadata(cfg, ("config",)), {"value": 1.5})
    meta, results = read_json(path)
    assert meta["config_digest"] == cfg.digest
    assert results == {"value": 1.5}


def test_read_json_rejects_missing_provenance(tmp_path) -> None:
    """A bare JSON file is rejected, which is how orphaned artifacts are caught."""
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"value": 1}), encoding="utf-8")
    with pytest.raises(KeyError):
        read_json(path)


@pytest.mark.parametrize("name", ARTIFACTS)
def test_every_artifact_has_provenance(name: str, repo_root: Path) -> None:
    """Every committed artifact carries a complete provenance block."""
    path = repo_root / PROCESSED / name
    if not path.exists():
        pytest.skip(f"missing {path}; run `make all` first")
    meta, _results = read_json(path)
    assert meta["config_digest"]
    assert meta["code_digest"]
    assert meta["versions"]["python"]


@pytest.mark.parametrize("name", ARTIFACTS)
def test_artifacts_contain_no_formatted_strings(name: str, repo_root: Path) -> None:
    """Result payloads hold numbers, never ``"8.58%"``-style strings.

    This is the invariant that makes the manuscript reconciliation test possible: a
    pre-formatted string cannot be compared against a manuscript cell.
    """
    path = repo_root / PROCESSED / name
    if not path.exists():
        pytest.skip(f"missing {path}; run `make all` first")
    _meta, results = read_json(path)

    def walk(node) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif isinstance(node, str):
            assert not node.endswith("%"), f"{name} contains a formatted percentage: {node!r}"
            assert "±" not in node, f"{name} contains a formatted dispersion: {node!r}"
            assert not node.startswith("+0."), f"{name} contains a formatted decimal: {node!r}"

    walk(results)


@needs(f"{PROCESSED}/table1.json")
def test_manuscript_tables_are_in_sync(repo_root: Path) -> None:
    """``docs/JOURNAL.md`` matches the regenerated tables, cell for cell.

    This is the test that would have caught the Hist-MV row drifting from 29.34% to
    30.51% between the manuscript and its artifact.
    """
    import os

    cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        blocks = load_blocks(PROCESSED)
        assert not sync_manuscript(blocks, repo_root / "docs/JOURNAL.md", write=False), (
            "docs/JOURNAL.md is out of sync with data/processed; run `make tables`"
        )
    finally:
        os.chdir(cwd)


@needs(f"{PROCESSED}/table1.json")
def test_table_renderers_are_deterministic(repo_root: Path) -> None:
    """Rendering the same artifact twice produces identical Markdown."""
    _meta1, table1 = read_json(repo_root / PROCESSED / "table1.json")
    _meta2, table2 = read_json(repo_root / PROCESSED / "table2.json")
    _meta3, table3 = read_json(repo_root / PROCESSED / "table3.json")
    assert render_table1(table1) == render_table1(table1)
    assert render_table2(table2) == render_table2(table2)
    assert render_table3(table3) == render_table3(table3)


@needs(f"{PROCESSED}/table1.json")
def test_golden_artifact_hashes(repo_root: Path) -> None:
    """The numeric payload of each artifact matches the committed golden hash.

    Regenerate with ``python -m stillwar tables --bless`` after an intentional change.
    """
    golden_path = repo_root / "tests" / "golden.json"
    if not golden_path.exists():
        pytest.skip("no golden file; run the bless step once")

    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    for name, expected in golden.items():
        path = repo_root / PROCESSED / name
        if not path.exists():
            pytest.skip(f"missing {path}")
        _meta, results = read_json(path)
        actual = hashlib.sha256(json.dumps(results, sort_keys=True).encode()).hexdigest()[:16]
        assert actual == expected, f"{name} changed; re-bless if the change is intended"
