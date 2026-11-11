"""Compatibility shim for the old ``stillwar.provenance`` import path.

New code should import from :mod:`stillwar.provenance.metadata`.
"""

from stillwar.provenance.metadata import (
    artifact_metadata,
    code_digest,
    library_versions,
    read_json,
    write_json,
)

__all__ = [
    "artifact_metadata",
    "code_digest",
    "library_versions",
    "read_json",
    "write_json",
]
