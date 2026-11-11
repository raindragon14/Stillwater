"""Data package: universe, ingestion, manifest and aligned panel."""

from stillwar.data.ingest import download, verify_manifest, write_manifest
from stillwar.data.panel import OHLC, Panel, load_panel, select_universe
from stillwar.data.universe import CANDIDATES, IDX_SECTOR_MAPPING, SECTORS

__all__ = [
    "CANDIDATES",
    "IDX_SECTOR_MAPPING",
    "OHLC",
    "Panel",
    "SECTORS",
    "download",
    "load_panel",
    "select_universe",
    "verify_manifest",
    "write_manifest",
]
