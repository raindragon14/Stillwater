"""Compatibility shim for the old ``stillwar.universe`` import path.

New code should import from :mod:`stillwar.data.universe`.
"""

from stillwar.data.universe import CANDIDATES, IDX_SECTOR_MAPPING, SECTORS

__all__ = ["CANDIDATES", "IDX_SECTOR_MAPPING", "SECTORS"]
