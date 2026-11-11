"""Module entry point for ``python -m stillwater``.

The implementation lives in :mod:`stillwater.cli`; this wrapper keeps the historical
``stillwater.__main__`` import path working for tests and for the console script.
"""

from stillwater.cli import ALL_STAGES, STAGES, main

__all__ = ["ALL_STAGES", "STAGES", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
