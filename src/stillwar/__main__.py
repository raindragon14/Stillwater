"""Module entry point for ``python -m stillwar``.

The implementation lives in :mod:`stillwar.cli`; this wrapper keeps the historical
``stillwar.__main__`` import path working for tests and for the console script.
"""

from stillwar.cli import ALL_STAGES, STAGES, main

__all__ = ["ALL_STAGES", "STAGES", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
