#!/usr/bin/env python3
"""Structural checks for docs/JOURNAL.md.

The manuscript is the source of truth.  This script fails if an editorial
placeholder survives, if a Markdown table is ragged, if a reference is missing or
uncited, or if a figure referenced by the manuscript is absent.

Run from the repository root:

    python tools/check_manuscript.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "docs" / "JOURNAL.md"
REFERENCES_HEADING = "# DAFTAR PUSTAKA"

PLACEHOLDERS = (
    "[RESULT",
    "[CONFIRM",
    "[IF ",
    "[check",
    "[verify",
    "[ ]",
    "[X]",
    "[Y]",
    "[email address]",
    "[merugi/untung]",
    "[lost/gained]",
)


def check_placeholders(text: str, errors: list[str]) -> None:
    for token in PLACEHOLDERS:
        count = text.count(token)
        if count:
            errors.append(f"placeholder {token!r} appears {count} time(s)")


def check_tables(text: str, errors: list[str]) -> None:
    header_cols: int | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped.startswith("|"):
            header_cols = None
            continue
        cells = stripped.strip("|").split("|")
        if header_cols is None:
            header_cols = len(cells)
        elif all(set(cell.strip()) <= set("-: ") for cell in cells):
            continue  # separator row
        elif len(cells) != header_cols:
            errors.append(
                f"line {lineno}: table row has {len(cells)} columns, expected {header_cols}"
            )


def check_references(text: str, errors: list[str]) -> None:
    if REFERENCES_HEADING not in text:
        errors.append(f"missing reference heading {REFERENCES_HEADING!r}")
        return
    body, references = text.split(REFERENCES_HEADING, 1)
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\]", references, flags=re.MULTILINE)]
    if not numbers:
        errors.append("no numbered references found")
        return
    expected = list(range(1, max(numbers) + 1))
    missing = sorted(set(expected) - set(numbers))
    if missing:
        errors.append(f"missing reference entries: {missing}")
    cited: set[int] = set()
    for match in re.finditer(r"\[([0-9][0-9,\s]*)\]", body):
        cited.update(int(n) for n in re.findall(r"\d+", match.group(1)))
    uncited = sorted(set(numbers) - cited)
    if uncited:
        errors.append(f"reference entries never cited: {uncited}")


def check_figures(text: str, errors: list[str]) -> None:
    for match in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", text):
        target = (MANUSCRIPT.parent / match.group(1)).resolve()
        if not target.is_file():
            errors.append(f"figure not found: {match.group(1)}")


def check_equations(text: str, errors: list[str]) -> None:
    if text.count("$$") % 2:
        errors.append("unbalanced display-math delimiters ($$)")
    tags = [int(n) for n in re.findall(r"\\tag\{(\d+)\}", text)]
    expected = list(range(1, len(tags) + 1))
    if tags != expected:
        errors.append(f"equation tags are not sequential: {tags}")


def main() -> int:
    if not MANUSCRIPT.is_file():
        print(f"ERROR: manuscript not found at {MANUSCRIPT}", file=sys.stderr)
        return 1
    text = MANUSCRIPT.read_text(encoding="utf-8")
    errors: list[str] = []
    check_placeholders(text, errors)
    check_tables(text, errors)
    check_references(text, errors)
    check_figures(text, errors)
    check_equations(text, errors)

    if errors:
        print(f"FAILED: {len(errors)} issue(s) in {MANUSCRIPT.relative_to(ROOT)}")
        for error in errors:
            print(f"  - {error}")
        return 1

    lines = len(text.splitlines())
    words = len(text.split())
    print(f"OK: {MANUSCRIPT.relative_to(ROOT)} ({lines} lines, {words} words)")
    print("    no placeholders, tables rectangular, references complete and cited, figures present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
