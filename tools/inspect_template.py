"""Inspect the journal DOCX template structure and styles."""

from __future__ import annotations

import sys
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt


def describe_style(style) -> str:
    font = style.font
    parts = [f"{style.type}"]
    if font.name:
        parts.append(f"font={font.name}")
    if font.size:
        parts.append(f"size={font.size.pt:.1f}pt")
    if font.bold:
        parts.append("bold")
    if font.italic:
        parts.append("italic")
    if font.color and font.color.rgb:
        parts.append(f"color={font.color.rgb}")
    pf = style.paragraph_format
    if pf.alignment is not None:
        parts.append(f"align={pf.alignment}")
    if pf.space_before is not None:
        parts.append(f"before={pf.space_before.pt:.1f}pt")
    if pf.space_after is not None:
        parts.append(f"after={pf.space_after.pt:.1f}pt")
    if pf.line_spacing is not None:
        parts.append(f"line={pf.line_spacing}")
    if pf.first_line_indent is not None:
        parts.append(f"indent={pf.first_line_indent.pt:.1f}pt")
    return ", ".join(parts)


def main(path: str) -> None:
    doc = Document(path)
    print("=== SECTIONS ===")
    for i, section in enumerate(doc.sections):
        print(
            f"[{i}] page={section.page_width.pt:.1f}x{section.page_height.pt:.1f}pt "
            f"orient={section.orientation} "
            f"margins L/R/T/B={section.left_margin.pt:.1f}/{section.right_margin.pt:.1f}/"
            f"{section.top_margin.pt:.1f}/{section.bottom_margin.pt:.1f} "
            f"header/footer={section.header_distance.pt:.1f}/{section.footer_distance.pt:.1f}"
        )
        print(f"    header: {section.header.is_linked_to_previous=} text={[p.text for p in section.header.paragraphs]}")
        print(f"    footer: {section.footer.is_linked_to_previous=} text={[p.text for p in section.footer.paragraphs]}")

    print("\n=== STYLES (used in document) ===")
    used = []
    for p in doc.paragraphs:
        if p.style.name not in used:
            used.append(p.style.name)
    for name in used:
        style = doc.styles[name]
        print(f"{name!r}: {describe_style(style)}")

    print("\n=== PARAGRAPHS ===")
    for i, p in enumerate(doc.paragraphs):
        text = p.text.replace("\n", "\\n")
        if len(text) > 160:
            text = text[:157] + "..."
        runs = []
        for r in p.runs[:4]:
            fmt = []
            if r.bold:
                fmt.append("B")
            if r.italic:
                fmt.append("I")
            if r.font.size:
                fmt.append(f"{r.font.size.pt:.0f}pt")
            if r.font.name:
                fmt.append(r.font.name)
            runs.append(f"{r.text[:30]!r}{'[' + ','.join(fmt) + ']' if fmt else ''}")
        print(f"[{i:03d}] style={p.style.name!r} align={p.alignment} text={text!r}")
        if runs:
            print(f"      runs: {' | '.join(runs)}")

    print("\n=== TABLES ===")
    for ti, table in enumerate(doc.tables):
        print(f"[{ti}] rows={len(table.rows)} cols={len(table.columns)} style={table.style.name if table.style else None}")
        for ri, row in enumerate(table.rows[:4]):
            cells = [c.text.replace("\n", " ")[:40] for c in row.cells]
            print(f"    row {ri}: {cells}")

    print("\n=== CORE PROPERTIES ===")
    cp = doc.core_properties
    for attr in ("title", "subject", "author", "last_modified_by", "created", "modified", "revision"):
        print(f"{attr}: {getattr(cp, attr, None)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\rainp\Journal\JOURNAL.docx")
