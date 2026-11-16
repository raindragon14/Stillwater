"""Inspect the template's table style and cell formatting."""

from __future__ import annotations

from docx import Document

PATH = r"C:\Users\rainp\Journal\JOURNAL.docx"


def main() -> None:
    doc = Document(PATH)
    table = doc.tables[0]
    print("=== TABLE 0 PROPERTIES ===")
    print(table._tbl.tblPr.xml)
    print("\n=== FIRST ROW CELL PARAGRAPHS ===")
    for cell in table.rows[0].cells[:3]:
        for p in cell.paragraphs:
            print(f"style={p.style.name!r} text={p.text!r} align={p.alignment}")
            print(p._p.xml[:1200])
    print("\n=== SECOND ROW CELL PARAGRAPHS ===")
    for cell in table.rows[1].cells[:2]:
        for p in cell.paragraphs:
            print(f"style={p.style.name!r} text={p.text!r} align={p.alignment}")
    print("\n=== THREE-LINE TABLE STYLE XML ===")
    style = doc.styles["Three-line Table"]
    print(style.element.xml[:5000])


if __name__ == "__main__":
    main()
