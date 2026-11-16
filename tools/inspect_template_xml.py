"""Dump selected OOXML fragments from the journal template."""

from __future__ import annotations

from docx import Document

PATH = r"C:\Users\rainp\Journal\JOURNAL.docx"


def main() -> None:
    doc = Document(PATH)
    print("=== STYLE IDS ===")
    for style in doc.styles:
        print(f"{style.style_id!r:24} {style.name!r:28} type={style.type}")

    print("\n=== EQUATION PARAGRAPH 33 XML ===")
    print(doc.paragraphs[33]._p.xml[:6000])

    print("\n=== TABLE 0 XML (first 4000 chars) ===")
    print(doc.tables[0]._tbl.xml[:4000])

    print("\n=== SECTION XML ===")
    print(doc.sections[0]._sectPr.xml)


if __name__ == "__main__":
    main()
