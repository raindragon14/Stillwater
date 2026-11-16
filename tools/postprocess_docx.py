"""Post-process the Pandoc DOCX into the journal template's layout.

Pandoc supplies native OMML equations and the body content; this script applies
the template's paragraph styles, converts display equations to the template's
tabbed Equation layout, rebuilds the three-line tables, inserts the figures and
styles the captions and references.
"""

from __future__ import annotations

from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, Twips

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "build" / "journal_body.docx"
OUT = ROOT / "build" / "JOURNAL_Stillwater.docx"

FRONT_STYLES = {
    "Title",
    "Author",
    "Affiliation",
    "Correspondence",
    "CorrespondenceLast",
    "AbstractHeading",
    "AbstractText",
    "Keywords",
    "AbstractHeadingEnglish",
    "AbstractTextEnglish",
    "KeywordsEnglish",
}

TABLE_GRIDS = {
    0: [2003, 1057, 1057, 1057, 1057, 1057, 1057, 1057],
    1: [2821, 1097, 1097, 1097, 1097, 1097, 1097],
    2: [3762, 1881, 1881, 1881],
}


def set_table_width(table, width_twips: int) -> None:
    tbl_pr = table._tbl.tblPr
    tbl_w = tbl_pr.find(qn("w:tblW"))
    if tbl_w is None:
        tbl_w = OxmlElement("w:tblW")
        tbl_pr.append(tbl_w)
    tbl_w.set(qn("w:w"), str(width_twips))
    tbl_w.set(qn("w:type"), "dxa")

    layout = tbl_pr.find(qn("w:tblLayout"))
    if layout is None:
        layout = OxmlElement("w:tblLayout")
        tbl_pr.append(layout)
    layout.set(qn("w:type"), "fixed")


def set_grid(table, widths: list[int]) -> None:
    grid = table._tbl.tblGrid
    for grid_col, width in zip(grid.gridCol_lst, widths, strict=True):
        grid_col.set(qn("w:w"), str(width))
    for row in table.rows:
        for cell, width in zip(row.cells, widths):
            cell.width = Twips(width)


def merge_group_row(table, row_idx: int) -> None:
    row = table.rows[row_idx]
    text = row.cells[0].text.strip()
    if not text:
        return
    merged = row.cells[0].merge(row.cells[-1])
    for paragraph in list(merged._tc.findall(qn("w:p"))):
        merged._tc.remove(paragraph)
    paragraph = OxmlElement("w:p")
    p_pr = OxmlElement("w:pPr")
    p_style = OxmlElement("w:pStyle")
    p_style.set(qn("w:val"), "TableGroup")
    p_pr.append(p_style)
    paragraph.append(p_pr)
    run = OxmlElement("w:r")
    text_el = OxmlElement("w:t")
    text_el.set(qn("xml:space"), "preserve")
    text_el.text = text
    run.append(text_el)
    paragraph.append(run)
    merged._tc.append(paragraph)
    merged.width = Twips(9402)


def style_cell(cell, style_name: str) -> None:
    paragraphs = cell.paragraphs
    if not paragraphs:
        cell.add_paragraph()
        paragraphs = cell.paragraphs
    for paragraph in paragraphs:
        paragraph.style = style_name


def is_group_row(row) -> bool:
    first = row.cells[0].text.strip()
    if not (first.startswith("Panel A") or first.startswith("Panel B")):
        return False
    return all(not cell.text.strip() for cell in row.cells[1:])


def convert_equation(paragraph, number: int) -> bool:
    math_para = paragraph._p.find(qn("m:oMathPara"))
    if math_para is None:
        return False
    math = math_para.find(qn("m:oMath"))
    if math is None:
        return False
    new_paragraph = paragraph.insert_paragraph_before(style="Equation")
    run = new_paragraph.add_run()
    run.add_tab()
    math_para.remove(math)
    new_paragraph._p.append(math)
    run = new_paragraph.add_run()
    run.add_tab()
    new_paragraph.add_run(f"({number})")
    old = paragraph._p
    old.getparent().remove(old)
    return True


def remove_bold(paragraph) -> None:
    for run in paragraph.runs:
        run.bold = False


def replace_figure(paragraph, image_path: Path) -> None:
    for run in list(paragraph.runs):
        run._r.getparent().remove(run._r)
    paragraph.style = "Normal"
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.space_before = Pt(6)
    paragraph.paragraph_format.space_after = Pt(3)
    paragraph.paragraph_format.keep_with_next = True
    run = paragraph.add_run()
    run.add_picture(str(image_path), width=Inches(6.5))


def main() -> None:
    doc = Document(SRC)
    paragraphs = list(doc.paragraphs)

    equation_number = 0
    after_equation = False
    in_references = False

    for paragraph in paragraphs:
        style_name = paragraph.style.name
        text = paragraph.text.strip()

        if style_name in FRONT_STYLES:
            continue

        if style_name.startswith("Heading"):
            if text == "DAFTAR PUSTAKA":
                in_references = True
            elif text.startswith("APPENDIX"):
                in_references = False
            after_equation = False
            continue

        if paragraph._p.find(qn("m:oMathPara")) is not None:
            equation_number += 1
            convert_equation(paragraph, equation_number)
            after_equation = True
            continue

        if text.startswith("[[FIGURE_1]]"):
            replace_figure(paragraph, ROOT / "reports" / "figures" / "fig1_kkt_spread.png")
            after_equation = False
            continue
        if text.startswith("[[FIGURE_2]]"):
            replace_figure(paragraph, ROOT / "reports" / "figures" / "fig2_trigger_decomposition.png")
            after_equation = False
            continue

        if text.startswith("Table ") and (text.startswith("Table 1.") or text.startswith("Table 2.") or text.startswith("Table 3.")):
            paragraph.style = "TableCaption"
            remove_bold(paragraph)
            after_equation = False
            continue

        if text.startswith("Figure 1.") or text.startswith("Figure 2."):
            paragraph.style = "Caption"
            remove_bold(paragraph)
            after_equation = False
            continue

        if text.startswith("Notes:") or text.startswith("*Notes:*"):
            paragraph.style = "BodyTextContinued"
            for run in paragraph.runs:
                run.font.size = Pt(8)
            after_equation = False
            continue

        if in_references:
            paragraph.style = "Reference"
            after_equation = False
            continue

        paragraph.style = "BodyTextContinued" if after_equation else "Body Text"
        after_equation = False

    for index, table in enumerate(doc.tables):
        table.style = doc.styles["Three-line Table"]
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        set_table_width(table, 9402)
        widths = TABLE_GRIDS[index]
        set_grid(table, widths)

        group_rows = [row_idx for row_idx, row in enumerate(table.rows) if is_group_row(row)]
        for row_idx in group_rows:
            merge_group_row(table, row_idx)

        for row_idx, row in enumerate(table.rows):
            if row_idx in group_rows:
                continue
            for col_idx, cell in enumerate(row.cells):
                style_cell(cell, "TableTextFirst" if col_idx == 0 else "TableText")

    core = doc.core_properties
    core.title = (
        "When the Optimizer Refuses to Trade: A KKT No-Trade Zone and Lévy-Area "
        "Signal Activation under Weak Signals in LQ45 Equities"
    )
    core.author = "Muhammad Reihan Pandanarang"
    core.subject = "KKT no-trade zone, Lévy areas, LQ45, transaction costs"
    core.keywords = "KKT conditions; no-trade region; log-signature; transaction costs; LQ45; emerging markets"

    doc.save(OUT)
    print(f"Wrote {OUT}")
    print(f"equations numbered: {equation_number}")
    print(f"tables: {len(doc.tables)}")
    print(f"inline shapes: {len(doc.inline_shapes)}")


if __name__ == "__main__":
    main()
