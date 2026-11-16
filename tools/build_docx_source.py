"""Build the DOCX-specific Markdown source from the manuscript source of truth.

The journal template supplies the styles, so the front matter is expressed as
Pandoc custom-style divs and the body is the manuscript from ``# 1. INTRODUCTION``
onward. Figure images become placeholders that the DOCX post-processor replaces
with the actual PNGs.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "docs" / "JOURNAL.md"
OUT = ROOT / "build" / "journal_docx.md"


def main() -> None:
    lines = SOURCE.read_text(encoding="utf-8").splitlines()
    title = lines[0].removeprefix("# ").strip()
    authors = lines[2].strip()
    affil1 = lines[5].strip()
    affil2 = lines[7].strip()
    corr = lines[9].strip()
    abstract_id = lines[13].strip()
    keywords_id = lines[15].strip()
    abstract_en = lines[19].strip()
    keywords_en = lines[21].strip()

    body_start = next(i for i, line in enumerate(lines) if line.startswith("# 1. INTRODUCTION"))
    body = lines[body_start:]

    # Unicode superscripts in the author line become real superscript runs.
    authors = authors.replace("¹,\\*", "^1,\\*^").replace("²", "^2^")
    # Escape the affiliation numbering so Pandoc does not read it as a list.
    affil1 = re.sub(r"^\((\d)\)", r"\(\1\)", affil1)
    affil2 = re.sub(r"^\((\d)\)", r"\(\1\)", affil2)
    corr_label, corr_email = corr.split(": ", 1)

    cleaned: list[str] = []
    for line in body:
        if line.startswith("<!-- BEGIN TABLE") or line.startswith("<!-- END TABLE"):
            continue
        if line.startswith("![Figure 1:"):
            cleaned.append("[[FIGURE_1]]")
            continue
        if line.startswith("![Figure 2:"):
            cleaned.append("[[FIGURE_2]]")
            continue
        if line.startswith("(ii)") or line.startswith("(iii)"):
            line = "\\" + line
        cleaned.append(re.sub(r"\\tag\{\d+\}", "", line))

    front = f"""::: {{custom-style="Title"}}
{title}
:::

::: {{custom-style="Author"}}
{authors}
:::

::: {{custom-style="Affiliation"}}
{affil1}
:::

::: {{custom-style="Affiliation"}}
{affil2}
:::

::: {{custom-style="Correspondence"}}
{corr_label}
:::

::: {{custom-style="CorrespondenceLast"}}
({corr_email})
:::

::: {{custom-style="AbstractHeading"}}
Abstrak
:::

::: {{custom-style="AbstractText"}}
{abstract_id}
:::

::: {{custom-style="Keywords"}}
{keywords_id}
:::

::: {{custom-style="AbstractHeadingEnglish"}}
Abstract
:::

::: {{custom-style="AbstractTextEnglish"}}
{abstract_en}
:::

::: {{custom-style="KeywordsEnglish"}}
{keywords_en}
:::

"""

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(front + "\n".join(cleaned) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(cleaned)} body lines)")


if __name__ == "__main__":
    main()
