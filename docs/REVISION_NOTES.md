# Revision notes

Open items before submission. The full draft audit is preserved in Git history
(commit `d48afd9`, `docs/EVALUATION.md`).

1. **Target journal and template.** The old DOCX was formatted for *Journal of
   Education Research*, which is not the right field. The Markdown is journal-agnostic;
   regenerate the DOCX with the target journal's template.
2. **Authors and corresponding email.** The draft lists one author; confirm the final
   author list, order, affiliations and corresponding email.
3. **Declarations.** Add CRediT contributions, funding, competing interests,
   data/code availability, and the journal's AI-use disclosure if required.
4. **Supplementary figures.** `make figures` also generates
   `reports/figures/figS1_architecture.*` and `figS2_sensitivity.*`; they are ignored by
   Git. Decide whether to include them as supplementary material.
5. **Alpha vs GMV and per-seed p-values.** Not present in the archived artifacts. The
   manuscript reports alpha vs EW and per-seed means/s.d.; compute the missing
   statistics if the journal requires them.
6. **Repository URL.** The code/data release is
   `https://github.com/raindragon14/Stillwar` (`v1.0.0`). Update it if the manuscript
   repository is published under a different name.
7. **Reference [20].** Still an SSRN working paper; check for a later journal version
   before submission.
