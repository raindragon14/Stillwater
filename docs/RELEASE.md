# Public release checklist

This checklist prepares the repository for a public GitHub release and a Zenodo
archive. It assumes the repository is renamed to **Stillwater**; the old `Stillwar`
URL redirects on GitHub.

## 1. Final local checks

```bash
git switch main
git pull --ff-only                   # single-branch workflow: main only
uv sync --extra dev
uv lock --check
make check                           # ruff + pytest + manuscript check
make typecheck
make all                             # regenerate tables and figures
git diff --exit-code                 # the generated artifacts must be committed
```

Optional, if Docker is available:

```bash
docker build -t stillwater:2.0.0 .
docker run --rm stillwater:2.0.0     # runs `python -m stillwater verify`
```

Optional package build:

```bash
uv build
ls dist/
```

## 2. GitHub repository settings

- Rename the repository to `Stillwater` (Settings → General → Repository name).
- Description: *Reproducible code and manuscript for “When the Optimizer Refuses to
  Trade: A KKT No-Trade Zone and Lévy-Area Signal Activation under Weak Signals in LQ45
  Equities”.*
- Topics: `reproducible-research`, `portfolio-optimization`, `log-signature`,
  `kkt-conditions`, `transaction-costs`, `idx`, `lq45`.
- Enable Issues; Discussions optional.
- Protect `main`: require the CI workflow and one review before merge.
- Enable Dependabot alerts and CodeQL if the repository is public.

## 3. Tag and GitHub release

```bash
git tag -a v2.0.0 -m "Stillwater v2.0.0"
git push origin main
git push origin v2.0.0
```

Create a GitHub release from `v2.0.0` with:

- Title: `Stillwater v2.0.0`
- Notes: copy the `2.0.0` section from [`CHANGELOG.md`](../CHANGELOG.md).
- Assets:
  - `raw_snapshot_v2.0.0.zip` containing `data/raw/` (51 CSVs + `manifest.sha256`).
  - `SHA256SUMS` for the assets.
  - Optional: `stillwater-2.0.0-py3-none-any.whl` and `.tar.gz` from `uv build`.

## 4. Zenodo archive and DOI

1. Connect the GitHub repository to Zenodo.
2. Archive the `v2.0.0` release.
3. Add the DOI to `CITATION.cff`:

   ```yaml
   doi: 10.5281/zenodo.XXXXXXX
   ```

4. Add the DOI badge/link to `README.md` and `docs/REVISION_NOTES.md`.
5. Re-run `make check` and commit the DOI update.

## 5. Manuscript and metadata

- Confirm the target journal and regenerate `docs/JOURNAL.docx` with its template
  (the Markdown is the source of truth; the DOCX is ignored by Git).
- Confirm the author list, affiliations, corresponding email, CRediT contributions,
  funding, competing interests, and AI-use disclosure.
- Check reference [20] for a later journal version.
- Decide whether to include the supplementary figures
  (`reports/figures/figS1_architecture.*`, `figS2_sensitivity.*`); they are generated
  locally and ignored by Git.

## 6. Post-release

- Verify the README links and the CI badge on a fresh clone.
- Verify `uv sync --frozen --extra dev` and `make check` on a clean machine.
- Announce the release with the paper title, DOI, and repository URL.
