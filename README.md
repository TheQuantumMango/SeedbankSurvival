# SeedbankSurvival

A seedbank viability assessment tool: fits regression-based deterioration curves
(genus-wide and per-species, falling back to per-origin where species data is
sparse) on GRIN-Global accession data, and produces two views of the result --
which accessions to prioritize for regeneration, and which physical packets to
test or discard.

The original analysis started as a single class-project notebook
(`notebooks/DATA115_Final4_Charpentier.ipynb`, kept for historical reference).
It's now a tested, installable package with a CLI that runs the whole pipeline
end-to-end.

## Quick start

```bash
seedbank-survival --inventory RawGRINExport.xlsx --genus Astragalus
```

(or `python -m seedbank_survival ...` if the console script isn't on PATH).
Not sure what genera are in the file? Run `--list-genera` first:

```bash
seedbank-survival --inventory RawGRINExport.xlsx --list-genera
```

This prints every distinct first-word-of-Taxon value with its row count --
raw exports routinely mix in other genera, companion species, and
administrative placeholders (e.g. `Undetermined nlgrp-backup`), so genus
selection is always explicit, never guessed. `--genus`/`-g` is repeatable
(e.g. current name + a synonymous former one spanning the same data).

Add `--accessions RawGRINAccessions.xlsx` (the accession-level export, one
row per Accession) to improve age coverage -- see below. Full options:
`seedbank-survival --help`.

Output, written to `--out-dir` (default: current directory):
`accession_view.csv`, `inventory_view.csv`, `report.html` (self-contained,
open it directly in any browser -- no server, works on a locked-down machine
with no admin install rights beyond what running the tool itself needs).

## Data sources

Two input paths, both ending up in the same normalized shape:

- **Reformatted CSV/XLSX** (`data_prep.load_accessions`) -- a pre-cleaned export
  (e.g. a class project's `Astragalus.csv`) already shaped like the pipeline's
  internal columns (`Accession, SeedAge, AgeAtTest, Viability, Status, Species,
  Origin, EstTotalSeed, ...`). Usable programmatically; not exposed by the CLI.
- **Raw GRIN-Global Curator Tool export** (`grin_import.adapt_raw_export`) -- an
  actual raw export (57 raw columns, mixed genera, one row per Inventory/seed
  lot). Filters to the chosen genus/genera first, drops administrative
  placeholder rows, parses each lot's `Inventory Suffix` into a year +
  original/increase marker, and salvages test results from lots whose own
  suffix didn't parse a year by borrowing a same-Accession sibling lot's year
  (see the module docstring for the full reasoning). Produces two frames:
  `df_primary` (feeds the normal pipeline via the *same, unmodified*
  `clean_ages` / `build_model_dataset` / `build_ranking_dataset`) and
  `df_borrowed` (model-fitting-only "borrowed" test points -- no `SeedAge`, so
  they can never reach either view).

### Optional accession-level file (`--accessions`)

A second raw export, one row per Accession (not per lot), whose `Received
Date` fills a *second, independent* SeedAge fallback: own suffix -> Received
Date (unconditional, every row) -- separate from `AgeAtTest`'s chain (own
suffix -> sibling-year, test-event-anchored, used for model fitting only).
A Received Date has no test event to anchor a plausibility check against, so
it isn't a substitute for sibling-year resolution -- only for "how old is
this lot today," which needs no such anchor. Rows resolved this way land in
`df_primary` (view-eligible), not `df_borrowed`, since the goal is never
silently dropping an accession that has real seed. Verified against real
data: SeedAge resolution goes from 72% to 100% of rows with this file
supplied.

**Known limitation**: neither raw export has a wild-collection-date field for
original (`"o"`-suffix) lots -- only when the lot was added to GRIN (Received
Date) or the suffix's own encoded year. `SeedAge` for original lots is
therefore a lower bound, not exact, and may understate true seed age for
accessions collected long before being formally accessioned. Fixing this
would require a different/additional GRIN export (true passport/collecting
data).

## The two views

- **Accession view** (`data_prep.build_ranking_dataset`) -- one row per
  Accession: the best-representative lot, for deciding what to regenerate.
  Included if the lot's Status is whitelisted (`DEFAULT_VALID_STATUSES`) OR
  it has real seed on hand (`EstTotalSeed > 0`) -- Status text can be stale
  or wrong in ways a nonzero quantity value isn't. Prefers the youngest
  non-depleted lot, falling back to the plain youngest lot only if every
  candidate for that accession is depleted, so an accession that's
  completely out of usable seed still gets surfaced, not silently dropped.
- **Inventory view** (`data_prep.build_inventory_view`) -- one row per
  physical packet with seed on hand (`EstTotalSeed > 0`), Status-independent,
  no per-Accession dedup -- for deciding which packets to test or discard.
  A packet predicted at ~0% viability still appears; that's exactly the kind
  of packet this view exists to surface.

Both views get `PredictedViability_2026` / `ModelUsed` / `ModelConfidence`
via the same `hierarchical.predict_hierarchical` and the same row-flagging
(`priority.build_priority_table`, called with `top_n` = every row -- neither
view is capped; the report's checkbox/filters are how you narrow what you're
looking at, not an export-time cutoff).

## Layout

```
src/seedbank_survival/
    data_prep.py        # load/clean a reformatted export; build both views
    grin_import.py       # adapt a RAW GRIN-Global inventory export into data_prep.py's shape
    grin_accessions.py    # Received Date year lookup from the accession-level export
    deterioration.py     # fit global and per-group (Species/Origin) OLS deterioration models
    hierarchical.py       # predict current viability: Species -> Origin -> Global, confidence-weighted
    priority.py           # years-to-zero, flagged reasons, the shared table-flagging logic
    report.py             # self-contained HTML report (both views + charts), no server
    cli.py, __main__.py   # seedbank-survival / python -m seedbank_survival
tests/
    legacy_baseline/     # a standalone oracle -- a near-verbatim port of the notebook's logic,
                         # used to prove data_prep.py reproduces original behavior where unchanged,
                         # and to pin down exactly where later fixes deliberately diverge from it
    fixtures/            # small synthetic dataset (old/reformatted schema) exercising every branch
    test_legacy_baseline.py, test_data_prep.py, test_grin_import.py, test_grin_accessions.py,
    test_hierarchical_confidence.py, test_report.py, test_cli.py    # inline-built rows, no fixture files
    test_real_data_smoke.py, test_raw_grin_export_smoke.py, test_cli_smoke.py   # real local data, self-skipping
notebooks/
    DATA115_Final4_Charpentier.ipynb   # archived original notebook, not maintained
```

Real accession exports (`docs/*.csv`, `docs/*.xlsx`) are local-only and
gitignored -- they aren't required to run the test suite, only to exercise
the `*_smoke.py` tests locally.

## Dev setup

```bash
python -m venv .venv
.venv/Scripts/pip install -e ".[dev]"    # Windows; use .venv/bin/pip on macOS/Linux
```

`requirements.txt` is a pinned lockfile of a known-good, all-wheels environment
(useful for reproducing an exact working setup); `pyproject.toml` declares the
package's actual dependency floors.

## Running tests

```bash
.venv/Scripts/python -m pytest
```

## Fixed bugs (deliberate, tested divergences from the original notebook)

`tests/legacy_baseline/pipeline.py` stays frozen as a "before" oracle;
`tests/test_legacy_baseline.py` asserts each fix's exact, deliberate
divergence from it rather than just re-testing current behavior in isolation:

- **Status whitelist**: the original had `"Low viability"`, which never
  matched real data (the actual value is `"Low germination"`), and omitted
  most other real statuses. `data_prep.DEFAULT_VALID_STATUSES` now covers the
  full real vocabulary, with inline reasoning for every inclusion/exclusion.
- **Depleted-lot ranking fallback** -- the original always used the youngest
  lot regardless of whether it had seed left.
- **Real-seed-or-whitelisted-status broadening** -- the original only ever
  looked at Status text for the Accession view.
