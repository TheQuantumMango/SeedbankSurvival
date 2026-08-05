# SeedbankSurvival

A seedbank viability assessment tool: fits regression-based deterioration curves
(genus-wide and per-species, falling back to per-origin where species data is
sparse) on GRIN-Global accession data, and produces a ranked table of
regeneration-priority accessions.

The original analysis started as a single class-project notebook
(`notebooks/DATA115_Final4_Charpentier.ipynb`, kept for historical reference).
It's being refactored into a tested, installable package as the project grows
toward a shareable dashboard tool for seedbank curators.

## Data sources

Two input paths, both ending up in the same normalized shape:

- **Reformatted CSV/XLSX** (`data_prep.load_accessions`) -- a pre-cleaned export
  (e.g. a class project's `Astragalus.csv`) already shaped like the pipeline's
  internal columns (`Accession, SeedAge, AgeAtTest, Viability, Status, Species,
  Origin, EstTotalSeed, ...`).
- **Raw GRIN-Global Curator Tool export** (`grin_import.adapt_raw_export`) -- an
  actual raw export (57 raw columns, mixed genera, one row per Inventory/seed
  lot). Filters to the target genus first, drops administrative placeholder
  rows, parses each lot's `Inventory Suffix` into a year + original/increase
  marker, and salvages test results from lots whose own suffix didn't parse a
  year by borrowing a same-Accession sibling lot's year (see the module
  docstring for the full reasoning). Produces two frames: `df_primary` (feeds
  the normal pipeline via the *same, unmodified* `clean_ages` /
  `build_model_dataset` / `build_ranking_dataset`) and `df_borrowed`
  (model-fitting-only "borrowed" test points -- no `SeedAge`, so they can never
  reach the ranking table).

**Known limitation**: the raw export has no wild-collection-date field for
original ("o"-suffix) lots -- only when the lot was added to GRIN. `SeedAge`
for original lots is therefore a lower bound, not exact, and may understate
true seed age for accessions collected long before being formally
accessioned. Fixing this would require a different/additional GRIN export
(accession-level passport data).

## Layout

```
src/seedbank_survival/
    data_prep.py       # load/clean a reformatted export, build model & ranking datasets
    grin_import.py     # adapt a RAW GRIN-Global export into the same shape data_prep.py consumes
    deterioration.py   # fit global and per-group (Species/Origin) OLS deterioration models
    hierarchical.py     # predict current viability: Species -> Origin -> Global, confidence-weighted
    priority.py         # years-to-zero, flagged reasons, ranked priority table
tests/
    legacy_baseline/    # a standalone oracle -- a near-verbatim port of the notebook's logic,
                        # used to prove data_prep.py reproduces original behavior where unchanged,
                        # and to pin down exactly where later fixes deliberately diverge from it
    fixtures/           # small synthetic dataset (old/reformatted schema) exercising every branch
    test_legacy_baseline.py          # characterization tests against the oracle
    test_grin_import.py              # raw-export adapter tests (inline-built rows, no fixture file)
    test_hierarchical_confidence.py  # confidence-weighted tier selection, schema-independent
    test_real_data_smoke.py          # runs the reformatted-CSV pipeline against real local data
    test_raw_grin_export_smoke.py    # same, for the raw-export pipeline
notebooks/
    DATA115_Final4_Charpentier.ipynb   # archived original notebook, not maintained
```

Real accession exports (`docs/*.csv`, `docs/*.xlsx`) are local-only and
gitignored -- they aren't required to run the test suite, only to exercise
the two `*_smoke.py` tests locally.

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

## The priority table

`priority.build_priority_table` returns, per flagged accession: `Accession,
Species, Origin, SeedAge, PrimaryReason, EstimatedViability_2026,
YearsRemainingTo0%, ModelUsed, ModelConfidence`.

`ModelUsed`/`ModelConfidence`: prediction falls back Species -> Origin ->
Global by data availability, but a tier fit on very little data isn't
necessarily a *better* one -- `hierarchical.predict_hierarchical` compares
that tier's R^2 against the Global model's and prefers Global whenever it's
higher (including when the tier's R^2 is undefined -- e.g. a handful of
near-identical Viability values). `ModelConfidence` is the winning tier's R^2.

`build_ranking_dataset` picks, per Accession, the youngest lot that isn't
depleted (no seed on hand, or tested at exactly 0% viable), falling back to
the plain youngest lot only if every candidate is depleted -- an accession
with nothing usable left is exactly what this tool should surface, not
silently drop.

## Fixed bugs (deliberate, tested divergences from the original notebook)

`tests/legacy_baseline/pipeline.py` stays frozen as a "before" oracle;
`tests/test_legacy_baseline.py` asserts each fix's exact, deliberate
divergence from it rather than just re-testing current behavior in isolation:

- **Status whitelist**: the original had `"Low viability"`, which never
  matched real data (the actual value is `"Low germination"`), and omitted
  most other real statuses. `data_prep.DEFAULT_VALID_STATUSES` now covers the
  full real vocabulary, with inline reasoning for every inclusion/exclusion.
- **Depleted-lot ranking fallback** (see above) -- the original always used
  the youngest lot regardless of whether it had seed left.
