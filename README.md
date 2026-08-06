# SeedbankSurvival

A seedbank viability assessment tool: fits quadratic deterioration curves
(genus-wide and per-species, falling back to per-origin where species data is
sparse) on GRIN-Global accession data, and produces two views of the result --
which accessions to prioritize for regeneration, and which physical packets to
test or discard.

The original analysis started as a single class-project notebook
(`notebooks/DATA115_Final4_Charpentier.ipynb`, kept for historical reference).
It's now a tested, installable package with a CLI that runs the whole pipeline
end-to-end.

## Quick start

Easiest: double-click **`run_report.bat`** (Windows). It runs the CLI against
`docs/RawGRINAstragalusExport.xlsx` (+ the accessions file, if present) with
`--genus Astragalus`, writes into `docs/output/`, and opens `report.html`
automatically. Edit the `INVENTORY`/`ACCESSIONS`/`GENUS` lines near the top of
the file if you're working with different data.

From the command line:

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

Each table in `report.html` is sortable (click a header) and filterable.
Column widths are drag-resizable (grab a header's right edge), and the
"Columns ▾" menu toggles which columns are shown -- both independent per
view (Accession/Inventory) and reset on reload; nothing is saved back to disk.

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

Both views also carry `MaintenanceSite` (the raw export's `Inventory
Maintenance Site` -- e.g. "W6", "NLGRP", "DLEG") and `Location` (a
comma-joined `Location Section 1`-`4`, blank sections skipped -- where in the
facility a packet physically is). Both come from the raw-GRIN-export path
only (`grin_import.adapt_raw_export`); the older reformatted CSV/XLSX path
has no per-inventory location data, so `build_priority_table` fills both
columns blank there rather than requiring them. The report's "On-site only
(W6, Pullman)" checkbox filters on `MaintenanceSite`; `Location` is a plain
display column, shown far right in both table views.

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
- **Inherited-test-result double counting** -- a dated lot's viability test
  is routinely re-recorded on a same-Accession "Backup germplasm" row a few
  *hours* later the same day (an administrative echo of one physical test,
  not an independent measurement). The existing duplicate guard in
  `grin_import._build_borrowed_rows` matched on exact Tested Date timestamp
  and missed nearly all of these -- confirmed against real data that ~21% of
  the whole model-fitting dataset (162 accessions) was double-counted this
  way before matching on calendar day instead.
- **Statistically insignificant tier confidence** -- `hierarchical.py`'s
  Species/Origin-over-Global override compared R² alone. A group fit on very
  few points (e.g. n=3, 1 residual degree of freedom) routinely has a
  deceptively high R² by chance; confirmed against real data that most
  Origin-tier models winning on R² alone had a slope statistically
  indistinguishable from zero (95% CI spanning 0). `SlopeModel` (now
  `CurveModel`, see below) also carries `overall_pvalue`, and a tier must be
  significant (p<0.05) *in addition to* beating Global's R² to be trusted
  over it.
- **Linear deterioration curve, and an untested-viability sentinel counted as
  real data** -- two related fixes. Binning real Astragalus data by age
  showed viability holding roughly flat for ~40 years, then dropping sharply
  -- a shape a straight line can't represent. Checked several standard
  seed-science alternatives (probit-linear/Ellis-Roberts, logit-linear,
  exponential decay) against real data, compared fairly on the original
  percentage scale (not the transformed one, which isn't directly
  comparable) -- all of them fit *worse* than a straight line; only a
  quadratic (`Viability ~ AgeAtTest + AgeAtTest²`, now `deterioration.py`'s
  `CurveModel`, replacing `SlopeModel`) improved on it. While comparing
  forms, also found `Percent Viable == -1.0` (never any other negative
  value) is a GRIN sentinel for "no valid test," not a real -1% measurement
  -- was being fit as real data by every model, linear included. Any
  negative Percent Viable is now treated as untested at ingestion.
  `fit_group_models`'s `min_n` default rose from 3 to 4: a quadratic fit
  needs 3 parameters, so n=3 has zero residual degrees of freedom and is
  always a trivial, meaningless r2=1.0 fit regardless of the data.
- **Extrapolating from the population curve instead of an inventory's own
  test result** -- `PredictedViability_2026` was always computed from a
  tier's fitted intercept (the population-average starting point), even for
  a packet with its own real measured Viability. A row with its own
  Viability + AgeAtTest now extrapolates from that specific measurement by
  however much the selected tier's curve itself changes over the interval,
  instead of the population's average starting point. `PrimaryReason` gets
  `"Tested at low viability"` when that real result is itself ≤30%, to
  distinguish a lab-confirmed low result from a merely predicted one.
- **"Low germination" with no recorded percentage** -- a Status (or
  free-text note) documenting a known concern like this shouldn't quietly
  fall back to the tier's average-case curve just because no exact number
  was written down. Such rows now get an assumed 10% viability, anchored to
  the lot's own year (`ViabilityAssumed=True`, `PrimaryReason` gets
  `"Assumed low germination, no test data"`) -- excluded from curve-fitting
  itself so the assumption can't bias the population model. No effect on the
  current real Astragalus data (every such row there already has a real
  percentage recorded), but guards against it for this or another export.

**Known limitation of the quadratic curve**: with very little data (the
n=4 floor), a fitted parabola can take on a visually implausible shape
outside the tight cluster of the actual points -- checked against real
data, this does happen for a handful of the smallest significant groups.
The significance gate still screens out non-trending ones; it doesn't
guarantee a *plausible-shaped* one. Not fixed here -- flagged as a
follow-up if it turns out to matter in practice.
- **Ignoring an inventory's own test result** -- `PredictedViability_2026`
  was always computed from a tier's fitted intercept (the *population*
  average starting point), even for a packet with its own real measured
  Viability. A row with its own Viability + AgeAtTest now extrapolates from
  that specific measurement using the selected tier's slope instead --
  `PrimaryReason` gets `"Tested at low viability"` when that real result is
  itself ≤30%, to distinguish a lab-confirmed low result from a merely
  predicted one. Affects 808/1401 real Astragalus accession-view rows.
- **"Low germination" with no recorded percentage** -- a Status (or
  free-text note) documenting a known concern like this shouldn't quietly
  fall back to the tier's average-case curve just because no exact number
  was written down, understating that accession's regeneration need. Such
  rows now get an assumed 10% viability, anchored to the lot's own year
  (`ViabilityAssumed=True`, `PrimaryReason` gets `"Assumed low germination,
  no test data"`) -- excluded from curve-fitting itself (only affects that
  row's own prediction). No effect on the current real Astragalus data
  (every "Low germination" row there already has a real percentage
  recorded), but guards against it for this or another GRIN export.
