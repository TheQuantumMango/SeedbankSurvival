# SeedbankSurvival

A seedbank viability assessment tool: fits regression-based deterioration curves
(genus-wide and per-species, falling back to per-origin where species data is
sparse) on GRIN-Global accession data, and produces a ranked table of
regeneration-priority accessions.

The original analysis started as a single class-project notebook
(`notebooks/DATA115_Final4_Charpentier.ipynb`, kept for historical reference).
It's being refactored into a tested, installable package as the project grows
toward a shareable dashboard tool for seedbank curators.

## Layout

```
src/seedbank_survival/
    data_prep.py       # load/clean a raw accession export, build model & ranking datasets
    deterioration.py   # fit global and per-group (Species/Origin) OLS deterioration models
    hierarchical.py     # predict current viability, falling back Species -> Origin -> Global
    priority.py         # years-to-zero, flagged reasons, ranked priority table
tests/
    legacy_baseline/    # a standalone oracle -- a near-verbatim port of the notebook's logic,
                        # used to prove the modules above reproduce current behavior exactly
    fixtures/           # small synthetic dataset exercising every branch/edge case
    test_legacy_baseline.py     # characterization tests against the oracle
    test_real_data_smoke.py     # runs the pipeline against real local data if present (see below)
notebooks/
    DATA115_Final4_Charpentier.ipynb   # archived original notebook, not maintained
```

Real accession exports (`docs/*.csv`, `docs/*.xlsx`) are local-only and
gitignored -- they aren't required to run the test suite, only to exercise
`test_real_data_smoke.py` locally.

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

## Known preserved bugs

The original notebook's Status whitelist (`data_prep.DEFAULT_VALID_STATUSES`)
includes the literal string `"Low viability"`, which never matches real
GRIN-Global data (the actual value is `"Low germination"`); it also omits
several real statuses. This is intentionally preserved for now -- see the
module docstring and `tests/test_legacy_baseline.py::test_status_whitelist_bug_is_preserved`
-- and will be fixed as a deliberate, separately-reviewed change.
