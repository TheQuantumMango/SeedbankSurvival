"""determine_primary_reason's Viability-based reason text (priority.py). Uses
hand-built rows directly -- this behavior is independent of data source
(old-schema vs. raw-GRIN), so it doesn't need either fixture.
"""
from __future__ import annotations

import pandas as pd

from seedbank_survival.deterioration import SlopeModel
from seedbank_survival.priority import determine_primary_reason

_GLOBAL = SlopeModel(intercept=80, slope=-0.5, n=1000, r2=0.3, slope_pvalue=0.001)


def _row(**overrides):
    defaults = {
        "SeedAge": 10,
        "Species": "S1",
        "Origin": "O1",
        "ModelUsed": "Global",
        "PredictedViability_2026": 50.0,
    }
    defaults.update(overrides)
    return pd.Series(defaults)


def test_flags_assumed_low_germination():
    row = _row(ViabilityAssumed=True, Viability=10.0, AgeAtTest=0.0)
    reason = determine_primary_reason(row, {}, {}, _GLOBAL)
    assert "Assumed low germination, no test data" in reason


def test_flags_real_low_viability_test_distinctly_from_assumed():
    row = _row(ViabilityAssumed=False, Viability=15.0, AgeAtTest=5.0)
    reason = determine_primary_reason(row, {}, {}, _GLOBAL)
    assert "Tested at low viability" in reason
    assert "Assumed low germination" not in reason


def test_does_not_flag_real_high_viability_test():
    row = _row(ViabilityAssumed=False, Viability=85.0, AgeAtTest=5.0)
    reason = determine_primary_reason(row, {}, {}, _GLOBAL)
    assert "Tested at low viability" not in reason
    assert "Assumed low germination" not in reason


def test_no_own_test_data_omits_both_tags():
    # No Viability/AgeAtTest at all -- e.g. the old reformatted-CSV path for
    # a row that also predates this feature, or genuinely never tested.
    row = _row()
    reason = determine_primary_reason(row, {}, {}, _GLOBAL)
    assert "Tested at low viability" not in reason
    assert "Assumed low germination" not in reason


def test_missing_viability_assumed_column_defaults_to_not_assumed():
    # ViabilityAssumed only exists on the raw-GRIN path -- the older
    # reformatted CSV/XLSX path has no such column at all.
    row = _row(Viability=5.0, AgeAtTest=3.0)
    reason = determine_primary_reason(row, {}, {}, _GLOBAL)
    assert "Tested at low viability" in reason
    assert "Assumed low germination" not in reason
