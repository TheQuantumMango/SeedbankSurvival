"""determine_primary_reason's Viability-based reason text (priority.py). Uses
hand-built rows directly -- this behavior is independent of data source
(old-schema vs. raw-GRIN), so it doesn't need either fixture.
"""
from __future__ import annotations

import pandas as pd

from seedbank_survival.deterioration import QuadraticCurve
from seedbank_survival.priority import determine_primary_reason, estimate_years_to_zero

_GLOBAL = QuadraticCurve(intercept=80, linear_coef=-0.5, quad_coef=0.0, n=1000, r2=0.3, overall_pvalue=0.001, max_fit_age=100.0)


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


def test_fast_deterioration_reason_compares_instantaneous_slopes_at_same_age():
    # Species curve's local slope at SeedAge=20 must beat (be steeper than)
    # 1.5x Global's local slope at that SAME age -- not compared at age 0 or
    # via some other mismatched point, now that neither is a single constant.
    species_models = {"S1": QuadraticCurve(intercept=90, linear_coef=-1.0, quad_coef=-0.1, n=10, r2=0.9, overall_pvalue=0.01, max_fit_age=100.0)}
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=5.0)
    reason = determine_primary_reason(row, species_models, {}, _GLOBAL)
    assert "Fast species deterioration" in reason


def test_years_to_zero_uses_curves_instantaneous_slope_at_current_age():
    # predict(age) = 90 - age - 0.1*age^2 -> slope_at(20) = -1 - 0.2*20 = -5.
    # Predicted viability 25 -> 25 / 5 = 5 years remaining.
    model = QuadraticCurve(intercept=90, linear_coef=-1.0, quad_coef=-0.1, n=10, r2=0.9, overall_pvalue=0.01, max_fit_age=100.0)
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=25.0)
    years = estimate_years_to_zero(row, {"S1": model}, {}, _GLOBAL)
    assert years == 5.0


def test_years_to_zero_is_infinite_when_curve_is_flat_or_rising_at_current_age():
    model = QuadraticCurve(intercept=90, linear_coef=1.0, quad_coef=0.0, n=10, r2=0.9, overall_pvalue=0.01, max_fit_age=100.0)
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=95.0)
    years = estimate_years_to_zero(row, {"S1": model}, {}, _GLOBAL)
    assert years == float("inf")


def test_years_to_zero_is_zero_not_infinite_when_already_near_the_floor():
    # Regression test for a real inconsistency: a small-n quadratic can have
    # a locally positive slope_at() PAST its vertex (predict() rises again
    # after bottoming out) even while the row's own separately-bounded
    # PredictedViability_2026 is already ~0% -- verified on real data, two
    # "Critical viability" accessions showed a contradictory blank/infinite
    # years-to-zero. A near-zero prediction with a non-declining local slope
    # means "already there," not "stable forever."
    model = QuadraticCurve(intercept=90, linear_coef=1.0, quad_coef=0.0, n=10, r2=0.9, overall_pvalue=0.01, max_fit_age=100.0)
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=0.2)
    years = estimate_years_to_zero(row, {"S1": model}, {}, _GLOBAL)
    assert years == 0.0


def test_years_to_zero_caps_implausibly_large_estimates_to_infinite():
    # Regression test for a real outlier: a near-flat (but technically
    # negative) local slope near a weak curve's vertex computed a literal
    # 111,377.7 years-to-zero on real data -- mathematically consistent with
    # the formula, but not a meaningful distinction from "not currently
    # declining," and the false one-decimal precision on a 6-figure number
    # reads as a glitch. slope=-0.001/yr, viability=95 -> raw calculation
    # would be 95000 years, capped to inf instead.
    model = QuadraticCurve(intercept=95, linear_coef=-0.001, quad_coef=0.0, n=1000, r2=0.03, overall_pvalue=0.001, max_fit_age=100.0)
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=95.0)
    years = estimate_years_to_zero(row, {"S1": model}, {}, _GLOBAL)
    assert years == float("inf")


def test_years_to_zero_below_cap_is_reported_normally():
    # A genuinely meaningful decades-scale estimate must NOT get swept into
    # the same cap -- only implausibly large ones.
    model = QuadraticCurve(intercept=90, linear_coef=-1.0, quad_coef=0.0, n=1000, r2=0.3, overall_pvalue=0.001, max_fit_age=100.0)
    row = _row(SeedAge=20, ModelUsed="Species", PredictedViability_2026=90.0)
    years = estimate_years_to_zero(row, {"S1": model}, {}, _GLOBAL)
    assert years == 90.0
