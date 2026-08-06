"""Confidence-weighted tier selection (hierarchical.predict_hierarchical). Uses
hand-built QuadraticCurve instances directly (the logic tested here is
curve-kind-agnostic -- it only relies on predict()/slope_at(), which every
Curve implementation provides) -- this behavior applies equally to
old-schema and raw-GRIN data, so it doesn't need either fixture.
"""
from __future__ import annotations

import math

import pandas as pd

from seedbank_survival.deterioration import QuadraticCurve
from seedbank_survival.hierarchical import predict_hierarchical


def _df_ranking(**row):
    defaults = {"Accession": "A1", "SeedAge": 10, "Species": "S1", "Origin": "O1"}
    defaults.update(row)
    return pd.DataFrame([defaults])


def _curve(intercept, linear_coef, n, r2, overall_pvalue, quad_coef=0.0):
    return QuadraticCurve(
        intercept=intercept, linear_coef=linear_coef, quad_coef=quad_coef,
        n=n, r2=r2, overall_pvalue=overall_pvalue,
    )


def test_keeps_species_tier_when_it_has_higher_confidence_and_is_significant():
    species_models = {"S1": _curve(90, -1.0, n=10, r2=0.9, overall_pvalue=0.01)}
    global_model = _curve(80, -0.5, n=100, r2=0.3, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Species"
    assert result.loc[0, "ModelConfidence"] == 0.9


def test_overrides_to_global_when_global_has_higher_confidence():
    species_models = {"S1": _curve(90, -1.0, n=3, r2=0.2, overall_pvalue=0.01)}
    global_model = _curve(80, -0.5, n=1000, r2=0.7, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.7
    assert result.loc[0, "PredictedViability_2026"] == 80 - 0.5 * 10


def test_overrides_origin_tier_to_global_when_global_has_higher_confidence():
    origin_models = {"O1": _curve(85, -1.2, n=3, r2=0.4, overall_pvalue=0.01)}
    global_model = _curve(80, -0.5, n=1000, r2=0.6, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), {}, origin_models, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"


def test_global_tier_never_overrides_itself():
    global_model = _curve(80, -0.5, n=1000, r2=0.6, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), {}, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.6


def test_nan_confidence_defers_to_global():
    # A near-zero-variance group (e.g. 3 nearly-identical Viability values,
    # or exactly n=3 with a quadratic fit -- zero residual degrees of
    # freedom either way) can produce an undefined R^2 -- real on actual
    # data, not hypothetical.
    species_models = {"S1": _curve(96.0, 0.0, n=3, r2=float("nan"), overall_pvalue=float("nan"))}
    global_model = _curve(80, -0.5, n=1000, r2=0.6, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.6
    assert not math.isnan(result.loc[0, "ModelConfidence"])


def test_overrides_to_global_when_slope_not_significant_despite_higher_r2():
    # The core case this gate exists for: a tiny group can have a HIGHER r2
    # than Global almost by chance, with a curve that isn't statistically
    # distinguishable from flat -- verified this is the majority case for
    # real Origin-tier models that "won" on r2 alone. Such a tier must not
    # be trusted over Global.
    species_models = {"S1": _curve(90, -1.0, n=5, r2=0.95, overall_pvalue=0.4)}
    global_model = _curve(80, -0.5, n=1000, r2=0.3, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.3


def test_nan_overall_pvalue_defers_to_global_even_with_higher_r2():
    species_models = {"S1": _curve(90, -1.0, n=5, r2=0.95, overall_pvalue=float("nan"))}
    global_model = _curve(80, -0.5, n=1000, r2=0.3, overall_pvalue=0.001)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"


def test_extrapolates_from_own_test_result_instead_of_population_intercept():
    # This packet's own test (70% at AgeAtTest=5) is the ground truth for
    # itself -- prediction should extrapolate from THAT, not from the tier's
    # fitted intercept (90, the *population* average at age 0), which this
    # specific packet may be nowhere near.
    global_model = _curve(90, -2.0, n=1000, r2=0.5, overall_pvalue=0.001)
    row = _df_ranking(SeedAge=15, Viability=70.0, AgeAtTest=5)

    result = predict_hierarchical(row, {}, {}, global_model)

    # 70% at age 5, extrapolated 10 more years (age 15) at -2%/yr = 70 - 20 = 50.
    assert result.loc[0, "PredictedViability_2026"] == 50.0


def test_falls_back_to_curve_when_row_has_no_own_test_result():
    # No Viability/AgeAtTest columns at all -- must still use the
    # intercept-based curve exactly as before this feature existed.
    global_model = _curve(90, -2.0, n=1000, r2=0.5, overall_pvalue=0.001)
    row = _df_ranking(SeedAge=15)

    result = predict_hierarchical(row, {}, {}, global_model)

    assert result.loc[0, "PredictedViability_2026"] == 90 - 2.0 * 15


def test_falls_back_to_curve_when_own_test_result_is_nan():
    # Viability/AgeAtTest columns exist (as they do for every real
    # raw-GRIN-derived row) but are NaN for this particular untested packet.
    global_model = _curve(90, -2.0, n=1000, r2=0.5, overall_pvalue=0.001)
    row = _df_ranking(SeedAge=15, Viability=float("nan"), AgeAtTest=float("nan"))

    result = predict_hierarchical(row, {}, {}, global_model)

    assert result.loc[0, "PredictedViability_2026"] == 90 - 2.0 * 15


def test_extrapolation_uses_the_selected_tiers_curve_not_globals():
    # The tier selected by the existing confidence override still governs
    # the curve used for extrapolation -- only the starting point
    # (population intercept vs. this packet's own test) changes.
    species_models = {"S1": _curve(90, -3.0, n=10, r2=0.9, overall_pvalue=0.01)}
    global_model = _curve(80, -0.5, n=1000, r2=0.3, overall_pvalue=0.001)
    row = _df_ranking(SeedAge=12, Viability=60.0, AgeAtTest=10)

    result = predict_hierarchical(row, species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Species"
    # 60% at age 10, extrapolated 2 more years at Species's -3%/yr = 60 - 6 = 54.
    assert result.loc[0, "PredictedViability_2026"] == 54.0


def test_extrapolated_prediction_still_clipped_to_valid_range():
    global_model = _curve(90, -10.0, n=1000, r2=0.5, overall_pvalue=0.001)
    row = _df_ranking(SeedAge=50, Viability=20.0, AgeAtTest=5)

    result = predict_hierarchical(row, {}, {}, global_model)

    assert result.loc[0, "PredictedViability_2026"] == 0


def test_quadratic_curve_used_directly_when_no_own_test_result():
    # intercept=90, linear=-1, quad=-0.02 -> predict(20) = 90 - 20 - 0.02*400 = 62.
    global_model = _curve(90, -1.0, n=1000, r2=0.5, overall_pvalue=0.001, quad_coef=-0.02)
    row = _df_ranking(SeedAge=20)

    result = predict_hierarchical(row, {}, {}, global_model)

    assert result.loc[0, "PredictedViability_2026"] == 62.0


def test_quadratic_curve_extrapolates_by_the_curves_own_change_between_ages():
    # predict(age) = 90 - age - 0.02*age^2.
    # predict(5) = 90 - 5 - 0.5 = 84.5; predict(20) = 90 - 20 - 8 = 62.
    # Own test measured 70 (not 84.5) at age 5 -- extrapolation should apply
    # the curve's OWN delta (62 - 84.5 = -22.5) on top of that real 70, not
    # treat 70 as if it matched the curve's fitted value at age 5.
    global_model = _curve(90, -1.0, n=1000, r2=0.5, overall_pvalue=0.001, quad_coef=-0.02)
    row = _df_ranking(SeedAge=20, Viability=70.0, AgeAtTest=5)

    result = predict_hierarchical(row, {}, {}, global_model)

    assert result.loc[0, "PredictedViability_2026"] == 47.5
