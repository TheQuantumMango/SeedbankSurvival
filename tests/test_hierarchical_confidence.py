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


def _curve(intercept, linear_coef, n, r2, overall_pvalue, quad_coef=0.0, max_fit_age=100.0):
    return QuadraticCurve(
        intercept=intercept, linear_coef=linear_coef, quad_coef=quad_coef,
        n=n, r2=r2, overall_pvalue=overall_pvalue, max_fit_age=max_fit_age,
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


def test_extrapolation_does_not_apply_curve_shape_past_its_max_fit_age():
    # Regression test for a real, verified bug: a species-tier curve fit
    # only up to age 4 was getting extrapolated out to age 17, producing a
    # predicted 0% for a packet tested at a real, healthy 79% just a year
    # earlier. Straight line, slope -1/yr, but max_fit_age=10 -- beyond that
    # age the curve's rate is no longer trusted; held flat at its value AT
    # max_fit_age instead of continuing to extrapolate the same rate.
    # predict(5)=85, predict(10)=80 (the age-30 evaluation is never reached
    # unclamped -- min(30, 10) substitutes 10 first).
    global_model = _curve(90, -1.0, n=1000, r2=0.5, overall_pvalue=0.001, max_fit_age=10.0)
    row = _df_ranking(SeedAge=30, Viability=70.0, AgeAtTest=5)

    result = predict_hierarchical(row, {}, {}, global_model)

    # 70 + (predict(10)=80 - predict(5)=85) = 70 - 5 = 65 -- NOT
    # 70 + (predict(30)=60 - predict(5)=85) = 70 - 25 = 45, what an
    # unclamped extrapolation would have given.
    assert result.loc[0, "PredictedViability_2026"] == 65.0


def test_extrapolation_clips_each_curve_evaluation_before_differencing():
    # A curve evaluation far outside [0, 100] at ONE endpoint must not
    # compound with the other endpoint's (possibly also out-of-range) value
    # before the delta is taken -- each is clipped individually first.
    # predict(age) = 300 - 26*age: predict(1)=274 (clips to 100),
    # predict(10)=40 (already in range).
    global_model = _curve(300, -26.0, n=1000, r2=0.5, overall_pvalue=0.001, max_fit_age=100.0)
    row = _df_ranking(SeedAge=10, Viability=90.0, AgeAtTest=1)

    result = predict_hierarchical(row, {}, {}, global_model)

    # 90 + (clip(40)=40 - clip(274)=100) = 90 - 60 = 30 -- NOT
    # 90 + (40 - 274) = 90 - 234 = -144, clipped to a misleading flat 0
    # that would hide how the two failure modes compound.
    assert result.loc[0, "PredictedViability_2026"] == 30.0
