"""Curve fitting for all three model kinds (deterioration.py). Weibull and
breakpoint are tested for the actual property that motivated adding them --
guaranteed non-increasing predictions, unlike quadratic, which real data
showed can turn back upward past its vertex.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import seedbank_survival.deterioration as det
from seedbank_survival.deterioration import (
    BreakpointCurve,
    QuadraticCurve,
    WeibullCurve,
    fit_global_model,
    fit_group_models,
)


def _declining_df(n=20, noise=0.0, seed=0):
    # Plateau then drop, roughly matching the shape found in real Astragalus
    # data -- flat near 85% until age 25, then declining at 3%/yr.
    rng = np.random.default_rng(seed)
    age = np.linspace(1, 50, n)
    viability = np.where(age < 25, 85.0, 85.0 - (age - 25) * 3)
    viability = np.clip(viability + rng.normal(0, noise, n), 0, 100)
    return pd.DataFrame({"AgeAtTest": age, "Viability": viability, "Group": "G1"})


def _assert_monotonic_non_increasing(model, age_max=60):
    ages = np.linspace(0, age_max, 200)
    preds = [model.predict(a) for a in ages]
    violations = [
        (ages[i], preds[i], ages[i + 1], preds[i + 1])
        for i in range(len(preds) - 1)
        if preds[i + 1] > preds[i] + 1e-9
    ]
    assert violations == [], f"viability increased with age: {violations[:5]}"


class TestQuadratic:
    def test_fits_and_returns_quadratic_curve(self):
        model = fit_global_model(_declining_df(), "quadratic")
        assert isinstance(model, QuadraticCurve)
        assert model.n == 20

    def test_can_turn_back_upward_past_its_vertex(self):
        # Documenting the exact limitation the other two kinds fix -- a
        # quadratic with a positive quad_coef at old ages predicts RISING
        # viability, which real seed viability never does.
        model = QuadraticCurve(intercept=50, linear_coef=-5, quad_coef=0.5, n=10, r2=0.9, overall_pvalue=0.01, max_fit_age=100.0)
        assert model.predict(20) > model.predict(10)


class TestWeibull:
    def test_fits_and_returns_weibull_curve(self):
        model = fit_global_model(_declining_df(), "weibull")
        assert isinstance(model, WeibullCurve)

    def test_predictions_are_monotonic_non_increasing(self):
        model = fit_global_model(_declining_df(n=30, noise=3.0), "weibull")
        _assert_monotonic_non_increasing(model)

    def test_significant_on_a_clearly_declining_group(self):
        model = fit_global_model(_declining_df(noise=1.0), "weibull")
        assert model.overall_pvalue < 0.05

    def test_group_models_skip_groups_below_min_n(self):
        df = _declining_df(n=3)
        models = fit_group_models(df, "Group", "weibull")
        assert models == {}

    def test_group_models_fit_groups_at_or_above_min_n(self):
        df = _declining_df(n=4)
        models = fit_group_models(df, "Group", "weibull")
        assert "G1" in models
        assert isinstance(models["G1"], WeibullCurve)

    def test_fit_global_model_raises_clearly_if_fit_fails(self, monkeypatch):
        monkeypatch.setattr(det, "_fit_weibull", lambda df: None)
        with pytest.raises(RuntimeError, match="weibull"):
            fit_global_model(_declining_df(), "weibull")


class TestBreakpoint:
    def test_fits_and_returns_breakpoint_curve(self):
        model = fit_global_model(_declining_df(), "breakpoint")
        assert isinstance(model, BreakpointCurve)

    def test_predictions_are_monotonic_non_increasing(self):
        model = fit_global_model(_declining_df(n=30, noise=3.0), "breakpoint")
        _assert_monotonic_non_increasing(model)

    def test_finds_the_plateau_and_drop_shape(self):
        model = fit_global_model(_declining_df(noise=0.0), "breakpoint")
        assert 15 <= model.t0 <= 35  # near the true breakpoint at 25
        assert model.plateau == pytest.approx(85.0, abs=2)
        assert model.slope < 0

    def test_slope_never_positive_even_if_unconstrained_fit_would_be(self):
        # Flat, noisy data with no real trend -- an unconstrained
        # least-squares slope on the "after" segment could easily come out
        # slightly positive from noise; must be clamped to 0, not left as-is.
        rng = np.random.default_rng(1)
        age = np.linspace(1, 30, 15)
        viab = np.clip(80 + rng.normal(0, 2, 15), 0, 100)
        df = pd.DataFrame({"AgeAtTest": age, "Viability": viab})
        model = fit_global_model(df, "breakpoint")
        assert model.slope <= 0

    def test_group_models_skip_groups_below_min_n(self):
        df = _declining_df(n=3)
        models = fit_group_models(df, "Group", "breakpoint")
        assert models == {}


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        fit_global_model(_declining_df(), "bogus")  # type: ignore[arg-type]


@pytest.mark.parametrize("kind", ["quadratic", "weibull", "breakpoint"])
def test_max_fit_age_matches_the_oldest_age_actually_fit_on(kind):
    # hierarchical.py's own-test extrapolation relies on this being the true
    # oldest AgeAtTest the curve saw, not an estimate -- it's the boundary
    # past which the curve's shape is trusted no further.
    df = _declining_df(n=20)
    model = fit_global_model(df, kind)
    assert model.max_fit_age == df["AgeAtTest"].max()


@pytest.mark.parametrize("kind", ["quadratic", "weibull", "breakpoint"])
def test_group_models_max_fit_age_is_per_group_not_whole_dataset(kind):
    # A species fit only on young lots shouldn't inherit a larger
    # max_fit_age from some OTHER species' older lots in the same df_model.
    young = pd.DataFrame({
        "AgeAtTest": [1.0, 3.0, 5.0, 7.0, 9.0, 10.0],
        "Viability": [85.0, 84.0, 82.0, 80.0, 78.0, 77.0],
        "Group": "Young",
    })
    old = _declining_df(n=6, seed=2)
    old["Group"] = "Old"
    df = pd.concat([young, old], ignore_index=True)

    models = fit_group_models(df, "Group", kind)
    assert models["Young"].max_fit_age == young["AgeAtTest"].max()
    assert models["Old"].max_fit_age == old["AgeAtTest"].max()
    assert models["Young"].max_fit_age < models["Old"].max_fit_age
