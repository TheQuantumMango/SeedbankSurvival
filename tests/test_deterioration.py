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
        model = QuadraticCurve(intercept=50, linear_coef=-5, quad_coef=0.5, n=10, r2=0.9, overall_pvalue=0.01)
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
