from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import curve_fit
from statsmodels.formula.api import ols

ModelKind = Literal["quadratic", "weibull", "breakpoint"]
_DEFAULT_MODEL_KIND: ModelKind = "quadratic"

# Bounds used to keep scipy's nonlinear solver away from degenerate regions
# (e.g. lambda/k near 0) and to keep every fitted Weibull curve strictly
# monotonic non-increasing by construction -- see WeibullCurve.
_WEIBULL_BOUNDS = ([1.0, 0.1, 0.1], [150.0, 500.0, 20.0])
_WEIBULL_MAXFEV = 8000


@dataclass(frozen=True)
class QuadraticCurve:
    """A fitted quadratic viability-deterioration curve:
    Viability ~ AgeAtTest + AgeAtTest^2.

    The original model form here. Simple (closed-form OLS fit, no solver
    convergence risk) and often fits real data about as well as the
    alternatives below, but nothing about a parabola stops it from turning
    back upward past its vertex -- checked against real data, this does
    happen for some groups (a visually implausible "gets more viable with
    age" shape). WeibullCurve and BreakpointCurve exist specifically to
    guarantee that can't happen; kept as an option since it isn't always the
    worse fit, and a curator may want to compare.

    overall_pvalue is the regression F-test p-value: "does age (linear and
    quadratic terms together) explain any of the variation in Viability."

    max_fit_age is the oldest AgeAtTest actually observed in the data this
    curve was fit on -- hierarchical.py's own-test extrapolation won't trust
    this curve's shape past that age, since real data showed a curve's rate
    of change extrapolated far beyond where it was ever fit can swing wildly
    (verified: this is what produced "tested at 79%, predicted 0%" results).
    """

    intercept: float
    linear_coef: float
    quad_coef: float
    n: int
    r2: float
    overall_pvalue: float
    max_fit_age: float

    def predict(self, age: float) -> float:
        return self.intercept + self.linear_coef * age + self.quad_coef * age**2

    def slope_at(self, age: float) -> float:
        return self.linear_coef + 2 * self.quad_coef * age


@dataclass(frozen=True)
class WeibullCurve:
    """A fitted Weibull decay curve: Viability(t) = v0 * exp(-(t/lam)^k).

    The standard time-to-failure/survival-analysis form, also used in
    published seed-ageing literature. Strictly non-increasing for t >= 0
    given v0, lam, k > 0 (guaranteed by _WEIBULL_BOUNDS keeping the solver
    in that region) -- can never show viability rising with age, unlike
    QuadraticCurve. The shape parameter k controls the curve's character:
    k > 1 gives a plateau that accelerates into decline (the shape real
    Astragalus data showed when binned by age); k < 1 gives fast-then-slow
    decline; k = 1 is plain exponential decay.

    overall_pvalue: an F-test comparing this fit's residual sum of squares
    against a mean-only (1-parameter) null model -- the same test
    statsmodels' regression F-test performs for QuadraticCurve, computed by
    hand here since this isn't an OLS fit. Answers the same question ("does
    age explain anything") the same way, so tiers built on different curve
    kinds remain comparable.
    """

    v0: float
    lam: float
    k: float
    n: int
    r2: float
    overall_pvalue: float
    max_fit_age: float

    def predict(self, age: float) -> float:
        age = max(age, 0.0)
        return self.v0 * np.exp(-((age / self.lam) ** self.k))

    def slope_at(self, age: float) -> float:
        # Closed-form derivative of predict(). Guarded away from age=0 --
        # for k<1 the derivative blows up there (a real, if unlikely,
        # possibility for a freshly-added lot with AgeAtTest/SeedAge = 0).
        age = max(age, 1e-6)
        return -self.predict(age) * self.k * (age / self.lam) ** (self.k - 1) / self.lam


@dataclass(frozen=True)
class BreakpointCurve:
    """A fitted two-segment curve: flat at `plateau` until age `t0`, then
    declining at `slope` (constrained <= 0) after.

    The most literal match to a "holds steady, then drops" shape, and the
    easiest to explain to a curator ("viability holds until age X, then
    declines at Y%/yr"). Fit by grid search over candidate breakpoints
    (every observed age, trying each as t0) rather than a nonlinear solver
    -- deterministic, no convergence failures possible. The post-breakpoint
    slope is clamped <= 0 so the curve can never rise, even if the
    unconstrained least-squares slope for that segment came out slightly
    positive from noise.
    """

    t0: float
    plateau: float
    slope: float
    n: int
    r2: float
    overall_pvalue: float
    max_fit_age: float

    def predict(self, age: float) -> float:
        if age <= self.t0:
            return self.plateau
        return self.plateau + self.slope * (age - self.t0)

    def slope_at(self, age: float) -> float:
        return 0.0 if age <= self.t0 else self.slope


Curve = QuadraticCurve | WeibullCurve | BreakpointCurve


def _f_test_pvalue(rss_full: float, rss_null: float, n: int, p_full: int) -> float:
    """F-test p-value for "does the full model explain more than a
    mean-only (1-parameter) null model" -- the same test statsmodels'
    regression f_pvalue performs, computed by hand for a fit that isn't OLS.
    """
    df1 = p_full - 1
    df2 = n - p_full
    if df1 <= 0 or df2 <= 0 or rss_full <= 0:
        return float("nan")
    f_stat = ((rss_null - rss_full) / df1) / (rss_full / df2)
    if f_stat < 0:
        return 1.0
    return float(scipy_stats.f.sf(f_stat, df1, df2))


def _r2(rss_full: float, rss_null: float) -> float:
    if rss_null <= 0:
        return float("nan")
    return 1 - rss_full / rss_null


def _fit_quadratic(df: pd.DataFrame) -> QuadraticCurve:
    fit = ols("Viability ~ AgeAtTest + I(AgeAtTest ** 2)", data=df).fit()
    return QuadraticCurve(
        intercept=fit.params["Intercept"],
        linear_coef=fit.params["AgeAtTest"],
        quad_coef=fit.params["I(AgeAtTest ** 2)"],
        n=len(df),
        r2=fit.rsquared,
        overall_pvalue=fit.f_pvalue,
        max_fit_age=float(df["AgeAtTest"].max()),
    )


def _weibull_func(t: np.ndarray, v0: float, lam: float, k: float) -> np.ndarray:
    return v0 * np.exp(-((np.maximum(t, 0.0) / lam) ** k))


def _fit_weibull(df: pd.DataFrame) -> WeibullCurve | None:
    age = df["AgeAtTest"].to_numpy(dtype=float)
    viab = df["Viability"].to_numpy(dtype=float)
    n = len(age)

    try:
        popt, _ = curve_fit(
            _weibull_func, age, viab,
            p0=[max(float(viab.max()), 1.0), max(float(age.mean()), 1.0), 1.5],
            bounds=_WEIBULL_BOUNDS, maxfev=_WEIBULL_MAXFEV,
        )
    except (RuntimeError, ValueError):
        return None

    v0, lam, k = (float(x) for x in popt)
    pred = _weibull_func(age, v0, lam, k)
    rss_full = float(np.sum((viab - pred) ** 2))
    rss_null = float(np.sum((viab - viab.mean()) ** 2))
    return WeibullCurve(
        v0=v0, lam=lam, k=k, n=n,
        r2=_r2(rss_full, rss_null),
        overall_pvalue=_f_test_pvalue(rss_full, rss_null, n, p_full=3),
        max_fit_age=float(age.max()),
    )


def _fit_breakpoint(df: pd.DataFrame) -> BreakpointCurve | None:
    age = df["AgeAtTest"].to_numpy(dtype=float)
    viab = df["Viability"].to_numpy(dtype=float)
    n = len(age)

    best = None
    for t0 in sorted(set(age)):
        before = age <= t0
        after = ~before
        if before.sum() < 2 or after.sum() < 2:
            continue
        plateau = float(viab[before].mean())
        x = age[after] - t0
        denom = float(np.sum(x * x))
        slope = min(0.0, float(np.sum(x * (viab[after] - plateau)) / denom)) if denom > 0 else 0.0
        pred = np.where(before, plateau, plateau + slope * (age - t0))
        rss = float(np.sum((viab - pred) ** 2))
        if best is None or rss < best[0]:
            best = (rss, t0, plateau, slope)

    if best is None:
        return None
    rss_full, t0, plateau, slope = best
    rss_null = float(np.sum((viab - viab.mean()) ** 2))
    return BreakpointCurve(
        t0=t0, plateau=plateau, slope=slope, n=n,
        r2=_r2(rss_full, rss_null),
        overall_pvalue=_f_test_pvalue(rss_full, rss_null, n, p_full=3),
        max_fit_age=float(age.max()),
    )


def _fit(df: pd.DataFrame, model_kind: ModelKind) -> Curve | None:
    if model_kind == "quadratic":
        return _fit_quadratic(df)
    if model_kind == "weibull":
        return _fit_weibull(df)
    if model_kind == "breakpoint":
        return _fit_breakpoint(df)
    raise ValueError(f"unknown model_kind: {model_kind!r}")


def fit_global_model(df_model: pd.DataFrame, model_kind: ModelKind = _DEFAULT_MODEL_KIND) -> Curve:
    """Fit one deterioration curve across the whole dataset (the genus-level curve).

    Unlike fit_group_models, this has no fallback if fitting fails -- the
    genus-wide curve is the final tier everything else falls back to, so a
    failure here (only possible for weibull, if scipy's solver can't
    converge) is raised rather than silently producing nothing.
    """
    model = _fit(df_model, model_kind)
    if model is None:
        raise RuntimeError(
            f"failed to fit a {model_kind} curve to the genus-wide dataset "
            f"(n={len(df_model)}) -- try a different --model"
        )
    return model


def fit_group_models(
    df_model: pd.DataFrame,
    group_col: str,
    model_kind: ModelKind = _DEFAULT_MODEL_KIND,
    min_n: int = 4,
) -> dict[str, Curve]:
    """Fit one deterioration curve per distinct value of group_col with enough data.

    Groups with fewer than min_n rows are skipped. min_n defaults to 4, not
    3: every curve kind here has 3 parameters, and a 3-parameter fit on
    exactly 3 points has zero residual degrees of freedom -- always a
    trivial, meaningless r2=1.0 (quadratic) or an unfittable/degenerate case
    (weibull, breakpoint) regardless of the data. 4 is the minimum for even
    one residual degree of freedom.

    A group where fitting fails to converge (weibull only -- quadratic and
    breakpoint can't fail this way) is silently skipped, the same as a
    group with too little data -- hierarchical.py's Species -> Origin ->
    Global fallback already handles "no model for this group" by design.
    """
    models: dict[str, Curve] = {}
    for group_value in df_model[group_col].dropna().unique():
        data = df_model[df_model[group_col] == group_value]
        if len(data) < min_n:
            continue
        model = _fit(data, model_kind)
        if model is not None:
            models[group_value] = model
    return models
