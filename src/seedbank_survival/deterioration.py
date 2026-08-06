from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.formula.api import ols


@dataclass(frozen=True)
class CurveModel:
    """A fitted quadratic viability-deterioration curve:
    Viability ~ AgeAtTest + AgeAtTest^2.

    Replaced a straight-line (single slope) fit: binning real Astragalus
    data by age showed viability holding roughly flat for ~40 years, then
    dropping sharply -- a plateau-then-cliff shape a straight line can't
    represent, but a quadratic tracks meaningfully better (checked against
    several sigmoid-shaped alternatives -- probit/logit/exponential decay,
    the standard seed-science forms -- which all fit real Astragalus data
    *worse* than a straight line once compared fairly on the original
    percentage scale; only the quadratic improved on it).

    overall_pvalue is the regression F-test p-value: "does age (linear and
    quadratic terms together) explain any of the variation in Viability."
    This is the direct generalization of a single-predictor model's slope
    p-value -- the two are mathematically identical when there's only one
    age term, i.e. the previous linear model -- and is the right question
    to ask of a curve as a whole, which no longer has one fixed "the slope."
    """

    intercept: float
    linear_coef: float
    quad_coef: float
    n: int
    r2: float
    overall_pvalue: float

    def predict(self, age: float) -> float:
        """Fitted viability at a given age, from this curve alone (not clipped)."""
        return self.intercept + self.linear_coef * age + self.quad_coef * age**2

    def slope_at(self, age: float) -> float:
        """Instantaneous rate of change at a given age (d/d(age) of predict)."""
        return self.linear_coef + 2 * self.quad_coef * age


def _fit(df: pd.DataFrame) -> CurveModel:
    fit = ols("Viability ~ AgeAtTest + I(AgeAtTest ** 2)", data=df).fit()
    return CurveModel(
        intercept=fit.params["Intercept"],
        linear_coef=fit.params["AgeAtTest"],
        quad_coef=fit.params["I(AgeAtTest ** 2)"],
        n=len(df),
        r2=fit.rsquared,
        overall_pvalue=fit.f_pvalue,
    )


def fit_global_model(df_model: pd.DataFrame) -> CurveModel:
    """Fit one deterioration curve across the whole dataset (the genus-level curve)."""
    return _fit(df_model)


def fit_group_models(
    df_model: pd.DataFrame, group_col: str, min_n: int = 4
) -> dict[str, CurveModel]:
    """Fit one deterioration curve per distinct value of group_col with enough data.

    Groups with fewer than min_n rows are skipped. min_n defaults to 4, not
    3: a 3-parameter quadratic fit on exactly 3 points has zero residual
    degrees of freedom, which always produces a trivial, meaningless r2=1.0
    with an undefined (NaN) overall_pvalue -- 4 is the minimum for even one
    residual degree of freedom. (The NaN p-value would defer to Global
    anyway via hierarchical.py's confidence override, but there's no reason
    to fit an always-degenerate curve in the first place.)
    """
    models: dict[str, CurveModel] = {}
    for group_value in df_model[group_col].dropna().unique():
        data = df_model[df_model[group_col] == group_value]
        if len(data) >= min_n:
            models[group_value] = _fit(data)
    return models
