from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from statsmodels.formula.api import ols


@dataclass(frozen=True)
class SlopeModel:
    """A fitted linear viability-deterioration model: Viability ~ AgeAtTest.

    slope_pvalue is the two-sided p-value for the null hypothesis that the
    slope is zero (no deterioration trend) -- distinct from r2. A small group
    (e.g. n=3, 1 residual degree of freedom) can have a high r2 almost by
    chance; slope_pvalue is what actually says whether the fitted trend is
    distinguishable from noise. See hierarchical.py's confidence override,
    which requires both.
    """

    intercept: float
    slope: float
    n: int
    r2: float
    slope_pvalue: float


def _fit(df: pd.DataFrame) -> SlopeModel:
    fit = ols("Viability ~ AgeAtTest", data=df).fit()
    return SlopeModel(
        intercept=fit.params["Intercept"],
        slope=fit.params["AgeAtTest"],
        n=len(df),
        r2=fit.rsquared,
        slope_pvalue=fit.pvalues["AgeAtTest"],
    )


def fit_global_model(df_model: pd.DataFrame) -> SlopeModel:
    """Fit one deterioration model across the whole dataset (the genus-level curve)."""
    return _fit(df_model)


def fit_group_models(
    df_model: pd.DataFrame, group_col: str, min_n: int = 3
) -> dict[str, SlopeModel]:
    """Fit one deterioration model per distinct value of group_col with enough data.

    Groups with fewer than min_n rows are skipped (too little data for a stable fit).
    """
    models: dict[str, SlopeModel] = {}
    for group_value in df_model[group_col].dropna().unique():
        data = df_model[df_model[group_col] == group_value]
        if len(data) >= min_n:
            models[group_value] = _fit(data)
    return models
