from __future__ import annotations

import numpy as np
import pandas as pd

from .deterioration import CurveModel


def _predict_row(
    row: pd.Series,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
) -> pd.Series:
    age = row["SeedAge"]

    if row["Species"] in species_models:
        model, model_used = species_models[row["Species"]], "Species"
    elif row["Origin"] in origin_models:
        model, model_used = origin_models[row["Origin"]], "Origin"
    else:
        model, model_used = global_model, "Global"

    # A tighter-sample-size tier isn't necessarily a better-fit one -- if the
    # genus-wide curve explains the data better than the tier the fallback
    # above picked, prefer it instead. A tier's R^2 can be NaN (a tiny,
    # near-zero-variance group, e.g. 3 nearly-identical Viability values --
    # rare but real on actual data) -- treat unknown confidence the same as
    # losing the comparison, since there's no basis to trust it over Global.
    #
    # R^2 alone isn't enough: a group fit on very few points routinely has a
    # high R^2 by chance, with a curve that's statistically indistinguishable
    # from flat -- verified against real data, where the large majority of
    # tier models that "won" on R^2 alone had a slope whose 95% CI spanned
    # zero. A tier's overall_pvalue (does age explain anything at all) must
    # also be significant (p < 0.05) to be trusted over Global; NaN (same
    # near-zero-variance/zero-residual-df case as above) fails this the same
    # way NaN r2 does.
    if model_used != "Global" and (
        not (model.r2 >= global_model.r2) or not (model.overall_pvalue < 0.05)
    ):
        model, model_used = global_model, "Global"

    tested_viability = row.get("Viability")
    age_at_test = row.get("AgeAtTest")
    if pd.notna(tested_viability) and pd.notna(age_at_test):
        # This specific inventory has its own observed test result (real, or
        # a documented assumption -- see grin_import.py's low-germination
        # imputation) -- anchor to that and extrapolate forward by however
        # much the selected tier's curve itself changes between the test age
        # and now, instead of the tier's population-average starting point
        # (its fitted intercept). More accurate for this individual packet
        # than treating it as "average for its tier," and the reason a test
        # result exists at all. Reduces to the old slope*(age-age_at_test)
        # extrapolation exactly when the curve is a straight line.
        predicted = tested_viability + (model.predict(age) - model.predict(age_at_test))
    else:
        predicted = model.predict(age)
    predicted = np.clip(predicted, 0, 100)

    return pd.Series([predicted, model_used, model.r2])


def predict_hierarchical(
    df_ranking: pd.DataFrame,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
) -> pd.DataFrame:
    """Predict current viability per accession.

    Falls back Species -> Origin -> Global by data availability, then uses
    Global instead whenever its R^2 beats the tier that fallback picked (and
    the tier's curve is statistically significant) -- a tier fit on very
    little data can look "specific" while actually explaining the
    deterioration worse than the genus-wide curve, or looking confident by
    chance. Whichever tier is selected, a row with its own observed
    Viability/AgeAtTest extrapolates from that specific measurement using
    how the tier's curve itself changes over that interval, rather than from
    the tier's population-average starting point -- see _predict_row.
    """
    df_ranking = df_ranking.copy()
    df_ranking[["PredictedViability_2026", "ModelUsed", "ModelConfidence"]] = df_ranking.apply(
        _predict_row,
        axis=1,
        args=(species_models, origin_models, global_model),
    )
    return df_ranking
