from __future__ import annotations

import numpy as np
import pandas as pd

from .deterioration import SlopeModel


def _predict_row(
    row: pd.Series,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
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
    if model_used != "Global" and not (model.r2 >= global_model.r2):
        model, model_used = global_model, "Global"

    predicted = model.intercept + (model.slope * age)
    predicted = np.clip(predicted, 0, 100)

    return pd.Series([predicted, model_used, model.r2])


def predict_hierarchical(
    df_ranking: pd.DataFrame,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
) -> pd.DataFrame:
    """Predict current viability per accession.

    Falls back Species -> Origin -> Global by data availability, then uses
    Global instead whenever its R^2 beats the tier that fallback picked --
    a tier fit on very little data can look "specific" while actually
    explaining the deterioration worse than the genus-wide curve.
    """
    df_ranking = df_ranking.copy()
    df_ranking[["PredictedViability_2026", "ModelUsed", "ModelConfidence"]] = df_ranking.apply(
        _predict_row,
        axis=1,
        args=(species_models, origin_models, global_model),
    )
    return df_ranking
