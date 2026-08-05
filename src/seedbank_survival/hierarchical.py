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
        model = species_models[row["Species"]]
        model_used = "Species"
    elif row["Origin"] in origin_models:
        model = origin_models[row["Origin"]]
        model_used = "Origin"
    else:
        model = global_model
        model_used = "Global"

    predicted = model.intercept + (model.slope * age)
    predicted = np.clip(predicted, 0, 100)

    return pd.Series([predicted, model_used])


def predict_hierarchical(
    df_ranking: pd.DataFrame,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
) -> pd.DataFrame:
    """Predict current viability per accession, falling back Species -> Origin -> Global."""
    df_ranking = df_ranking.copy()
    df_ranking[["PredictedViability_2026", "ModelUsed"]] = df_ranking.apply(
        _predict_row,
        axis=1,
        args=(species_models, origin_models, global_model),
    )
    return df_ranking
