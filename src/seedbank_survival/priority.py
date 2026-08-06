from __future__ import annotations

import numpy as np
import pandas as pd

from .deterioration import SlopeModel

_OUTPUT_COLUMNS = [
    "Accession",
    "Suffix",
    "Status",
    "Species",
    "Origin",
    "SeedAge",
    "PrimaryReason",
    "PredictedViability_2026",
    "YearsToZero",
    "ModelUsed",
    "ModelConfidence",
]

_OUTPUT_COLUMN_NAMES = [
    "Accession",
    "Suffix",
    "Status",
    "Species",
    "Origin",
    "SeedAge",
    "PrimaryReason",
    "EstimatedViability_2026",
    "YearsRemainingTo0%",
    "ModelUsed",
    "ModelConfidence",
]


def estimate_years_to_zero(
    row: pd.Series,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
) -> float:
    """Years until predicted viability reaches 0%, given the model tier that produced it."""
    viability = row["PredictedViability_2026"]

    if row["ModelUsed"] == "Species":
        slope = species_models[row["Species"]].slope
    elif row["ModelUsed"] == "Origin":
        slope = origin_models[row["Origin"]].slope
    else:
        slope = global_model.slope

    if slope >= 0:
        return np.inf

    years_remaining = viability / abs(slope)
    return max(0, years_remaining)


def determine_primary_reason(
    row: pd.Series,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
) -> str:
    """Human-readable summary of why an accession is flagged as regeneration priority."""
    reasons = []

    if row["SeedAge"] >= 75:
        reasons.append("Extreme seed age")
    elif row["SeedAge"] >= 40:
        reasons.append("Old seed age")

    if row["ModelUsed"] == "Species":
        slope = species_models[row["Species"]].slope
        if slope < global_model.slope * 1.5:
            reasons.append("Fast species deterioration")

    if row["ModelUsed"] == "Origin":
        slope = origin_models[row["Origin"]].slope
        if slope < global_model.slope * 1.5:
            reasons.append("Fast origin deterioration")

    if row["PredictedViability_2026"] <= 10:
        reasons.append("Critical viability")
    elif row["PredictedViability_2026"] <= 30:
        reasons.append("Low viability")

    if not reasons:
        reasons.append("General deterioration")

    return "; ".join(reasons)


def build_priority_table(
    df_ranking: pd.DataFrame,
    species_models: dict[str, SlopeModel],
    origin_models: dict[str, SlopeModel],
    global_model: SlopeModel,
    top_n: int = 50,
) -> pd.DataFrame:
    """The N lowest-predicted-viability accessions, with deterioration diagnostics attached.

    df_ranking must already have PredictedViability_2026 and ModelUsed columns
    (see hierarchical.predict_hierarchical).
    """
    df_ranking = df_ranking.copy()

    df_ranking["YearsToZero"] = df_ranking.apply(
        estimate_years_to_zero,
        axis=1,
        args=(species_models, origin_models, global_model),
    )
    df_ranking["PrimaryReason"] = df_ranking.apply(
        determine_primary_reason,
        axis=1,
        args=(species_models, origin_models, global_model),
    )

    priority_table = (
        df_ranking.sort_values("PredictedViability_2026").head(top_n).copy()
    )
    priority_table = priority_table[_OUTPUT_COLUMNS]
    priority_table.columns = _OUTPUT_COLUMN_NAMES

    priority_table["EstimatedViability_2026"] = priority_table[
        "EstimatedViability_2026"
    ].round(1)
    priority_table["YearsRemainingTo0%"] = (
        priority_table["YearsRemainingTo0%"].replace(np.inf, np.nan).round(1)
    )
    # 2 decimals, distinct from the 1-decimal convention above -- this is a
    # model R^2 (goodness of fit), not a viability percentage.
    priority_table["ModelConfidence"] = priority_table["ModelConfidence"].round(2)

    return priority_table
