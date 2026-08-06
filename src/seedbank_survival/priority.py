from __future__ import annotations

import numpy as np
import pandas as pd

from .deterioration import CurveModel

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
    "MaintenanceSite",
    "Location",
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
    "MaintenanceSite",
    "Location",
]

# Only populated on the raw-GRIN-export path (grin_import.adapt_raw_export) --
# the older reformatted CSV/XLSX path (data_prep.load_accessions) has no
# per-inventory location data at all, so those columns are filled blank
# rather than required, to keep that path usable.
_LOCATION_COLUMNS = ("MaintenanceSite", "Location")


def _model_for_row(
    row: pd.Series,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
) -> CurveModel:
    if row["ModelUsed"] == "Species":
        return species_models[row["Species"]]
    if row["ModelUsed"] == "Origin":
        return origin_models[row["Origin"]]
    return global_model


def estimate_years_to_zero(
    row: pd.Series,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
) -> float:
    """Years until predicted viability reaches 0%, given the model tier that produced it.

    A quadratic curve's rate of decline isn't constant, so this uses the
    curve's INSTANTANEOUS slope at the row's current age as a local linear
    approximation of the near-term trajectory, rather than solving for where
    the full curve eventually crosses zero (which can have 0, 1, or 2 real
    roots, or curve back upward -- not a meaningful "years remaining" for a
    curator's near-term planning). Reduces to the exact previous calculation
    when the curve is a straight line (slope_at is then constant everywhere).
    """
    viability = row["PredictedViability_2026"]
    model = _model_for_row(row, species_models, origin_models, global_model)
    slope = model.slope_at(row["SeedAge"])

    if slope >= 0:
        return np.inf

    years_remaining = viability / abs(slope)
    return max(0, years_remaining)


def determine_primary_reason(
    row: pd.Series,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
) -> str:
    """Human-readable summary of why an accession is flagged as regeneration priority."""
    reasons = []

    if row["SeedAge"] >= 75:
        reasons.append("Extreme seed age")
    elif row["SeedAge"] >= 40:
        reasons.append("Old seed age")

    # Compare instantaneous rates of decline at this row's own age -- the
    # only fair way to compare two curves' local steepness at a shared point,
    # now that neither has a single constant "the slope."
    if row["ModelUsed"] == "Species":
        slope = species_models[row["Species"]].slope_at(row["SeedAge"])
        if slope < global_model.slope_at(row["SeedAge"]) * 1.5:
            reasons.append("Fast species deterioration")

    if row["ModelUsed"] == "Origin":
        slope = origin_models[row["Origin"]].slope_at(row["SeedAge"])
        if slope < global_model.slope_at(row["SeedAge"]) * 1.5:
            reasons.append("Fast origin deterioration")

    # Distinguish a directly-observed result from a purely modeled one --
    # matters for triage credibility either way: an assumed value (see
    # grin_import.py's low-germination imputation, for a documented concern
    # with no recorded percentage) shouldn't read as equally certain as a
    # real lab measurement, and a real low measurement deserves to stand out
    # from a merely-predicted one.
    if row.get("ViabilityAssumed", False):
        reasons.append("Assumed low germination, no test data")
    elif pd.notna(row.get("Viability")) and pd.notna(row.get("AgeAtTest")) and row["Viability"] <= 30:
        reasons.append("Tested at low viability")

    if row["PredictedViability_2026"] <= 10:
        reasons.append("Critical viability")
    elif row["PredictedViability_2026"] <= 30:
        reasons.append("Low viability")

    if not reasons:
        reasons.append("General deterioration")

    return "; ".join(reasons)


def build_priority_table(
    df_ranking: pd.DataFrame,
    species_models: dict[str, CurveModel],
    origin_models: dict[str, CurveModel],
    global_model: CurveModel,
    top_n: int = 50,
) -> pd.DataFrame:
    """The N lowest-predicted-viability accessions, with deterioration diagnostics attached.

    df_ranking must already have PredictedViability_2026 and ModelUsed columns
    (see hierarchical.predict_hierarchical).
    """
    df_ranking = df_ranking.copy()
    for col in _LOCATION_COLUMNS:
        if col not in df_ranking.columns:
            df_ranking[col] = ""

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
