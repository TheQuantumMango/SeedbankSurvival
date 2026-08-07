from __future__ import annotations

import numpy as np
import pandas as pd

from .deterioration import Curve

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

# Beyond this, a computed years-to-zero stops being a meaningful distinction
# for a seed bank's realistic planning horizon (decades, not centuries) --
# see estimate_years_to_zero.
_YEARS_TO_ZERO_CAP = 200.0


def _model_for_row(
    row: pd.Series,
    species_models: dict[str, Curve],
    origin_models: dict[str, Curve],
    global_model: Curve,
) -> Curve:
    if row["ModelUsed"] == "Species":
        return species_models[row["Species"]]
    if row["ModelUsed"] == "Origin":
        return origin_models[row["Origin"]]
    return global_model


def estimate_years_to_zero(
    row: pd.Series,
    species_models: dict[str, Curve],
    origin_models: dict[str, Curve],
    global_model: Curve,
) -> float:
    """Years until predicted viability reaches 0%, given the model tier that produced it.

    A quadratic curve's rate of decline isn't constant, so this uses the
    curve's INSTANTANEOUS slope at the row's current age as a local linear
    approximation of the near-term trajectory, rather than solving for where
    the full curve eventually crosses zero (which can have 0, 1, or 2 real
    roots, or curve back upward -- not a meaningful "years remaining" for a
    curator's near-term planning). Reduces to the exact previous calculation
    when the curve is a straight line (slope_at is then constant everywhere).

    slope >= 0 normally means "not currently declining," correctly inf. But
    a small-n quadratic can have a locally positive slope_at() PAST its
    vertex -- even a barely-positive one -- while the row's own (separately
    bounded, see hierarchical.py) PredictedViability_2026 already sits at
    the floor -- verified against real data (2 accessions predicted ~0.2%
    and ~1.0% showed inf/blank years-to-zero, contradicting their own
    "Critical viability" tag). Report 0 in that specific case rather than
    trust a local slope reading the row's own prediction has already
    effectively contradicted; every other slope>=0 case (viability
    genuinely not near the floor) is untouched.
    """
    viability = row["PredictedViability_2026"]
    model = _model_for_row(row, species_models, origin_models, global_model)
    slope = model.slope_at(row["SeedAge"])

    if slope >= 0:
        return 0.0 if viability <= 2.0 else np.inf

    years_remaining = viability / abs(slope)
    if years_remaining > _YEARS_TO_ZERO_CAP:
        # A near-flat (but technically negative) local slope near a weak
        # curve's vertex -- verified against real data, one Global-tier row
        # (r2=0.035) computed a literal 111,377.7 -- is mathematically
        # "correct" but not a meaningful distinction from "not currently
        # declining" for a curator's planning horizon, and the false
        # precision (one decimal place on a 6-figure number) reads as a
        # glitch rather than a real estimate. Reported the same as inf.
        return np.inf
    return max(0, years_remaining)


def determine_primary_reason(
    row: pd.Series,
    species_models: dict[str, Curve],
    origin_models: dict[str, Curve],
    global_model: Curve,
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
    species_models: dict[str, Curve],
    origin_models: dict[str, Curve],
    global_model: Curve,
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
