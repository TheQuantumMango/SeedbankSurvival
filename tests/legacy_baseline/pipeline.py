"""Near-verbatim transcription of DATA115_Final4_Charpentier.ipynb's core computational
path (Phases 1, 2.1, 2.6, 2.7 -- the parts that produce the priority table). Used only
as a fixed oracle by tests/test_legacy_baseline.py, to verify the src/seedbank_survival
modules reproduce current notebook behavior -- known bugs included -- before any future
refactor is allowed to change behavior on purpose.

Deliberately NOT covered here (out of scope for this refactor step, left in the archived
notebook): the ANOVA group-effect diagnostics (Phase 2.3) and the separate whole-dataset
descriptive viability estimate (Phase 2.2). Neither feeds the priority table, and Phase
2.2 is never reconciled with the hierarchical prediction path used here.

Do NOT "fix" anything in this file -- it defines what "current behavior" means. Bugs get
fixed in src/seedbank_survival/*, and this file is what proves a given fix is deliberate.

Two intentional deviations from the notebook, both behavior-neutral:
  1. Takes a DataFrame parameter instead of reading a hardcoded Downloads path.
  2. All plt.*/print() calls are dropped (no effect on returned data).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols

# Copied verbatim from the notebook. "Low viability" never matches real data
# (which uses "Low germination"); several real statuses are omitted entirely.
# This is a known bug, preserved on purpose -- see test_legacy_baseline.py.
VALID_STATUSES = [
    "Available",
    "Original lot received",
    "Low inventory",
    "Low viability",
    "Packaged",
    "Quantity low, but inventory onhand above iv.dcritical",
    "Replaced older sample with newer sample",
]


def run_legacy_pipeline(df: pd.DataFrame) -> dict:
    # --- Phase 1: data preparation & aggregation ---
    df = df.copy()
    df["SeedAge"] = pd.to_numeric(df["SeedAge"], errors="coerce")
    df["AgeAtTest"] = pd.to_numeric(df["AgeAtTest"], errors="coerce")
    df = df.dropna(subset=["SeedAge"]).reset_index(drop=True)

    df_model = df[(df["AgeAtTest"] > 0) & (df["Viability"].notna())].copy()

    df_exists = df[df["Status"].isin(VALID_STATUSES)].copy()
    df_exists["SeedAge"].fillna(200, inplace=True)

    df_ranking = df_exists.loc[
        df_exists.groupby("Accession")["SeedAge"].idxmin()
    ].reset_index(drop=True)

    # --- Phase 2.1: global deterioration model ---
    df_deterioration = df_model
    model_global = ols("Viability ~ AgeAtTest", data=df_deterioration).fit()
    global_intercept = model_global.params["Intercept"]
    global_slope = model_global.params["AgeAtTest"]

    # --- Phase 2.6: hierarchical species -> origin -> global models ---
    species_models = {}
    for species in df_deterioration["Species"].dropna().unique():
        data_sp = df_deterioration[df_deterioration["Species"] == species]
        if len(data_sp) >= 3:
            model_sp = ols("Viability ~ AgeAtTest", data=data_sp).fit()
            species_models[species] = {
                "intercept": model_sp.params["Intercept"],
                "slope": model_sp.params["AgeAtTest"],
                "n": len(data_sp),
                "r2": model_sp.rsquared,
            }

    origin_models = {}
    for origin in df_deterioration["Origin"].dropna().unique():
        data_or = df_deterioration[df_deterioration["Origin"] == origin]
        if len(data_or) >= 3:
            model_or = ols("Viability ~ AgeAtTest", data=data_or).fit()
            origin_models[origin] = {
                "intercept": model_or.params["Intercept"],
                "slope": model_or.params["AgeAtTest"],
                "n": len(data_or),
                "r2": model_or.rsquared,
            }

    def predict_viability_hierarchical(row):
        age = row["SeedAge"]
        if row["Species"] in species_models:
            intercept = species_models[row["Species"]]["intercept"]
            slope = species_models[row["Species"]]["slope"]
            model_used = "Species"
        elif row["Origin"] in origin_models:
            intercept = origin_models[row["Origin"]]["intercept"]
            slope = origin_models[row["Origin"]]["slope"]
            model_used = "Origin"
        else:
            intercept = global_intercept
            slope = global_slope
            model_used = "Global"

        predicted = intercept + (slope * age)
        predicted = np.clip(predicted, 0, 100)
        return pd.Series([predicted, model_used])

    df_ranking[["PredictedViability_2026", "ModelUsed"]] = df_ranking.apply(
        predict_viability_hierarchical, axis=1
    )

    # --- Phase 2.7: priority ranking ---
    def estimate_years_to_zero(row):
        viability = row["PredictedViability_2026"]
        if row["ModelUsed"] == "Species":
            slope = species_models[row["Species"]]["slope"]
        elif row["ModelUsed"] == "Origin":
            slope = origin_models[row["Origin"]]["slope"]
        else:
            slope = global_slope

        if slope >= 0:
            return np.inf

        years_remaining = viability / abs(slope)
        return max(0, years_remaining)

    def determine_primary_reason(row):
        reasons = []
        if row["SeedAge"] >= 75:
            reasons.append("Extreme seed age")
        elif row["SeedAge"] >= 40:
            reasons.append("Old seed age")

        if row["ModelUsed"] == "Species":
            slope = species_models[row["Species"]]["slope"]
            if slope < global_slope * 1.5:
                reasons.append("Fast species deterioration")

        if row["ModelUsed"] == "Origin":
            slope = origin_models[row["Origin"]]["slope"]
            if slope < global_slope * 1.5:
                reasons.append("Fast origin deterioration")

        if row["PredictedViability_2026"] <= 10:
            reasons.append("Critical viability")
        elif row["PredictedViability_2026"] <= 30:
            reasons.append("Low viability")

        if len(reasons) == 0:
            reasons.append("General deterioration")

        return "; ".join(reasons)

    df_ranking["YearsToZero"] = df_ranking.apply(estimate_years_to_zero, axis=1)
    df_ranking["PrimaryReason"] = df_ranking.apply(determine_primary_reason, axis=1)

    priority_table = df_ranking.sort_values("PredictedViability_2026").head(50).copy()
    priority_table = priority_table[
        [
            "Accession",
            "Species",
            "Origin",
            "SeedAge",
            "PrimaryReason",
            "PredictedViability_2026",
            "YearsToZero",
            "ModelUsed",
        ]
    ]
    priority_table.columns = [
        "Accession",
        "Species",
        "Origin",
        "SeedAge",
        "PrimaryReason",
        "EstimatedViability_2026",
        "YearsRemainingTo0%",
        "ModelUsed",
    ]
    priority_table["EstimatedViability_2026"] = priority_table[
        "EstimatedViability_2026"
    ].round(1)
    priority_table["YearsRemainingTo0%"] = (
        priority_table["YearsRemainingTo0%"].replace(np.inf, np.nan).round(1)
    )

    return {
        "df_cleaned": df,
        "df_model": df_model,
        "df_exists": df_exists,
        "df_ranking": df_ranking,
        "global_intercept": global_intercept,
        "global_slope": global_slope,
        "species_models": species_models,
        "origin_models": origin_models,
        "priority_table": priority_table,
    }
