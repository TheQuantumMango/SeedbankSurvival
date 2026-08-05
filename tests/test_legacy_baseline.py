"""Characterization tests: the new src/seedbank_survival modules must reproduce
tests/legacy_baseline/pipeline.py (the oracle) for the parts of the pipeline that
haven't been deliberately changed yet -- currently that's everything upstream of
the Status whitelist (data cleaning, model fitting). The Status whitelist itself
was fixed in data_prep.DEFAULT_VALID_STATUSES (see that module for the full
included/excluded reasoning), so ranking membership is now EXPECTED to diverge
from the oracle for specific accessions -- that divergence is asserted directly
below, not treated as a failure.

tests/fixtures/synthetic_accessions.csv is small and hand-designed so every
branch below can be reasoned about directly, without needing to compute
regression coefficients by hand:

  Species alpha (Origin A): qualifying rows (viability clearly decreasing
    with age) -> gets its own Species-tier model. Includes several rows whose
    only purpose is exercising Status filtering (see below), which all still
    count toward the Species alpha model fit since model fitting never looks
    at Status.
  Species beta / gamma (Origin B): 1 and 2 qualifying rows respectively --
    neither reaches the n=3 Species threshold, but combined they give Origin
    B exactly 3 -> both species fall back to the Origin-tier model.
  Species epsilon (Origin E): 3 qualifying rows with viability *increasing*
    with age (perfectly linear, slope = +1.0) -> gets a Species-tier model
    with a non-negative slope, exercising the years-to-zero "inf" branch.
  Species kappa (Origin K): 2 rows sharing one Accession at different
    SeedAge -- exercises both the Accession dedup (idxmin) logic and the
    Global-tier fallback (neither Species nor Origin reaches n=3).
  ACC-H1 (SeedAge "unk"): non-numeric SeedAge -> coerced to NaN -> the whole
    row is dropped upstream, before any other filter runs.
  ACC-I1 (AgeAtTest 0, Viability blank): valid for ranking (Status
    "Packaged") but excluded from the model-fitting dataset.
  ACC-J1 (SeedAge 80): triggers the "Extreme seed age" reason text.

  Status-whitelist-fix coverage (all Species alpha / Origin A, so ranking
  membership is the only thing that varies):
  ACC-F1 (Status "Low germination"): the original bug -- a real GRIN status
    the old whitelist never matched. Now included.
  ACC-G1 (Status "Not viable"): seed exists, tested at ~0% viability. The
    old whitelist excluded it; now deliberately included, since this is
    exactly the kind of accession the tool exists to flag.
  ACC-L1 (Status "Exhausted supply"): deliberately excluded, unchanged.
  ACC-M1 (Status "Lot used for regeneration"): deliberately excluded --
    the original lot was consumed to grow a new one.
  ACC-N1 (Status "Hold inventory sample for further clarification"): an
    administrative-hold status, judged to mean seed is still on hand ->
    included.
"""
from __future__ import annotations

import pandas as pd
import pytest

import seedbank_survival as sb
from tests.legacy_baseline.pipeline import run_legacy_pipeline


def run_modular_pipeline(df: pd.DataFrame) -> dict:
    df_clean = sb.clean_ages(df)
    df_model = sb.build_model_dataset(df_clean)
    df_ranking = sb.build_ranking_dataset(df_clean)

    global_model = sb.fit_global_model(df_model)
    species_models = sb.fit_group_models(df_model, "Species", min_n=3)
    origin_models = sb.fit_group_models(df_model, "Origin", min_n=3)

    df_ranking = sb.predict_hierarchical(
        df_ranking, species_models, origin_models, global_model
    )
    priority_table = sb.build_priority_table(
        df_ranking, species_models, origin_models, global_model, top_n=50
    )

    return {
        "df_model": df_model,
        "df_ranking": df_ranking,
        "global_model": global_model,
        "species_models": species_models,
        "origin_models": origin_models,
        "priority_table": priority_table,
    }


def test_model_fitting_matches_legacy_oracle(synthetic_accessions_df):
    """Model fitting never looks at Status, so it must still match the oracle
    exactly even after the Status whitelist fix below."""
    oracle = run_legacy_pipeline(synthetic_accessions_df)
    modular = run_modular_pipeline(synthetic_accessions_df)

    assert modular["global_model"].intercept == pytest.approx(oracle["global_intercept"])
    assert modular["global_model"].slope == pytest.approx(oracle["global_slope"])

    assert set(modular["species_models"]) == set(oracle["species_models"])
    for species, oracle_model in oracle["species_models"].items():
        got = modular["species_models"][species]
        assert got.intercept == pytest.approx(oracle_model["intercept"])
        assert got.slope == pytest.approx(oracle_model["slope"])
        assert got.n == oracle_model["n"]
        assert got.r2 == pytest.approx(oracle_model["r2"])

    assert set(modular["origin_models"]) == set(oracle["origin_models"])
    for origin, oracle_model in oracle["origin_models"].items():
        got = modular["origin_models"][origin]
        assert got.intercept == pytest.approx(oracle_model["intercept"])
        assert got.slope == pytest.approx(oracle_model["slope"])


def test_status_whitelist_fix_changes_ranking_membership(synthetic_accessions_df):
    """The fixed whitelist (data_prep.DEFAULT_VALID_STATUSES) now includes
    "Low germination", "Not viable", and "Hold inventory..." -- all absent
    from the oracle's ranking, which still runs the original (buggy/narrower)
    whitelist. "Exhausted supply" and "Lot used for regeneration" stay
    excluded in both, proving those are deliberate exclusions, not
    accidental omissions."""
    oracle = run_legacy_pipeline(synthetic_accessions_df)
    modular = run_modular_pipeline(synthetic_accessions_df)

    oracle_accessions = set(oracle["df_ranking"]["Accession"])
    modular_accessions = set(modular["df_ranking"]["Accession"])

    now_included = {"ACC-F1", "ACC-G1", "ACC-N1"}
    for accession in now_included:
        assert accession not in oracle_accessions, accession
        assert accession in modular_accessions, accession

    still_excluded = {"ACC-L1", "ACC-M1"}
    for accession in still_excluded:
        assert accession not in oracle_accessions, accession
        assert accession not in modular_accessions, accession


def test_non_numeric_seed_age_drops_row_entirely(synthetic_accessions_df):
    df_clean = sb.clean_ages(synthetic_accessions_df)
    assert "ACC-H1" not in df_clean["Accession"].values


def test_accession_dedup_keeps_lowest_seed_age(synthetic_accessions_df):
    modular = run_modular_pipeline(synthetic_accessions_df)
    k_rows = modular["df_ranking"][modular["df_ranking"]["Accession"] == "ACC-K"]
    assert len(k_rows) == 1
    assert k_rows.iloc[0]["SeedAge"] == 12


def test_model_tier_selection(synthetic_accessions_df):
    modular = run_modular_pipeline(synthetic_accessions_df)
    by_accession = modular["df_ranking"].set_index("Accession")["ModelUsed"]

    expected_species_tier = [
        "ACC-A1", "ACC-A2", "ACC-A3", "ACC-A4", "ACC-I1", "ACC-J1",
        "ACC-E1", "ACC-E2", "ACC-E3",
        "ACC-F1", "ACC-G1", "ACC-N1",  # newly included by the whitelist fix
    ]
    expected_origin_tier = ["ACC-B1", "ACC-C1", "ACC-C2"]
    expected_global_tier = ["ACC-K"]

    for accession in expected_species_tier:
        assert by_accession[accession] == "Species", accession
    for accession in expected_origin_tier:
        assert by_accession[accession] == "Origin", accession
    for accession in expected_global_tier:
        assert by_accession[accession] == "Global", accession


def test_non_negative_slope_group_gives_nan_years_remaining(synthetic_accessions_df):
    """Species epsilon's viability rises with age (positive slope) -- years-to-zero
    is undefined (np.inf internally), which becomes NaN in the exported table."""
    modular = run_modular_pipeline(synthetic_accessions_df)
    table = modular["priority_table"].set_index("Accession")

    for accession in ["ACC-E1", "ACC-E2", "ACC-E3"]:
        assert pd.isna(table.loc[accession, "YearsRemainingTo0%"])
        assert table.loc[accession, "PrimaryReason"] == "General deterioration"


def test_extreme_seed_age_reason_text(synthetic_accessions_df):
    modular = run_modular_pipeline(synthetic_accessions_df)
    table = modular["priority_table"].set_index("Accession")
    assert "Extreme seed age" in table.loc["ACC-J1", "PrimaryReason"]
