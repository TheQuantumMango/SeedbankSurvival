"""Characterization tests: the new src/seedbank_survival modules must reproduce
tests/legacy_baseline/pipeline.py (the oracle) for the parts of the pipeline that
haven't been deliberately changed -- currently that's row-level data cleaning and
WHICH points feed a given group's model (group fitting itself never looks at
Status). Several things have since been deliberately changed on top of that --
the Status whitelist, depleted-lot fallback, real-seed broadening, and (most
recently) the deterioration curve's functional form itself (linear -> quadratic,
see deterioration.py's CurveModel) -- each asserted directly below as an
intentional divergence, not treated as a failure.

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
  Species epsilon (Origin E): 4 qualifying rows with viability *increasing*
    with age (perfectly linear, slope = +1.0) -> gets a Species-tier model
    with a non-negative slope, exercising the years-to-zero "inf" branch.
    4, not 3: a quadratic fit (deterioration.py's CurveModel) on exactly 3
    points has zero residual degrees of freedom and is always degenerate
    (r2=1.0, undefined p-value) regardless of the true underlying
    relationship -- 4 is the minimum for the fit to mean anything.
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

  Depleted-lot ranking fallback (data_prep.build_ranking_dataset):
  ACC-P has two rows -- SeedAge 5 with EstTotalSeed 0 (depleted), and
  SeedAge 15 with EstTotalSeed 500 (has seed). The oracle (dumb
  youngest-wins) picks the depleted SeedAge-5 row; the fixed pipeline
  skips it and picks the non-depleted SeedAge-15 row instead.
  ACC-L1/ACC-M1 have EstTotalSeed 0 -- realistic for "Exhausted supply"/
  "Lot used for regeneration" -- so they stay excluded for two
  independent reasons (bad Status AND no seed), not just one.

  Real-seed-OR-whitelisted-status broadening (data_prep.build_ranking_dataset):
  ACC-Q (Status "Unknown status", not whitelisted, EstTotalSeed 500):
  excluded by Status alone, but included because it has real seed on
  hand -- proves the OR-gate, not just the Status whitelist, drives
  inclusion now.
"""
from __future__ import annotations

import pandas as pd

import seedbank_survival as sb
from tests.legacy_baseline.pipeline import run_legacy_pipeline


def run_modular_pipeline(df: pd.DataFrame) -> dict:
    df_clean = sb.clean_ages(df)
    df_model = sb.build_model_dataset(df_clean)
    df_ranking = sb.build_ranking_dataset(df_clean)

    global_model = sb.fit_global_model(df_model)
    species_models = sb.fit_group_models(df_model, "Species", min_n=4)
    origin_models = sb.fit_group_models(df_model, "Origin", min_n=4)

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


def test_model_fitting_group_membership_matches_legacy_oracle(synthetic_accessions_df):
    """Model fitting never looks at Status, so WHICH rows feed a given group's
    fit (n) must still match the oracle exactly, for every group both the
    oracle (min_n=3) and the modular pipeline (min_n=4) agree reaches
    threshold. The fitted VALUES (intercept/slope) are NOT compared here --
    the oracle fits a straight line and the modular pipeline now fits a
    quadratic curve (see deterioration.py's CurveModel), a later, deliberate
    divergence layered on top of this one, so the two are no longer the same
    kind of number."""
    oracle = run_legacy_pipeline(synthetic_accessions_df)
    modular = run_modular_pipeline(synthetic_accessions_df)

    common_species = set(modular["species_models"]) & set(oracle["species_models"])
    assert common_species, "expected at least one species both pipelines fit"
    for species in common_species:
        assert modular["species_models"][species].n == oracle["species_models"][species]["n"]

    common_origins = set(modular["origin_models"]) & set(oracle["origin_models"])
    for origin in common_origins:
        assert modular["origin_models"][origin].n == oracle["origin_models"][origin]["n"]


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
        "ACC-E1", "ACC-E2", "ACC-E3", "ACC-E4",
        "ACC-F1", "ACC-G1", "ACC-N1",  # newly included by the whitelist fix
        "ACC-P",  # depleted-lot fallback case, still Species alpha either way
        "ACC-Q",  # included via the real-seed-OR-whitelisted-status broadening
    ]
    # ACC-B1/ACC-C1/ACC-C2 would naively fall to the Origin CountryB model
    # (Species beta/gamma have too few rows each), but CountryB only has 3
    # rows combined -- one short of min_n=4, so no Origin CountryB model is
    # even attempted (a quadratic fit on exactly 3 points is always
    # degenerate anyway; see deterioration.py's CurveModel). Falls straight
    # to Global.
    expected_origin_tier = []
    expected_global_tier = ["ACC-K", "ACC-B1", "ACC-C1", "ACC-C2"]

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

    for accession in ["ACC-E1", "ACC-E2", "ACC-E3", "ACC-E4"]:
        assert pd.isna(table.loc[accession, "YearsRemainingTo0%"])
        assert table.loc[accession, "PrimaryReason"] == "General deterioration"


def test_extreme_seed_age_reason_text(synthetic_accessions_df):
    modular = run_modular_pipeline(synthetic_accessions_df)
    table = modular["priority_table"].set_index("Accession")
    assert "Extreme seed age" in table.loc["ACC-J1", "PrimaryReason"]


def test_depleted_lot_fallback_diverges_from_oracle(synthetic_accessions_df):
    """The oracle (dumb youngest-wins) picks ACC-P's depleted SeedAge-5 lot;
    the fixed build_ranking_dataset skips it for the non-depleted SeedAge-15
    lot instead -- a deliberate change, not a bug."""
    oracle = run_legacy_pipeline(synthetic_accessions_df)
    modular = run_modular_pipeline(synthetic_accessions_df)

    oracle_row = oracle["df_ranking"].set_index("Accession").loc["ACC-P"]
    modular_row = modular["df_ranking"].set_index("Accession").loc["ACC-P"]

    assert oracle_row["SeedAge"] == 5
    assert modular_row["SeedAge"] == 15


def test_priority_table_fills_blank_location_columns_when_absent_from_source(synthetic_accessions_df):
    """The reformatted CSV/XLSX path (this fixture's shape) has no per-inventory
    location data at all -- unlike the raw-GRIN path (grin_import.adapt_raw_export),
    which populates MaintenanceSite/Location for real. build_priority_table must
    still succeed and fill both columns blank rather than raising a KeyError."""
    modular = run_modular_pipeline(synthetic_accessions_df)
    table = modular["priority_table"]
    assert "MaintenanceSite" in table.columns
    assert "Location" in table.columns
    assert (table["MaintenanceSite"] == "").all()
    assert (table["Location"] == "").all()


def test_real_seed_includes_accession_despite_unwhitelisted_status(synthetic_accessions_df):
    """ACC-Q has a Status the whitelist has never covered ("Unknown status") but
    real seed on hand (EstTotalSeed 500) -- included via the OR-gate, absent
    from the oracle (which only ever looks at Status)."""
    oracle = run_legacy_pipeline(synthetic_accessions_df)
    modular = run_modular_pipeline(synthetic_accessions_df)

    assert "ACC-Q" not in set(oracle["df_ranking"]["Accession"])
    assert "ACC-Q" in set(modular["df_ranking"]["Accession"])
