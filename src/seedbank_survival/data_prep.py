from __future__ import annotations

from pathlib import Path

import pandas as pd

# Statuses observed across real GRIN-Global exports (Astragalus, Onobrychis)
# believed to mean seed physically remains in inventory -- i.e. worth
# including in the regeneration-priority ranking pool. Replaces the original
# notebook's whitelist, which had "Low viability" (never matches real data;
# the actual value is "Low germination") and omitted most other real statuses.
# See tests/legacy_baseline for the characterization test proving the
# original bug is preserved there and fixed here, deliberately.
DEFAULT_VALID_STATUSES = (
    "Available",
    "Original lot received",
    "Low inventory",
    "Low germination",
    "Packaged",
    "Quantity low, but inventory onhand above iv.dcritical",
    "Replaced older sample with newer sample",
    # Seed exists, tested at ~0% viability -- exactly the kind of accession
    # this tool exists to flag, not exclude.
    "Not viable",
    # Administrative states where seed is presumed to still be on hand;
    # low row counts in practice (1-3 rows per status across both real
    # exports), worth revisiting if that turns out to be wrong.
    "Hold inventory sample for further clarification",
    "Curator attention required",
    "Extra seed: was or will be distribution",
)

# Deliberately excluded -- no seed presumed to remain, or existence
# unconfirmed:
#   "Exhausted supply"                    -- supply is gone
#   "Lot used for regeneration"           -- original lot consumed to grow a
#   "Planted for regeneration"               new one; that new lot gets its
#                                             own Accession/Status once grown
#   "No harvest made"                     -- regeneration attempt consumed
#                                             the lot and produced nothing
#   "Not available", "Inventory does not exist" -- explicitly gone
#   "Unknown status"                      -- existence unconfirmed


def load_accessions(path: str | Path) -> pd.DataFrame:
    """Load a raw accession export. Dispatches on file extension, never column position."""
    path = Path(path)
    if path.suffix.lower() == ".xlsx":
        return pd.read_excel(path)
    return pd.read_csv(path)


def clean_ages(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce SeedAge/AgeAtTest to numeric and drop rows with no usable SeedAge."""
    df = df.copy()
    df["SeedAge"] = pd.to_numeric(df["SeedAge"], errors="coerce")
    df["AgeAtTest"] = pd.to_numeric(df["AgeAtTest"], errors="coerce")
    df = df.dropna(subset=["SeedAge"]).reset_index(drop=True)
    return df


def build_model_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Rows usable for fitting deterioration models: a real test age and a measured viability."""
    return df[(df["AgeAtTest"] > 0) & (df["Viability"].notna())].copy()


def build_ranking_dataset(
    df: pd.DataFrame, valid_statuses: tuple[str, ...] = DEFAULT_VALID_STATUSES
) -> pd.DataFrame:
    """One row per Accession (lowest SeedAge) among rows with an in-inventory Status."""
    df_exists = df[df["Status"].isin(valid_statuses)].copy()
    # Dead in practice: SeedAge-null rows are already dropped by clean_ages()
    # before this function runs. Kept for parity with the original notebook.
    df_exists["SeedAge"] = df_exists["SeedAge"].fillna(200)
    df_ranking = df_exists.loc[
        df_exists.groupby("Accession")["SeedAge"].idxmin()
    ].reset_index(drop=True)
    return df_ranking
