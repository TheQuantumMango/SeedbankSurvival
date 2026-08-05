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
    """One representative row per Accession among rows worth tracking.

    A row is worth tracking if its Status is whitelisted OR it has real seed
    on hand (EstTotalSeed > 0) -- an accession with physical seed is never
    dropped just because its Status text isn't one this tool recognizes;
    Status text can be stale or inconsistent in ways a nonzero quantity
    value isn't.

    Among an Accession's tracked rows, prefers the youngest (lowest SeedAge)
    lot, UNLESS it's depleted (no seed on hand, or tested at exactly 0%
    viable) -- in that case the youngest non-depleted lot is used instead,
    since the most recent lot isn't always the right one to represent an
    accession's current state (e.g. it was harvested immature, or has no
    seed left). Falls back to the plain youngest lot if every candidate is
    depleted, so an accession that's out of usable seed everywhere still
    gets surfaced rather than silently dropped -- that's exactly the kind
    of thing this tool should flag.
    """
    worth_tracking = df["Status"].isin(valid_statuses) | (df["EstTotalSeed"] > 0)
    df_exists = df[worth_tracking].copy()
    # Dead in practice: SeedAge-null rows are already dropped by clean_ages()
    # before this function runs. Kept for parity with the original notebook.
    df_exists["SeedAge"] = df_exists["SeedAge"].fillna(200)

    depleted = (df_exists["EstTotalSeed"].isna() | (df_exists["EstTotalSeed"] == 0)) | (
        df_exists["Viability"].notna() & (df_exists["Viability"] == 0)
    )
    df_exists["_depleted"] = depleted

    df_sorted = df_exists.sort_values(["Accession", "_depleted", "SeedAge"])
    df_ranking = df_sorted.groupby("Accession").head(1)
    df_ranking = df_ranking.drop(columns="_depleted").reset_index(drop=True)
    return df_ranking
