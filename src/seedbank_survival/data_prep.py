from __future__ import annotations

from pathlib import Path

import pandas as pd

# NOTE: "Low viability" never matches real GRIN-Global data, which uses
# "Low germination" instead -- this whitelist has a real mismatch bug,
# inherited verbatim from the original notebook (it also omits several
# real status values, e.g. "Not available", "Not viable", "Exhausted
# supply"). Preserved here on purpose; see tests/legacy_baseline for the
# characterization test that locks in this exact behavior before it's
# ever fixed deliberately.
DEFAULT_VALID_STATUSES = (
    "Available",
    "Original lot received",
    "Low inventory",
    "Low viability",
    "Packaged",
    "Quantity low, but inventory onhand above iv.dcritical",
    "Replaced older sample with newer sample",
)


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
