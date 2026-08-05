"""Adapts a raw GRIN-Global accession-level export (one row per Accession,
distinct from the inventory export's one row per seed lot) into a small
lookup used to fill in a lot year when nothing else can.

Genus filtering reuses grin_import.filter_to_genus/list_genera directly --
this file has the same Taxon column and the same bad-taxa contamination
problem as the inventory export.
"""
from __future__ import annotations

import pandas as pd


def build_received_year_lookup(df_accessions: pd.DataFrame) -> dict[str, int]:
    """Accession -> the year its Received Date falls in.

    Received Date Format ("mm/dd/yyyy" / "yyyy" / "mm/yyyy") is deliberately
    not consulted here -- it only reflects how much of the date is really
    known, and every observed value still resolves to a correct year, which
    is all this lookup needs. Rows with a missing Received Date are dropped.
    """
    received = pd.to_datetime(df_accessions["Received Date"], errors="coerce")
    years = received.dt.year
    lookup = dict(zip(df_accessions["Accession"], years))
    return {accession: int(year) for accession, year in lookup.items() if pd.notna(year)}
