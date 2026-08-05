from __future__ import annotations

import pandas as pd

from seedbank_survival.data_prep import build_inventory_view, build_ranking_dataset


def _row(**overrides):
    row = {
        "Accession": "A1",
        "SeedAge": 10,
        "Status": "Available",
        "EstTotalSeed": 500,
        "Viability": 90.0,
    }
    row.update(overrides)
    return row


def test_whitelisted_status_alone_is_included():
    df = pd.DataFrame([_row(Status="Available", EstTotalSeed=0)])
    result = build_ranking_dataset(df)
    assert "A1" in result["Accession"].values


def test_real_seed_alone_is_included_despite_bad_status():
    df = pd.DataFrame([_row(Status="Unknown status", EstTotalSeed=500)])
    result = build_ranking_dataset(df)
    assert "A1" in result["Accession"].values


def test_neither_real_seed_nor_whitelisted_status_is_excluded():
    df = pd.DataFrame([_row(Status="Unknown status", EstTotalSeed=0)])
    result = build_ranking_dataset(df)
    assert result.empty


def test_inventory_view_excludes_whitelisted_but_empty_row_ranking_would_keep():
    # Whitelisted Status but no seed on hand: build_ranking_dataset still
    # surfaces it (its depleted-lot fallback -- see that function's tests),
    # but there's no physical packet here for the Inventory view to show.
    df = pd.DataFrame([_row(Status="Available", EstTotalSeed=0)])
    assert "A1" in build_ranking_dataset(df)["Accession"].values
    assert build_inventory_view(df).empty


def test_inventory_view_excludes_rows_with_no_seed():
    df = pd.DataFrame([_row(EstTotalSeed=0), _row(Accession="A2", EstTotalSeed=None)])
    result = build_inventory_view(df)
    assert result.empty


def test_inventory_view_keeps_every_packet_no_dedup():
    df = pd.DataFrame(
        [
            _row(Accession="A1", SeedAge=5, EstTotalSeed=100),
            _row(Accession="A1", SeedAge=20, EstTotalSeed=200),
        ]
    )
    result = build_inventory_view(df)
    assert len(result) == 2


def test_inventory_view_keeps_zero_viability_packets():
    df = pd.DataFrame([_row(Viability=0.0, EstTotalSeed=500)])
    result = build_inventory_view(df)
    assert "A1" in result["Accession"].values
