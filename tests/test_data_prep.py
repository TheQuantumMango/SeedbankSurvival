from __future__ import annotations

import pandas as pd

from seedbank_survival.data_prep import build_ranking_dataset


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
