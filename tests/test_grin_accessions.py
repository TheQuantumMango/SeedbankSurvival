from __future__ import annotations

import pandas as pd

from seedbank_survival.grin_accessions import build_received_year_lookup


def _accession_row(**overrides):
    row = {
        "Accession": "PI 100000",
        "Taxon": "Astragalus cicer",
        "Received Date Format": "mm/dd/yyyy",
        "Received Date": pd.Timestamp("1990-02-05"),
    }
    row.update(overrides)
    return row


def test_build_received_year_lookup_extracts_year():
    df = pd.DataFrame([_accession_row()])
    assert build_received_year_lookup(df) == {"PI 100000": 1990}


def test_build_received_year_lookup_handles_all_date_formats():
    df = pd.DataFrame(
        [
            _accession_row(Accession="A1", **{"Received Date Format": "mm/dd/yyyy", "Received Date": pd.Timestamp("1990-02-05")}),
            _accession_row(Accession="A2", **{"Received Date Format": "yyyy", "Received Date": pd.Timestamp("1985-01-01")}),
            _accession_row(Accession="A3", **{"Received Date Format": "mm/yyyy", "Received Date": pd.Timestamp("1978-06-01")}),
        ]
    )
    result = build_received_year_lookup(df)
    assert result == {"A1": 1990, "A2": 1985, "A3": 1978}


def test_build_received_year_lookup_drops_missing_dates():
    df = pd.DataFrame(
        [
            _accession_row(Accession="A1"),
            _accession_row(Accession="A2", **{"Received Date": pd.NaT}),
        ]
    )
    result = build_received_year_lookup(df)
    assert result == {"A1": 1990}
    assert "A2" not in result
