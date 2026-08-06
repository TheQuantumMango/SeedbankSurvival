from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import pytest

from seedbank_survival.cli import run


def _inventory_row(**overrides):
    row = {
        "Taxon": "Astragalus cicer",
        "Inventory Type": "SD",
        "Inventory Suffix": "37o",
        "Accession": "PI 100000",
        "Inventory Status": "Available",
        "Percent Viable": 90.0,
        "Tested Date": pd.Timestamp("2003-04-22"),
        "Origin": "Turkey",
        "Quantity On Hand": 1000.0,
        "Inventory Maintenance Site": "W6",
        "Location Section 1": "minus20",
        "Location Section 2": None,
        "Location Section 3": None,
        "Location Section 4": None,
        "Status Note": None,
        "Web Availability Note": None,
        "Note": None,
    }
    row.update(overrides)
    return row


def _write_inventory_xlsx(path: Path, rows: list[dict]) -> Path:
    file_path = path / "inventory.xlsx"
    pd.DataFrame(rows).to_excel(file_path, index=False)
    return file_path


def _base_args(**overrides) -> argparse.Namespace:
    defaults = {
        "inventory": None,
        "accessions": None,
        "genera": None,
        "list_genera": False,
        "as_of_year": 2026,
        "out_dir": None,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


@pytest.fixture
def three_species_rows():
    # Enough rows for a real Species-tier model (min_n=3) so the pipeline has
    # something non-trivial to fit and predict for. Lot years (from the
    # suffix) must precede the default Tested Date's year (2003) so
    # AgeAtTest = ViabilityYear - lot_year comes out positive.
    return [
        _inventory_row(Accession=f"PI {i}", **{"Inventory Suffix": suffix, "Percent Viable": v})
        for i, (suffix, v) in enumerate(
            [("90o", 90.0), ("95o", 70.0), ("99o", 50.0)], start=1
        )
    ]


def test_list_genera_prints_and_writes_nothing(tmp_path, capsys, three_species_rows):
    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    args = _base_args(inventory=inventory_path, list_genera=True, out_dir=tmp_path / "out")

    exit_code = run(args)

    assert exit_code == 0
    assert "Astragalus" in capsys.readouterr().out
    assert not (tmp_path / "out").exists()


def test_missing_genus_without_list_genera_errors(tmp_path, capsys, three_species_rows):
    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    args = _base_args(inventory=inventory_path, out_dir=tmp_path / "out")

    exit_code = run(args)

    assert exit_code == 1
    assert "--genus is required" in capsys.readouterr().err


def test_genus_matching_zero_rows_errors_with_known_genera_listed(tmp_path, capsys, three_species_rows):
    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    args = _base_args(inventory=inventory_path, genera=["Nonexistent"], out_dir=tmp_path / "out")

    exit_code = run(args)

    stderr = capsys.readouterr().err
    assert exit_code == 1
    assert "no rows matched" in stderr
    assert "Astragalus" in stderr


def test_happy_path_writes_all_three_outputs(tmp_path, three_species_rows):
    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    out_dir = tmp_path / "out"
    args = _base_args(inventory=inventory_path, genera=["Astragalus"], out_dir=out_dir)

    exit_code = run(args)

    assert exit_code == 0
    assert (out_dir / "accession_view.csv").exists()
    assert (out_dir / "inventory_view.csv").exists()
    assert (out_dir / "report.html").exists()

    accession_csv = pd.read_csv(out_dir / "accession_view.csv")
    assert len(accession_csv) == 3
    assert "ModelConfidence" in accession_csv.columns
    assert "Location" in accession_csv.columns
    assert "MaintenanceSite" in accession_csv.columns
    assert (accession_csv["MaintenanceSite"] == "W6").all()
    assert (accession_csv["Location"] == "minus20").all()


def test_as_of_year_defaults_to_current_year_when_omitted(tmp_path, three_species_rows, monkeypatch):
    import datetime

    class _FixedDate(datetime.date):
        @classmethod
        def today(cls):
            return cls(2030, 1, 1)

    monkeypatch.setattr("seedbank_survival.cli.date", _FixedDate)

    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    out_dir = tmp_path / "out"
    args = _base_args(inventory=inventory_path, genera=["Astragalus"], out_dir=out_dir, as_of_year=None)

    run(args)

    accession_csv = pd.read_csv(out_dir / "accession_view.csv")
    # SeedAge = as_of_year - lot_year; lot_year for "90o" is 1990.
    assert accession_csv.set_index("Accession").loc["PI 1", "SeedAge"] == 2030 - 1990


def test_accessions_file_is_optional(tmp_path, three_species_rows):
    inventory_path = _write_inventory_xlsx(tmp_path, three_species_rows)
    out_dir = tmp_path / "out"
    args = _base_args(inventory=inventory_path, genera=["Astragalus"], accessions=None, out_dir=out_dir)

    exit_code = run(args)

    assert exit_code == 0
    assert (out_dir / "report.html").exists()
