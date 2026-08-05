"""Runs the actual CLI against real local GRIN-Global export data, if present.

docs/ is gitignored (private, local-only), so this test self-skips in a fresh
clone or CI, matching test_real_data_smoke.py / test_raw_grin_export_smoke.py.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from seedbank_survival.cli import main

DOCS_DIR = Path(__file__).parents[1] / "docs"
RAW_EXPORT_PATH = DOCS_DIR / "RawGRINAstragalusExport.xlsx"
ACCESSIONS_PATH = DOCS_DIR / "RawGRINAstragalusExportAccessions.xlsx"


@pytest.mark.skipif(
    not RAW_EXPORT_PATH.exists(),
    reason="requires local docs/RawGRINAstragalusExport.xlsx (gitignored, not present in a fresh clone)",
)
def test_cli_runs_end_to_end_on_real_data(tmp_path, capsys):
    argv = [
        "--inventory", str(RAW_EXPORT_PATH),
        "--genus", "Astragalus",
        "--as-of-year", "2026",
        "--out-dir", str(tmp_path),
    ]
    if ACCESSIONS_PATH.exists():
        argv += ["--accessions", str(ACCESSIONS_PATH)]

    exit_code = main(argv)

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Accession view" in out
    assert "Inventory view" in out

    accession_csv = pd.read_csv(tmp_path / "accession_view.csv")
    inventory_csv = pd.read_csv(tmp_path / "inventory_view.csv")
    assert len(accession_csv) > 0
    assert len(inventory_csv) >= len(accession_csv)

    report_html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "SeedbankSurvival report" in report_html
    assert 'href="http' not in report_html
    assert 'src="http' not in report_html
