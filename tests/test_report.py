from __future__ import annotations

import pandas as pd
import pytest

from seedbank_survival.deterioration import SlopeModel
from seedbank_survival.report import build_report

_COLUMNS = [
    "Accession", "Suffix", "Status", "Species", "Origin", "SeedAge", "PrimaryReason",
    "EstimatedViability_2026", "YearsRemainingTo0%", "ModelUsed", "ModelConfidence",
]


def _table_row(**overrides):
    row = {
        "Accession": "PI 1",
        "Suffix": "37o",
        "Status": "Available",
        "Species": "Astragalus cicer",
        "Origin": "Turkey",
        "SeedAge": 20,
        "PrimaryReason": "Critical viability",
        "EstimatedViability_2026": 5.0,
        "YearsRemainingTo0%": 2.0,
        "ModelUsed": "Species",
        "ModelConfidence": 0.75,
    }
    row.update(overrides)
    return pd.Series(row)


@pytest.fixture
def report_inputs():
    accession_table = pd.DataFrame([_table_row()])[_COLUMNS]
    inventory_table = pd.DataFrame(
        [_table_row(), _table_row(Accession="PI 1", ModelUsed="Species")]
    )[_COLUMNS]
    df_model = pd.DataFrame(
        {
            "AgeAtTest": [10, 20, 30],
            "Viability": [80.0, 60.0, 40.0],
            "SpeciesGroup": ["Astragalus cicer"] * 3,
        }
    )
    species_models = {"Astragalus cicer": SlopeModel(intercept=90, slope=-1.5, n=3, r2=0.95)}
    global_model = SlopeModel(intercept=85, slope=-1.0, n=3, r2=0.8)
    return {
        "accession_table": accession_table,
        "inventory_table": inventory_table,
        "df_model": df_model,
        "species_models": species_models,
        "origin_models": {},
        "global_model": global_model,
        "genera": ["Astragalus"],
        "as_of_year": 2026,
    }


def test_build_report_includes_every_accession_row(report_inputs):
    html = build_report(**report_inputs)
    assert "PI 1" in html
    assert "Astragalus cicer" in html


def test_build_report_includes_winning_species_chart(report_inputs):
    html = build_report(**report_inputs)
    # One global chart + one for the species that actually won ModelUsed=="Species".
    assert html.count('"title":') == 2


def test_build_report_omits_species_that_never_won_a_row():
    inputs = {
        "accession_table": pd.DataFrame([_table_row(ModelUsed="Global")])[_COLUMNS],
        "inventory_table": pd.DataFrame([_table_row(ModelUsed="Global")])[_COLUMNS],
        "df_model": pd.DataFrame(
            {"AgeAtTest": [10, 20, 30], "Viability": [80.0, 60.0, 40.0],
             "SpeciesGroup": ["Astragalus cicer"] * 3}
        ),
        # Species has a fitted model, but no row's ModelUsed ever resolved to "Species"
        # (all overridden to Global) -- must not get its own chart.
        "species_models": {"Astragalus cicer": SlopeModel(intercept=90, slope=-1.5, n=3, r2=0.2)},
        "origin_models": {},
        "global_model": SlopeModel(intercept=85, slope=-1.0, n=3, r2=0.8),
        "genera": ["Astragalus"],
        "as_of_year": 2026,
    }
    html = build_report(**inputs)
    assert html.count('"title":') == 1  # only the genus-wide chart


def test_build_report_has_no_external_resource_references(report_inputs):
    html = build_report(**report_inputs)
    assert 'href="http' not in html
    assert 'src="http' not in html


def test_build_report_has_light_and_dark_tokens(report_inputs):
    html = build_report(**report_inputs)
    assert "prefers-color-scheme: dark" in html
    assert 'data-theme="dark"' in html


def test_build_report_has_view_toggle_tabs(report_inputs):
    html = build_report(**report_inputs)
    assert 'data-view="accession"' in html
    assert 'data-view="inventory"' in html
    # Inventory card starts hidden -- the toggle, not two stacked sections, controls visibility.
    assert 'id="inventoryCard" hidden' in html


def test_build_report_has_three_way_viability_radio_group_on_both_views(report_inputs):
    html = build_report(**report_inputs)
    for view in ("viabilityFilterAccession", "viabilityFilterInventory"):
        assert f'name="{view}" value="all"' in html
        assert f'name="{view}" value="exclude0"' in html
        assert f'name="{view}" value="only0"' in html
    assert "deadOnlyCheckbox" not in html
    # Two independent radio groups, not one shared name -- otherwise
    # selecting a filter on one view would silently affect the other.
    assert 'name="viabilityFilter"' not in html


def test_build_report_has_sort_note_on_both_views(report_inputs):
    html = build_report(**report_inputs)
    assert html.count('class="sort-note"') == 2
    assert "highest need" in html.lower()
    assert "lowest predicted viability packets first" in html.lower()


def test_build_report_table_pane_is_independently_scrollable(report_inputs):
    html = build_report(**report_inputs)
    assert "max-height: 62vh" in html
    assert "overflow: auto" in html


def test_build_report_respects_chart_cap(report_inputs):
    species_models = {
        f"Astragalus species{i}": SlopeModel(intercept=90, slope=-1.0, n=5, r2=0.9)
        for i in range(10)
    }
    accession_rows = [
        _table_row(Accession=f"PI {i}", Species=f"Astragalus species{i}", ModelUsed="Species")
        for i in range(10)
    ]
    inputs = dict(report_inputs)
    inputs["accession_table"] = pd.DataFrame(accession_rows)[_COLUMNS]
    inputs["species_models"] = species_models
    inputs["df_model"] = pd.DataFrame(
        {
            "AgeAtTest": [10, 20, 30] * 10,
            "Viability": [80.0, 60.0, 40.0] * 10,
            "SpeciesGroup": [f"Astragalus species{i}" for i in range(10) for _ in range(3)],
        }
    )
    inputs["top_n_charts"] = 3
    html = build_report(**inputs)
    assert html.count('"title":') == 4  # global + 3 capped species charts


def test_build_report_includes_status_column_on_both_views(report_inputs):
    inputs = dict(report_inputs)
    inputs["accession_table"] = pd.DataFrame([_table_row(Status="Backup germplasm")])[_COLUMNS]
    inputs["inventory_table"] = pd.DataFrame([_table_row(Status="Exhausted supply")])[_COLUMNS]
    html = build_report(**inputs)
    assert '"Status"' in html
    assert "Backup germplasm" in html
    assert "Exhausted supply" in html
