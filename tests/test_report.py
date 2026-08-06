from __future__ import annotations

import pandas as pd
import pytest

from seedbank_survival.deterioration import BreakpointCurve, QuadraticCurve, WeibullCurve
from seedbank_survival.report import ModelReportData, build_report


def _curve(intercept, slope, n, r2, pvalue):
    return QuadraticCurve(intercept=intercept, linear_coef=slope, quad_coef=0.0, n=n, r2=r2, overall_pvalue=pvalue, max_fit_age=100.0)


_COLUMNS = [
    "Accession", "Suffix", "Status", "Species", "Origin", "SeedAge", "PrimaryReason",
    "EstimatedViability_2026", "YearsRemainingTo0%", "ModelUsed", "ModelConfidence", "Location",
]
# Carried alongside _COLUMNS but not one of the displayed table columns --
# only used by the on-site filter checkbox (see report.py's _FILTER_ONLY_COLUMNS).
_EXTRA_ROW_COLUMNS = ["MaintenanceSite"]


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
        "Location": "minus20, C09",
        "MaintenanceSite": "W6",
    }
    row.update(overrides)
    return pd.Series(row)


def _model_data(
    accession_rows=None, inventory_rows=None, df_model=None,
    species_models=None, origin_models=None, global_model=None,
):
    accession_rows = accession_rows if accession_rows is not None else [_table_row()]
    inventory_rows = inventory_rows if inventory_rows is not None else [_table_row(), _table_row(Accession="PI 1", ModelUsed="Species")]
    df_model = df_model if df_model is not None else pd.DataFrame(
        {"AgeAtTest": [10, 20, 30], "Viability": [80.0, 60.0, 40.0], "SpeciesGroup": ["Astragalus cicer"] * 3}
    )
    species_models = species_models if species_models is not None else {"Astragalus cicer": _curve(90, -1.5, n=3, r2=0.95, pvalue=0.01)}
    global_model = global_model if global_model is not None else _curve(85, -1.0, n=3, r2=0.8, pvalue=0.01)
    return ModelReportData(
        accession_table=pd.DataFrame(accession_rows)[_COLUMNS + _EXTRA_ROW_COLUMNS],
        inventory_table=pd.DataFrame(inventory_rows)[_COLUMNS + _EXTRA_ROW_COLUMNS],
        df_model=df_model,
        species_models=species_models,
        origin_models=origin_models if origin_models is not None else {},
        global_model=global_model,
    )


@pytest.fixture
def report_inputs():
    return {
        "models": {"quadratic": _model_data()},
        "genera": ["Astragalus"],
        "as_of_year": 2026,
        "default_model": "quadratic",
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
    # Species has a fitted model, but no row's ModelUsed ever resolved to "Species"
    # (all overridden to Global) -- must not get its own chart.
    inputs = {
        "models": {"quadratic": _model_data(
            accession_rows=[_table_row(ModelUsed="Global")],
            inventory_rows=[_table_row(ModelUsed="Global")],
            species_models={"Astragalus cicer": _curve(90, -1.5, n=3, r2=0.2, pvalue=0.01)},
        )},
        "genera": ["Astragalus"],
        "as_of_year": 2026,
        "default_model": "quadratic",
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
        f"Astragalus species{i}": _curve(90, -1.0, n=5, r2=0.9, pvalue=0.01)
        for i in range(10)
    }
    accession_rows = [
        _table_row(Accession=f"PI {i}", Species=f"Astragalus species{i}", ModelUsed="Species")
        for i in range(10)
    ]
    df_model = pd.DataFrame(
        {
            "AgeAtTest": [10, 20, 30] * 10,
            "Viability": [80.0, 60.0, 40.0] * 10,
            "SpeciesGroup": [f"Astragalus species{i}" for i in range(10) for _ in range(3)],
        }
    )
    inputs = dict(report_inputs)
    inputs["models"] = {"quadratic": _model_data(
        accession_rows=accession_rows, species_models=species_models, df_model=df_model,
    )}
    inputs["top_n_charts"] = 3
    html = build_report(**inputs)
    assert html.count('"title":') == 4  # global + 3 capped species charts


def test_build_report_includes_location_column(report_inputs):
    html = build_report(**report_inputs)
    assert '"Location"' in html
    assert "minus20, C09" in html


def test_build_report_has_on_site_checkbox_on_both_views(report_inputs):
    html = build_report(**report_inputs)
    assert 'id="onSiteAccession"' in html
    assert 'id="onSiteInventory"' in html
    assert "W6" in html and "Pullman" in html


def test_build_report_maintenance_site_used_for_filtering_not_displayed(report_inputs):
    html = build_report(**report_inputs)
    # Row payloads carry MaintenanceSite so the on-site checkbox can filter on it...
    assert '"MaintenanceSite": "W6"' in html
    # ...but it must not be one of the rendered/sortable table columns.
    columns_segment = html.split('"models"')[0]
    assert "MaintenanceSite" not in columns_segment


def test_build_report_has_column_visibility_toggle_on_both_views(report_inputs):
    html = build_report(**report_inputs)
    assert 'data-menu="colMenuAccession"' in html
    assert 'data-menu="colMenuInventory"' in html
    assert 'id="colMenuAccession"' in html
    assert 'id="colMenuInventory"' in html
    assert "buildColumnMenu" in html


def test_build_report_column_menu_hidden_rule_beats_display_flex(report_inputs):
    # Regression test: `.col-menu { display: flex; ... }` alone silently beats
    # the browser's built-in `[hidden] { display: none }` rule at equal
    # specificity (this stylesheet loads after the UA one), so the menu
    # rendered open by default on page load -- caught via live browser
    # verification, not visible from a static HTML diff alone. An explicit
    # `.col-menu[hidden] { display: none; }` rule is required to win back.
    html = build_report(**report_inputs)
    assert ".col-menu[hidden]" in html


def test_build_report_has_column_resize_handles(report_inputs):
    html = build_report(**report_inputs)
    assert "col-resize-handle" in html
    assert "tableLayout" in html


def test_build_report_includes_status_column_on_both_views(report_inputs):
    inputs = dict(report_inputs)
    inputs["models"] = {"quadratic": _model_data(
        accession_rows=[_table_row(Status="Backup germplasm")],
        inventory_rows=[_table_row(Status="Exhausted supply")],
    )}
    html = build_report(**inputs)
    assert '"Status"' in html
    assert "Backup germplasm" in html
    assert "Exhausted supply" in html


def test_build_report_renders_weibull_curve_payload():
    models = {"weibull": _model_data(
        global_model=WeibullCurve(v0=90, lam=20, k=1.5, n=50, r2=0.4, overall_pvalue=0.01, max_fit_age=50.0),
        species_models={},
    )}
    html = build_report(models, genera=["Astragalus"], as_of_year=2026, default_model="weibull")
    assert '"kind": "weibull"' in html
    assert '"v0": 90' in html
    assert '"lam": 20' in html


def test_build_report_renders_breakpoint_curve_payload():
    models = {"breakpoint": _model_data(
        global_model=BreakpointCurve(t0=25, plateau=85, slope=-3, n=50, r2=0.4, overall_pvalue=0.01, max_fit_age=50.0),
        species_models={},
    )}
    html = build_report(models, genera=["Astragalus"], as_of_year=2026, default_model="breakpoint")
    assert '"kind": "breakpoint"' in html
    assert '"t0": 25' in html
    assert '"plateau": 85' in html


def test_build_report_quadratic_payload_tagged_with_kind(report_inputs):
    html = build_report(**report_inputs)
    assert '"kind": "quadratic"' in html


def test_build_report_embeds_all_provided_model_kinds():
    models = {
        "quadratic": _model_data(),
        "weibull": _model_data(global_model=WeibullCurve(v0=90, lam=20, k=1.5, n=50, r2=0.1, overall_pvalue=0.5, max_fit_age=50.0)),
        "breakpoint": _model_data(global_model=BreakpointCurve(t0=25, plateau=85, slope=-3, n=50, r2=0.1, overall_pvalue=0.5, max_fit_age=50.0)),
    }
    html = build_report(models, genera=["Astragalus"], as_of_year=2026, default_model="quadratic")
    assert '"kind": "quadratic"' in html
    assert '"kind": "weibull"' in html
    assert '"kind": "breakpoint"' in html
    # One "models" JS-data section covering all three -- not three separate reports.
    assert html.count('"defaultModel"') == 1


def test_build_report_has_model_selector_on_both_views(report_inputs):
    html = build_report(**report_inputs)
    assert 'id="modelSelectAccession"' in html
    assert 'id="modelSelectInventory"' in html
    assert "buildModelSelector" in html
    assert "setModel" in html


def test_build_report_falls_back_to_an_available_model_if_default_missing():
    # e.g. --model weibull was requested but weibull failed to converge and
    # was skipped by the CLI -- the report must still pick SOME model to show
    # first rather than erroring or showing a blank page.
    models = {"breakpoint": _model_data()}
    html = build_report(models, genera=["Astragalus"], as_of_year=2026, default_model="weibull")
    assert '"defaultModel": "breakpoint"' in html


def test_build_report_row_counts_are_shared_not_per_model(report_inputs):
    html = build_report(**report_inputs)
    # counts.accession/counts.inventory sit outside "models" -- shared across
    # every model kind, since row SETS don't vary by curve, only predictions do.
    counts_segment = html.split('"counts"')[1].split("}")[0]
    assert '"accession": 1' in counts_segment
    assert '"inventory": 2' in counts_segment
