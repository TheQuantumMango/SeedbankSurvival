from __future__ import annotations

import pandas as pd

from seedbank_survival.grin_import import (
    adapt_raw_export,
    build_sibling_year_index,
    drop_placeholder_rows,
    filter_to_genus,
    infer_original_vs_increase,
    parse_inventory_suffix,
    resolve_borrowed_row,
)


def _raw_export_row(**overrides):
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
    }
    row.update(overrides)
    return row


def test_parse_two_digit_year_with_letter():
    assert parse_inventory_suffix("37o") == (1937, "o")
    assert parse_inventory_suffix("76i") == (1976, "i")


def test_parse_four_digit_year_with_letter():
    assert parse_inventory_suffix("2024i") == (2024, "i")
    assert parse_inventory_suffix("2008o") == (2008, "o")


def test_parse_site_code_and_sequence_suffix():
    assert parse_inventory_suffix("98ncai01") == (1998, "i")
    assert parse_inventory_suffix("23ncao01") == (2023, "o")
    assert parse_inventory_suffix("16ohwo01") == (2016, "o")


def test_parse_bare_year_no_letter():
    assert parse_inventory_suffix("1995") == (1995, None)
    assert parse_inventory_suffix("1992") == (1992, None)


def test_parse_unparseable_suffix():
    assert parse_inventory_suffix("BG") == (None, None)
    assert parse_inventory_suffix("1") == (None, None)
    assert parse_inventory_suffix("uni") == (None, None)
    assert parse_inventory_suffix("I") == (None, None)


def test_parse_bare_two_digit_no_letter_uses_same_windowing():
    # Doesn't occur in real data, but should still be consistent with the
    # windowed rule rather than a special case.
    assert parse_inventory_suffix("51") == (1951, None)
    assert parse_inventory_suffix("12") == (2012, None)


def test_parse_single_digit_suffix_is_unparseable():
    # NSSL backup-lot sequence numbers ("1", "2", "3", "0"), not years.
    assert parse_inventory_suffix("1") == (None, None)
    assert parse_inventory_suffix("0") == (None, None)


def test_infer_original_vs_increase_fills_earliest_as_original():
    rows = [
        ("PI 1", 1995, None),
        ("PI 1", 2001, None),
        ("PI 1", 2010, None),
    ]
    assert infer_original_vs_increase(rows) == ["o", "i", "i"]


def test_infer_original_vs_increase_respects_existing_letters():
    rows = [
        ("PI 2", 1980, "o"),
        ("PI 2", 1995, None),
    ]
    assert infer_original_vs_increase(rows) == ["o", "i"]


def test_infer_original_vs_increase_leaves_unknown_year_alone():
    rows = [("PI 3", None, None)]
    assert infer_original_vs_increase(rows) == [None]


def test_filter_to_genus_keeps_only_matching_taxon():
    df = pd.DataFrame({
        "Taxon": ["Astragalus cicer", "Elymus elymoides", "Astragalus spp.", "Poa secunda"],
        "Accession": ["A1", "A2", "A3", "A4"],
    })
    result = filter_to_genus(df, "Astragalus")
    assert result["Accession"].tolist() == ["A1", "A3"]


def test_drop_placeholder_rows_removes_double_star_type():
    df = pd.DataFrame({
        "Inventory Type": ["SD", "**", "SD", "**"],
        "Accession": ["A1", "A2", "A3", "A4"],
    })
    result = drop_placeholder_rows(df)
    assert result["Accession"].tolist() == ["A1", "A3"]


def test_adapt_raw_export_computes_seed_age_and_age_at_test():
    df_raw = pd.DataFrame([_raw_export_row()])
    result = adapt_raw_export(df_raw, as_of_year=2026)
    row = result.df_primary.iloc[0]
    assert row["SeedAge"] == 2026 - 1937
    assert row["AgeAtTest"] == 2003 - 1937
    assert row["Type"] == "ORIGINAL"
    assert row["Viability"] == 90.0
    assert row["Species"] == "Astragalus cicer"
    assert row["SpeciesGroup"] == "Astragalus cicer"


def test_adapt_raw_export_filters_genus_and_placeholders_first():
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1"),
            _raw_export_row(Accession="PI 2", Taxon="Elymus elymoides"),
            _raw_export_row(Accession="PI 3", **{"Inventory Type": "**"}),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026)
    assert result.df_primary["Accession"].tolist() == ["PI 1"]


def test_adapt_raw_export_excludes_unresolved_species_from_species_group():
    df_raw = pd.DataFrame([_raw_export_row(Taxon="Astragalus spp.")])
    result = adapt_raw_export(df_raw, as_of_year=2026)
    row = result.df_primary.iloc[0]
    assert row["Species"] == "Astragalus spp."
    assert pd.isna(row["SpeciesGroup"])


def test_adapt_raw_export_leaves_seed_age_nan_for_unparseable_suffix():
    df_raw = pd.DataFrame([_raw_export_row(**{"Inventory Suffix": "BG"})])
    result = adapt_raw_export(df_raw, as_of_year=2026)
    row = result.df_primary.iloc[0]
    assert pd.isna(row["SeedAge"])
    assert pd.isna(row["AgeAtTest"])


def test_build_sibling_year_index_groups_by_accession_and_sorts():
    index = build_sibling_year_index(
        accessions=["PI 1", "PI 1", "PI 2", "PI 1"],
        lot_years=[1995, 1980, 2001, None],
    )
    assert index == {"PI 1": [1980, 1995], "PI 2": [2001]}


def test_resolve_borrowed_row_picks_nearest_year_not_after_tested_year():
    index = {"PI 1": [1980, 1995, 2010]}
    # Tested in 2000 -- 2010 didn't exist yet, 1995 is the nearest eligible sibling.
    assert resolve_borrowed_row("PI 1", tested_year=2000, sibling_index=index) == 1995


def test_resolve_borrowed_row_none_when_no_sibling_predates_test():
    index = {"PI 1": [2010]}
    assert resolve_borrowed_row("PI 1", tested_year=2000, sibling_index=index) is None


def test_resolve_borrowed_row_none_for_unknown_accession():
    assert resolve_borrowed_row("PI 999", tested_year=2000, sibling_index={}) is None


def test_adapt_raw_export_borrows_age_from_sibling_for_unparseable_suffix():
    sibling_row = _raw_export_row(
        Accession="PI 1",
        **{
            "Inventory Suffix": "37o",
            "Percent Viable": None,
            "Tested Date": pd.NaT,
        },
    )
    unparseable_row = _raw_export_row(
        Accession="PI 1",
        **{
            "Inventory Suffix": "BG",
            "Percent Viable": 60.0,
            "Tested Date": pd.Timestamp("1995-06-01"),
        },
    )
    df_raw = pd.DataFrame([sibling_row, unparseable_row])

    result = adapt_raw_export(df_raw, as_of_year=2026)
    assert len(result.df_borrowed) == 1
    borrowed = result.df_borrowed.iloc[0]
    assert borrowed["Accession"] == "PI 1"
    assert borrowed["AgeAtTest"] == 1995 - 1937
    assert borrowed["Viability"] == 60.0
    # never gets a SeedAge -- can't reach ranking, only model fitting.
    assert "SeedAge" not in result.df_borrowed.columns


def test_adapt_raw_export_skips_borrowing_with_no_resolvable_sibling():
    df_raw = pd.DataFrame(
        [
            _raw_export_row(
                Accession="PI 2",
                **{
                    "Inventory Suffix": "BG",
                    "Percent Viable": 60.0,
                    "Tested Date": pd.Timestamp("1995-06-01"),
                },
            ),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026)
    assert len(result.df_borrowed) == 0


def test_adapt_raw_export_does_not_double_count_inherited_test_result():
    # Same (Accession, Viability, Tested Date) on both an own-resolvable row and
    # an unparseable one -- the unparseable one's "test" is already represented.
    shared_test = {"Percent Viable": 60.0, "Tested Date": pd.Timestamp("1995-06-01")}
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1", **{"Inventory Suffix": "37o", **shared_test}),
            _raw_export_row(Accession="PI 1", **{"Inventory Suffix": "BG", **shared_test}),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026)
    assert len(result.df_borrowed) == 0
