from __future__ import annotations

import pandas as pd

from seedbank_survival.grin_import import (
    adapt_raw_export,
    assemble_model_dataset,
    build_sibling_year_index,
    drop_placeholder_rows,
    filter_to_genus,
    infer_original_vs_increase,
    list_genera,
    parse_inventory_suffix,
    resolve_borrowed_row,
)
from seedbank_survival.data_prep import clean_ages


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
        "Inventory Maintenance Site": "W6",
        "Location Section 1": "minus20",
        "Location Section 2": None,
        "Location Section 3": None,
        "Location Section 4": None,
    }
    row.update(overrides)
    return row


def _accession_row(**overrides):
    row = {
        "Accession": "PI 100000",
        "Taxon": "Astragalus cicer",
        "Received Date Format": "mm/dd/yyyy",
        "Received Date": pd.Timestamp("1990-02-05"),
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


def test_infer_original_vs_increase_leaves_nan_year_alone():
    # Regression test: adapt_raw_export actually builds `rows` by zipping a
    # pandas float64 Series, which silently turns a missing year into
    # float('nan'), not Python None. `nan is None` is False, so a naive
    # `is None` check here would let a NaN-year row fall through to being
    # mislabeled "i" (increase) instead of staying unknown -- this is exactly
    # what happened with real NSSL-backup-lot rows (Suffix "1"/"2"), which
    # have no parseable year at all.
    rows = [("PI 4", float("nan"), None)]
    assert infer_original_vs_increase(rows) == [None]


def test_infer_original_vs_increase_nan_year_does_not_corrupt_accession_minimum():
    # A NaN-year row must not participate in "which row is this accession's
    # earliest" at all -- it should neither become the minimum itself nor
    # prevent a real year from being recognized as the minimum.
    rows = [
        ("PI 5", float("nan"), None),
        ("PI 5", 1995, None),
        ("PI 5", 2001, None),
    ]
    assert infer_original_vs_increase(rows) == [None, "o", "i"]


def test_filter_to_genus_keeps_only_matching_taxon():
    df = pd.DataFrame({
        "Taxon": ["Astragalus cicer", "Elymus elymoides", "Astragalus spp.", "Poa secunda"],
        "Accession": ["A1", "A2", "A3", "A4"],
    })
    result = filter_to_genus(df, "Astragalus")
    assert result["Accession"].tolist() == ["A1", "A3"]


def test_filter_to_genus_accepts_multiple_genera():
    # e.g. a genus plus a synonymous former name present in the same export.
    df = pd.DataFrame({
        "Taxon": ["Astragalus cicer", "Homalobus cicer", "Elymus elymoides", "Poa secunda"],
        "Accession": ["A1", "A2", "A3", "A4"],
    })
    result = filter_to_genus(df, ["Astragalus", "Homalobus"])
    assert result["Accession"].tolist() == ["A1", "A2"]


def test_list_genera_counts_first_word_of_taxon():
    df = pd.DataFrame({
        "Taxon": [
            "Astragalus cicer",
            "Astragalus spp.",
            "Elymus elymoides",
            "Undetermined nlgrp-backup",
        ],
    })
    result = list_genera(df).set_index("Genus")["Count"]
    assert result["Astragalus"] == 2
    assert result["Elymus"] == 1
    assert result["Undetermined"] == 1


def test_drop_placeholder_rows_removes_double_star_type():
    df = pd.DataFrame({
        "Inventory Type": ["SD", "**", "SD", "**"],
        "Accession": ["A1", "A2", "A3", "A4"],
    })
    result = drop_placeholder_rows(df)
    assert result["Accession"].tolist() == ["A1", "A3"]


def test_adapt_raw_export_computes_seed_age_and_age_at_test():
    df_raw = pd.DataFrame([_raw_export_row()])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
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
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert result.df_primary["Accession"].tolist() == ["PI 1"]


def test_adapt_raw_export_accepts_multiple_genera():
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1", Taxon="Astragalus cicer"),
            _raw_export_row(Accession="PI 2", Taxon="Onobrychis viciifolia"),
            _raw_export_row(Accession="PI 3", Taxon="Elymus elymoides"),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026, genera=["Astragalus", "Onobrychis"])
    assert result.df_primary["Accession"].tolist() == ["PI 1", "PI 2"]


def test_adapt_raw_export_species_group_exclusion_is_genus_agnostic():
    df_raw = pd.DataFrame([_raw_export_row(Taxon="Onobrychis spp.")])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Onobrychis")
    row = result.df_primary.iloc[0]
    assert row["Species"] == "Onobrychis spp."
    assert pd.isna(row["SpeciesGroup"])


def test_adapt_raw_export_excludes_unresolved_species_from_species_group():
    df_raw = pd.DataFrame([_raw_export_row(Taxon="Astragalus spp.")])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    row = result.df_primary.iloc[0]
    assert row["Species"] == "Astragalus spp."
    assert pd.isna(row["SpeciesGroup"])


def test_adapt_raw_export_leaves_seed_age_nan_for_unparseable_suffix():
    df_raw = pd.DataFrame([_raw_export_row(**{"Inventory Suffix": "BG"})])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    row = result.df_primary.iloc[0]
    assert pd.isna(row["SeedAge"])
    assert pd.isna(row["AgeAtTest"])


def test_adapt_raw_export_carries_maintenance_site():
    df_raw = pd.DataFrame([_raw_export_row(**{"Inventory Maintenance Site": "NLGRP"})])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert result.df_primary.iloc[0]["MaintenanceSite"] == "NLGRP"


def test_adapt_raw_export_joins_location_sections_skipping_blanks():
    df_raw = pd.DataFrame([_raw_export_row(**{
        "Location Section 1": "minus20",
        "Location Section 2": None,
        "Location Section 3": "C09",
        "Location Section 4": "",
    })])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert result.df_primary.iloc[0]["Location"] == "minus20, C09"


def test_adapt_raw_export_location_is_empty_string_when_all_sections_blank():
    df_raw = pd.DataFrame([_raw_export_row(**{
        "Location Section 1": None, "Location Section 2": None,
        "Location Section 3": None, "Location Section 4": None,
    })])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert result.df_primary.iloc[0]["Location"] == ""


def test_adapt_raw_export_location_join_handles_non_string_section_values():
    # Real data has integer-valued sections (e.g. Location Section 2 == 4).
    df_raw = pd.DataFrame([_raw_export_row(**{
        "Location Section 1": "R040", "Location Section 2": 4,
        "Location Section 3": "C09", "Location Section 4": None,
    })])
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert result.df_primary.iloc[0]["Location"] == "R040, 4, C09"


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

    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
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
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
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
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert len(result.df_borrowed) == 0


def test_adapt_raw_export_does_not_double_count_inherited_test_result_same_day_different_time():
    # Regression test: real data shows a dated lot's test result routinely
    # re-recorded on a same-Accession "Backup germplasm" row a few HOURS
    # later the SAME DAY (e.g. 00:03:23 vs 22:33:55) -- an administrative
    # echo of one physical test, not an independent measurement. Matching on
    # exact Tested Date (as before) missed this; matching on calendar day
    # catches it. Verified this pattern affects ~21% of the real genus-wide
    # model-fitting dataset (162 accessions) before this fix.
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1", **{
                "Inventory Suffix": "37o", "Percent Viable": 60.0,
                "Tested Date": pd.Timestamp("1995-06-01 00:03:23"),
            }),
            _raw_export_row(Accession="PI 1", **{
                "Inventory Suffix": "BG", "Percent Viable": 60.0,
                "Tested Date": pd.Timestamp("1995-06-01 22:33:55"),
            }),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert len(result.df_borrowed) == 0


def test_adapt_raw_export_borrows_genuinely_distinct_same_day_test():
    # Two DIFFERENT test results (different Viability) recorded the same day
    # are two real, independent measurements -- must not be conflated with
    # the inherited-duplicate case just because the calendar day matches.
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1", **{
                "Inventory Suffix": "37o", "Percent Viable": 60.0,
                "Tested Date": pd.Timestamp("1995-06-01 00:03:23"),
            }),
            _raw_export_row(Accession="PI 1", **{
                "Inventory Suffix": "BG", "Percent Viable": 45.0,
                "Tested Date": pd.Timestamp("1995-06-01 22:33:55"),
            }),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert len(result.df_borrowed) == 1
    assert result.df_borrowed.iloc[0]["Viability"] == 45.0


def test_adapt_raw_export_received_date_resolves_seed_age_into_primary():
    df_raw = pd.DataFrame([_raw_export_row(Accession="PI 1", **{"Inventory Suffix": "BG"})])
    df_accessions = pd.DataFrame([_accession_row(Accession="PI 1")])

    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus", df_accessions=df_accessions)

    row = result.df_primary.iloc[0]
    assert row["SeedAge"] == 2026 - 1990
    # Lands in df_primary, i.e. ranking-eligible -- not merely salvaged for model fitting.
    assert len(result.df_borrowed) == 0


def test_adapt_raw_export_without_accessions_arg_is_unchanged():
    df_raw = pd.DataFrame([_raw_export_row(Accession="PI 1", **{"Inventory Suffix": "BG"})])

    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")

    assert pd.isna(result.df_primary.iloc[0]["SeedAge"])


def test_adapt_raw_export_own_suffix_wins_over_received_date():
    df_raw = pd.DataFrame([_raw_export_row(Accession="PI 1", **{"Inventory Suffix": "37o"})])
    df_accessions = pd.DataFrame([_accession_row(Accession="PI 1", **{"Received Date": pd.Timestamp("1950-01-01")})])

    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus", df_accessions=df_accessions)

    row = result.df_primary.iloc[0]
    assert row["SeedAge"] == 2026 - 1937  # from "37o", not the 1950 Received Date


def test_received_date_fallback_does_not_affect_age_at_test_or_borrowed_chain():
    # AgeAtTest / df_borrowed use their own chain (own suffix -> sibling); the
    # Received Date fallback is SeedAge-only and must not change either.
    unparseable_with_test = _raw_export_row(
        Accession="PI 1",
        **{
            "Inventory Suffix": "BG",
            "Percent Viable": 60.0,
            "Tested Date": pd.Timestamp("1995-06-01"),
        },
    )
    df_raw = pd.DataFrame([unparseable_with_test])
    df_accessions = pd.DataFrame([_accession_row(Accession="PI 1")])

    without = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    with_acc = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus", df_accessions=df_accessions)

    assert len(without.df_borrowed) == len(with_acc.df_borrowed) == 0
    assert pd.isna(without.df_primary.iloc[0]["AgeAtTest"])
    assert pd.isna(with_acc.df_primary.iloc[0]["AgeAtTest"])
    # SeedAge does differ -- that's the whole point of the fallback.
    assert pd.isna(without.df_primary.iloc[0]["SeedAge"])
    assert with_acc.df_primary.iloc[0]["SeedAge"] == 2026 - 1990


def test_empty_df_borrowed_does_not_break_assemble_model_dataset_dtypes():
    # Regression test: an empty df_borrowed (no unparseable-suffix rows at
    # all -- realistic for a small dataset) used to leave AgeAtTest/Viability
    # as object dtype (pandas can't infer numeric from an empty list), which
    # silently upcast the WHOLE combined column to object on concat and broke
    # statsmodels' OLS fit downstream. Never showed up against real data
    # (df_borrowed is never empty there), but a small synthetic set hits it.
    df_raw = pd.DataFrame(
        [
            _raw_export_row(Accession="PI 1", **{"Inventory Suffix": "90o", "Percent Viable": 90.0}),
            _raw_export_row(Accession="PI 2", **{"Inventory Suffix": "95o", "Percent Viable": 70.0}),
            _raw_export_row(Accession="PI 3", **{"Inventory Suffix": "99o", "Percent Viable": 50.0}),
        ]
    )
    result = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    assert len(result.df_borrowed) == 0
    assert result.df_borrowed["AgeAtTest"].dtype == "float64"
    assert result.df_borrowed["Viability"].dtype == "float64"

    df_model = assemble_model_dataset(clean_ages(result.df_primary), result.df_borrowed)
    assert df_model["AgeAtTest"].dtype == "float64"
    assert df_model["Viability"].dtype == "float64"
    # This is the actual failure mode: fitting used to raise on object dtype.
    from seedbank_survival.deterioration import fit_global_model

    model = fit_global_model(df_model)
    assert model.n == 3
