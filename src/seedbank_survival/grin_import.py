"""Adapts a raw GRIN-Global Curator Tool export into the normalized shape
data_prep.py already consumes (Accession, SeedAge, AgeAtTest, Viability,
Status, Species, Origin, ...).

Each row in a raw export is one Inventory (a seed packet/lot) under an
Accession (a population at time of collection). The Inventory Suffix
("IVS") encodes the lot's harvest/production year, plus a letter marking
it "o" (original) or "i" (increase) -- e.g. "37o" = 1937 original,
"2024i" = 2024 increase.

KNOWN LIMITATION: this export has no wild-collection-date field for
original lots -- only Inventory Suffix, which (for "o" lots) records when
the lot was added to GRIN, not necessarily the true collection date.
SeedAge for original lots is therefore a lower bound, not exact, and may
understate true seed age for accessions collected long before being
formally accessioned. Fixing this would require a different/additional
GRIN export (accession-level passport data) -- out of scope here.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from . import data_prep
from .grin_accessions import build_received_year_lookup

_SUFFIX_WITH_LETTER_RE = re.compile(r"^(\d{2}|\d{4})[A-Za-z]*?([oiOI])\d*$")
_SUFFIX_BARE_YEAR_RE = re.compile(r"^(\d{2}|\d{4})$")

# Two-digit years use a rolling century window, not a flat 19XX assumption --
# verified against real data: site-coded suffixes ("00ncai01".."27ncao01")
# have Created Date in 2000-2027, while plain-style suffixes ("30o".."99o")
# are confirmed 1930-1999 via cross-check against the old reformatted
# dataset's already-computed Year column. The cutoff sits in the unobserved
# 28-29 gap between those two confirmed ranges.
_TWO_DIGIT_YEAR_CUTOFF = 30


def _resolve_year(year_digits: str) -> int:
    if len(year_digits) == 4:
        return int(year_digits)
    n = int(year_digits)
    return 2000 + n if n < _TWO_DIGIT_YEAR_CUTOFF else 1900 + n


def parse_inventory_suffix(suffix: object) -> tuple[int | None, str | None]:
    """Extract (lot_year, type_letter) from a raw Inventory Suffix.

    "37o" -> (1937, "o"); "2024i" -> (2024, "i"); "98ncai01" -> (1998, "i")
    (site code and trailing sequence number are ignored); "23ncao01" ->
    (2023, "o") (two-digit years roll over at _TWO_DIGIT_YEAR_CUTOFF);
    "1995" (no type letter at all) -> (1995, None); "BG"/"1"/"uni" ->
    (None, None) (no leading year digits at all).
    """
    s = str(suffix)

    m = _SUFFIX_WITH_LETTER_RE.match(s)
    if m:
        year_digits, letter = m.group(1), m.group(2).lower()
        return _resolve_year(year_digits), letter

    m = _SUFFIX_BARE_YEAR_RE.match(s)
    if m:
        return _resolve_year(m.group(1)), None

    return None, None


def infer_original_vs_increase(rows: list[tuple[str, int | None, str | None]]) -> list[str | None]:
    """Fill in the type letter for rows whose suffix had a year but no letter.

    rows: a list of (accession, lot_year, type_letter) tuples (type_letter is
    None where parse_inventory_suffix couldn't determine it). Within each
    accession, the earliest-year row is inferred "o" (original) and every
    other no-letter row is inferred "i" (increase) -- validated against real
    data: 725 of 733 accessions (99%) with an explicit "o" row have that
    row's year at or before every sibling row's year.

    Returns inferred letters in the same order as `rows`, one per row:
    the original (possibly still-None) letter is returned unchanged for
    rows that already had one.
    """
    min_year_by_accession: dict[str, int] = {}
    for accession, year, _letter in rows:
        if year is None:
            continue
        current = min_year_by_accession.get(accession)
        if current is None or year < current:
            min_year_by_accession[accession] = year

    inferred: list[str | None] = []
    for accession, year, letter in rows:
        if letter is not None:
            inferred.append(letter)
        elif year is not None and year == min_year_by_accession.get(accession):
            inferred.append("o")
        elif year is not None:
            inferred.append("i")
        else:
            inferred.append(None)
    return inferred


def list_genera(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Distinct first-word-of-Taxon values in a raw export, with row counts.

    Meant for presenting a selection dropdown/checklist -- these exports
    routinely mix in other genera, companion/cover species, and
    administrative placeholders (e.g. "Undetermined nlgrp-backup"). There's
    no reliable way to tell a genuine genus name from a bad one
    automatically, so the person loading the file must pick which ones are
    real, not have that inferred for them.
    """
    genus = df_raw["Taxon"].astype(str).str.split().str[0]
    return genus.value_counts().rename_axis("Genus").reset_index(name="Count")


def filter_to_genus(df_raw: pd.DataFrame, genera: str | list[str]) -> pd.DataFrame:
    """Keep only rows whose Taxon starts with one of the given genus/genera.

    Must run before any other processing on a raw export -- these exports
    routinely mix in other genera, companion/cover species, and
    administrative placeholders. `genera` is deliberately required (no
    default) -- see list_genera for presenting the real choices in a file
    to whoever is loading it, rather than guessing.
    """
    if isinstance(genera, str):
        genera = [genera]
    prefixes = tuple(genera)
    return df_raw[df_raw["Taxon"].astype(str).str.startswith(prefixes)].copy()


def drop_placeholder_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop administrative container rows (Inventory Type == "**").

    These carry an Accession/Taxon but no Suffix, test data, or quantity --
    verified they have nothing salvageable for either model-fitting or
    ranking, so dropping them first is safe.
    """
    return df[df["Inventory Type"] != "**"].copy()


_TYPE_LETTER_TO_LABEL = {"o": "ORIGINAL", "i": "INCREASE"}


@dataclass(frozen=True)
class AdaptedGrinExport:
    """Output of adapt_raw_export: two frames feeding the existing pipeline differently.

    df_primary: one row per surviving lot, in the same normalized shape
    data_prep.py already consumes. Meant to be run through the existing,
    unmodified clean_ages -> build_model_dataset / build_ranking_dataset,
    exactly like the old CSV/XLSX path.

    df_borrowed: "borrowed" test points -- rows with real test data but no
    own resolvable SeedAge, salvaged via a same-Accession sibling lot's year
    (see resolve_borrowed_row). Has no SeedAge column at all, so it can only
    ever reach model-fitting, never ranking.
    """

    df_primary: pd.DataFrame
    df_borrowed: pd.DataFrame


def _species_group(species: object) -> str | None:
    """Species value, except an unresolved-to-species Taxon ("<Genus> spp.")
    becomes NaN.

    Excludes it from ever forming its own species-tier deterioration model
    (mixing unrelated species into one curve would be meaningless) while
    still letting it participate fully in Origin/Global fits. Genus-agnostic
    by design -- verified this "<Genus> spp." pattern holds identically
    across every genus present in real exports (Astragalus, Onobrychis,
    Oxytropis, Trifolium, ...), not just the one being actively filtered to.
    """
    if isinstance(species, str) and species.endswith(" spp."):
        return None
    return species


def build_sibling_year_index(
    accessions: list, lot_years: list
) -> dict[object, list[int]]:
    """Accession -> sorted list of resolved lot years among its rows.

    Used to find a plausible age for a lot whose own Inventory Suffix
    didn't resolve to a year, by borrowing a same-Accession sibling's year.
    """
    index: dict[object, list[int]] = {}
    for accession, year in zip(accessions, lot_years):
        if year is None or (isinstance(year, float) and pd.isna(year)):
            continue
        index.setdefault(accession, []).append(int(year))
    for years in index.values():
        years.sort()
    return index


def resolve_borrowed_row(
    accession: object, tested_year: int, sibling_index: dict[object, list[int]]
) -> int | None:
    """Nearest sibling lot year that isn't after tested_year, or None if none qualify.

    A lot can't have been the physical seed tested in a year before it existed --
    only sibling years at or before the test's year are eligible; among those,
    the most recent one is the best guess for "the seed most likely tested."
    """
    years = sibling_index.get(accession)
    if not years:
        return None
    candidates = [year for year in years if year <= tested_year]
    if not candidates:
        return None
    return max(candidates)


def adapt_raw_export(
    df_raw: pd.DataFrame,
    as_of_year: int,
    genera: str | list[str],
    df_accessions: pd.DataFrame | None = None,
) -> AdaptedGrinExport:
    """Adapt a raw GRIN-Global Curator Tool export into the shape data_prep.py consumes.

    `genera` is required, not defaulted -- see list_genera/filter_to_genus.
    Accepts one genus or several (e.g. current and synonymous former names),
    since a curator's data may legitimately span more than one.

    `df_accessions` is an optional raw accession-level export (one row per
    Accession -- see grin_accessions.py). When given, its Received Date backs
    a SECOND, independent fallback chain used only for SeedAge (current age,
    every row): own suffix -> Received Date. This is deliberately separate
    from AgeAtTest's chain (own suffix -> sibling-year, in
    _build_borrowed_rows): a Received Date is accession-level and has no
    test event to anchor a plausibility check against, so it isn't a
    substitute for sibling-year resolution -- only for "how old is this lot
    today," which has no such anchor to begin with. A row resolved this way
    lands in df_primary (ranking-eligible), not df_borrowed.
    """
    df = filter_to_genus(df_raw, genera)
    df = drop_placeholder_rows(df)

    parsed = df["Inventory Suffix"].apply(parse_inventory_suffix)
    lot_years = pd.Series([t[0] for t in parsed], index=df.index, dtype="float64")
    own_letters = [t[1] for t in parsed]

    rows = list(zip(df["Accession"], lot_years, own_letters))
    letters = infer_original_vs_increase(rows)

    viability_year = pd.to_datetime(df["Tested Date"], errors="coerce").dt.year.astype("float64")

    seed_age_years = lot_years
    if df_accessions is not None:
        received_year_lookup = build_received_year_lookup(df_accessions)
        received_fallback = df["Accession"].map(received_year_lookup).astype("float64")
        seed_age_years = lot_years.where(lot_years.notna(), received_fallback)

    df_primary = pd.DataFrame(
        {
            "Accession": df["Accession"].to_numpy(),
            "SeedAge": (as_of_year - seed_age_years).to_numpy(),
            "AgeAtTest": (viability_year - lot_years).to_numpy(),
            "Viability": df["Percent Viable"].to_numpy(),
            "ViabilityYear": viability_year.to_numpy(),
            "Status": df["Inventory Status"].to_numpy(),
            "Species": df["Taxon"].to_numpy(),
            "Origin": df["Origin"].to_numpy(),
            "Type": [_TYPE_LETTER_TO_LABEL.get(letter) for letter in letters],
            "EstTotalSeed": df["Quantity On Hand"].to_numpy(),
        }
    )
    df_primary["SpeciesGroup"] = df_primary["Species"].apply(_species_group)
    df_primary["EstLiveSeed"] = df_primary["EstTotalSeed"] * df_primary["Viability"] / 100

    df_borrowed = _build_borrowed_rows(df, lot_years)

    return AdaptedGrinExport(df_primary=df_primary, df_borrowed=df_borrowed)


def _build_borrowed_rows(df: pd.DataFrame, lot_years: pd.Series) -> pd.DataFrame:
    """Salvage (AgeAtTest, Viability) points for rows with real test data but no
    own resolvable year, via a same-Accession sibling's year (resolve_borrowed_row).

    "Unrepeated" guard: a test event already represented -- either by a row
    with its own resolvable year, or by an earlier borrowed row -- for the same
    (Accession, Viability, Tested Date) is never added twice. Duplicate
    (Accession, Viability, Tested Date) rows typically mean the result was
    inherited/copied onto a backup lot's record rather than independently
    measured.
    """
    sibling_index = build_sibling_year_index(df["Accession"], lot_years)

    has_test_data = df["Percent Viable"].notna() & df["Tested Date"].notna()
    needs_borrowing = lot_years.isna() & has_test_data

    represented_events = {
        (accession, viable, tested)
        for accession, year, viable, tested in zip(
            df["Accession"], lot_years, df["Percent Viable"], df["Tested Date"]
        )
        if not pd.isna(year)
    }

    records = []
    for idx in df.index[needs_borrowing]:
        accession = df.at[idx, "Accession"]
        viable = df.at[idx, "Percent Viable"]
        tested = df.at[idx, "Tested Date"]
        event = (accession, viable, tested)
        if event in represented_events:
            continue

        tested_year = pd.Timestamp(tested).year
        borrowed_year = resolve_borrowed_row(accession, tested_year, sibling_index)
        if borrowed_year is None:
            continue

        species = df.at[idx, "Taxon"]
        records.append(
            {
                "Accession": accession,
                "AgeAtTest": tested_year - borrowed_year,
                "Viability": viable,
                "Species": species,
                "SpeciesGroup": _species_group(species),
                "Origin": df.at[idx, "Origin"],
            }
        )
        represented_events.add(event)

    df_borrowed = pd.DataFrame(
        records, columns=["Accession", "AgeAtTest", "Viability", "Species", "SpeciesGroup", "Origin"]
    )
    # An empty `records` list leaves AgeAtTest/Viability as object dtype (pandas
    # can't infer numeric from nothing) -- harmless alone, but pd.concat in
    # assemble_model_dataset upcasts the WHOLE combined column to object if
    # either side is object-dtyped, silently breaking downstream OLS fits.
    # Only bites when df_borrowed is empty, which real data never hits (there's
    # always something to salvage) but small synthetic fixtures/tests do.
    df_borrowed["AgeAtTest"] = df_borrowed["AgeAtTest"].astype("float64")
    df_borrowed["Viability"] = df_borrowed["Viability"].astype("float64")
    return df_borrowed


def assemble_model_dataset(df_primary_clean: pd.DataFrame, df_borrowed: pd.DataFrame) -> pd.DataFrame:
    """Union the two model-fitting sources of an AdaptedGrinExport.

    df_primary_clean is data_prep.clean_ages(adapted.df_primary); df_borrowed is
    adapted.df_borrowed as-is (it has no SeedAge column and never needs
    clean_ages -- build_model_dataset only reads AgeAtTest/Viability).
    """
    return pd.concat(
        [
            data_prep.build_model_dataset(df_primary_clean),
            data_prep.build_model_dataset(df_borrowed),
        ],
        ignore_index=True,
    )
