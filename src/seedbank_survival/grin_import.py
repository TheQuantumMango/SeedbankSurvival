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


def filter_to_genus(df_raw: pd.DataFrame, genus_prefix: str) -> pd.DataFrame:
    """Keep only rows whose Taxon starts with genus_prefix.

    Must run before any other processing on a raw export -- these exports
    routinely mix in other genera and companion/cover species.
    """
    return df_raw[df_raw["Taxon"].astype(str).str.startswith(genus_prefix)].copy()


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


def _species_group(species: object, genus_prefix: str) -> str | None:
    """Species value, except the unresolved-to-species placeholder becomes NaN.

    Excludes it from ever forming its own species-tier deterioration model
    (mixing unrelated species into one curve would be meaningless) while
    still letting it participate fully in Origin/Global fits.
    """
    if species == f"{genus_prefix} spp.":
        return None
    return species


def adapt_raw_export(
    df_raw: pd.DataFrame, as_of_year: int, genus_prefix: str = "Astragalus"
) -> AdaptedGrinExport:
    """Adapt a raw GRIN-Global Curator Tool export into the shape data_prep.py consumes."""
    df = filter_to_genus(df_raw, genus_prefix)
    df = drop_placeholder_rows(df)

    parsed = df["Inventory Suffix"].apply(parse_inventory_suffix)
    lot_years = pd.Series([t[0] for t in parsed], index=df.index, dtype="float64")
    own_letters = [t[1] for t in parsed]

    rows = list(zip(df["Accession"], lot_years, own_letters))
    letters = infer_original_vs_increase(rows)

    viability_year = pd.to_datetime(df["Tested Date"], errors="coerce").dt.year.astype("float64")

    df_primary = pd.DataFrame(
        {
            "Accession": df["Accession"].to_numpy(),
            "SeedAge": (as_of_year - lot_years).to_numpy(),
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
    df_primary["SpeciesGroup"] = df_primary["Species"].apply(
        lambda s: _species_group(s, genus_prefix)
    )
    df_primary["EstLiveSeed"] = df_primary["EstTotalSeed"] * df_primary["Viability"] / 100

    df_borrowed = pd.DataFrame(
        columns=["Accession", "AgeAtTest", "Viability", "Species", "SpeciesGroup", "Origin"]
    )

    return AdaptedGrinExport(df_primary=df_primary, df_borrowed=df_borrowed)
