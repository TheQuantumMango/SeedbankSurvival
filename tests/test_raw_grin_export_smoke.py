"""Runs the full pipeline against a real raw GRIN-Global Curator Tool export, if
present. Parallels test_real_data_smoke.py's pattern but for the raw-export path
(grin_import.adapt_raw_export) rather than the reformatted CSV/XLSX path.

docs/ is gitignored (private, local-only), so this test self-skips in a fresh
clone or CI.
"""
from __future__ import annotations

from pathlib import Path

import pytest

import seedbank_survival as sb
from seedbank_survival.grin_import import adapt_raw_export, assemble_model_dataset

DOCS_DIR = Path(__file__).parents[1] / "docs"
RAW_EXPORT_PATH = DOCS_DIR / "RawGRINAstragalusExport.xlsx"
ACCESSIONS_PATH = DOCS_DIR / "RawGRINAstragalusExportAccessions.xlsx"


@pytest.mark.skipif(
    not RAW_EXPORT_PATH.exists(),
    reason="requires local docs/RawGRINAstragalusExport.xlsx (gitignored, not present in a fresh clone)",
)
def test_pipeline_runs_end_to_end_on_raw_export():
    df_raw = sb.load_accessions(RAW_EXPORT_PATH)
    adapted = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")

    assert len(adapted.df_primary) > 0
    assert adapted.df_primary["Species"].astype(str).str.startswith("Astragalus").all()

    df_primary_clean = sb.clean_ages(adapted.df_primary)
    df_model = assemble_model_dataset(df_primary_clean, adapted.df_borrowed)
    df_ranking = sb.build_ranking_dataset(df_primary_clean)
    df_inventory = sb.build_inventory_view(df_primary_clean)

    assert len(df_model) > 0
    assert len(df_ranking) > 0
    assert len(df_inventory) >= len(df_ranking)  # no per-accession dedup

    global_model = sb.fit_global_model(df_model)
    species_models = sb.fit_group_models(df_model, "SpeciesGroup", min_n=3)
    origin_models = sb.fit_group_models(df_model, "Origin", min_n=3)
    assert "Astragalus spp." not in species_models

    df_ranking = sb.predict_hierarchical(df_ranking, species_models, origin_models, global_model)
    assert df_ranking["PredictedViability_2026"].between(0, 100).all()
    assert df_ranking["ModelConfidence"].between(0, 1).all()

    priority_table = sb.build_priority_table(
        df_ranking, species_models, origin_models, global_model, top_n=50
    )
    assert len(priority_table) == min(50, len(df_ranking))
    assert priority_table["EstimatedViability_2026"].between(0, 100).all()


@pytest.mark.skipif(
    not (RAW_EXPORT_PATH.exists() and ACCESSIONS_PATH.exists()),
    reason="requires local docs/RawGRINAstragalusExport.xlsx and docs/RawGRINAstragalusExportAccessions.xlsx (gitignored)",
)
def test_accession_file_fallback_resolves_more_seed_ages():
    df_raw = sb.load_accessions(RAW_EXPORT_PATH)
    df_accessions = sb.load_accessions(ACCESSIONS_PATH)

    without = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus")
    with_acc = adapt_raw_export(df_raw, as_of_year=2026, genera="Astragalus", df_accessions=df_accessions)

    resolved_without = without.df_primary["SeedAge"].notna().sum()
    resolved_with = with_acc.df_primary["SeedAge"].notna().sum()
    assert resolved_with > resolved_without

    # AgeAtTest / df_borrowed use a separate chain -- must be unaffected.
    assert without.df_primary["AgeAtTest"].notna().sum() == with_acc.df_primary["AgeAtTest"].notna().sum()
    assert len(without.df_borrowed) == len(with_acc.df_borrowed)
