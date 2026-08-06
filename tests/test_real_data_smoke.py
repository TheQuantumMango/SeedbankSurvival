"""Runs the full pipeline against real local GRIN-Global export data, if present.

docs/ is gitignored (private, local-only), so this test self-skips in a fresh
clone or CI. It exists to catch things a small synthetic fixture can't: real
data volume, real messiness, real column ordering differences between files.

Shape/range assertions only, not exact values -- the real data will eventually
be replaced by different GRIN exports as the tool moves off the reformatted
class-project files.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import seedbank_survival as sb

DOCS_DIR = Path(__file__).parents[1] / "docs"
REAL_DATA_FILES = sorted(DOCS_DIR.glob("*.csv")) if DOCS_DIR.exists() else []


@pytest.mark.skipif(not REAL_DATA_FILES, reason="requires local docs/*.csv (gitignored, not present in a fresh clone)")
@pytest.mark.parametrize("path", REAL_DATA_FILES, ids=lambda p: p.stem)
def test_pipeline_runs_end_to_end_on_real_export(path):
    df = sb.load_accessions(path)
    df_clean = sb.clean_ages(df)
    df_model = sb.build_model_dataset(df_clean)
    df_ranking = sb.build_ranking_dataset(df_clean)

    assert len(df_model) > 0
    assert len(df_ranking) > 0

    global_model = sb.fit_global_model(df_model)
    species_models = sb.fit_group_models(df_model, "Species")
    origin_models = sb.fit_group_models(df_model, "Origin")

    df_ranking = sb.predict_hierarchical(df_ranking, species_models, origin_models, global_model)
    assert df_ranking["PredictedViability_2026"].between(0, 100).all()
    assert set(df_ranking["ModelUsed"].unique()) <= {"Species", "Origin", "Global"}

    priority_table = sb.build_priority_table(
        df_ranking, species_models, origin_models, global_model, top_n=50
    )
    assert len(priority_table) == min(50, len(df_ranking))
    assert priority_table["EstimatedViability_2026"].between(0, 100).all()
    assert priority_table["EstimatedViability_2026"].is_monotonic_increasing
