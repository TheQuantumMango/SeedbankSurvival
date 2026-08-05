"""Confidence-weighted tier selection (hierarchical.predict_hierarchical). Uses
hand-built SlopeModel instances directly -- this behavior applies equally to
old-schema and raw-GRIN data, so it doesn't need either fixture.
"""
from __future__ import annotations

import math

import pandas as pd

from seedbank_survival.deterioration import SlopeModel
from seedbank_survival.hierarchical import predict_hierarchical


def _df_ranking(**row):
    defaults = {"Accession": "A1", "SeedAge": 10, "Species": "S1", "Origin": "O1"}
    defaults.update(row)
    return pd.DataFrame([defaults])


def test_keeps_species_tier_when_it_has_higher_confidence():
    species_models = {"S1": SlopeModel(intercept=90, slope=-1.0, n=10, r2=0.9)}
    global_model = SlopeModel(intercept=80, slope=-0.5, n=100, r2=0.3)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Species"
    assert result.loc[0, "ModelConfidence"] == 0.9


def test_overrides_to_global_when_global_has_higher_confidence():
    species_models = {"S1": SlopeModel(intercept=90, slope=-1.0, n=3, r2=0.2)}
    global_model = SlopeModel(intercept=80, slope=-0.5, n=1000, r2=0.7)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.7
    assert result.loc[0, "PredictedViability_2026"] == 80 - 0.5 * 10


def test_overrides_origin_tier_to_global_when_global_has_higher_confidence():
    origin_models = {"O1": SlopeModel(intercept=85, slope=-1.2, n=3, r2=0.4)}
    global_model = SlopeModel(intercept=80, slope=-0.5, n=1000, r2=0.6)

    result = predict_hierarchical(_df_ranking(), {}, origin_models, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"


def test_global_tier_never_overrides_itself():
    global_model = SlopeModel(intercept=80, slope=-0.5, n=1000, r2=0.6)

    result = predict_hierarchical(_df_ranking(), {}, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.6


def test_nan_confidence_defers_to_global():
    # A near-zero-variance group (e.g. 3 nearly-identical Viability values)
    # can produce an undefined R^2 -- real on actual data, not hypothetical.
    species_models = {"S1": SlopeModel(intercept=96.0, slope=0.0, n=3, r2=float("nan"))}
    global_model = SlopeModel(intercept=80, slope=-0.5, n=1000, r2=0.6)

    result = predict_hierarchical(_df_ranking(), species_models, {}, global_model)

    assert result.loc[0, "ModelUsed"] == "Global"
    assert result.loc[0, "ModelConfidence"] == 0.6
    assert not math.isnan(result.loc[0, "ModelConfidence"])
