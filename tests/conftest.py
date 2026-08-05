from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def synthetic_accessions_df() -> pd.DataFrame:
    return pd.read_csv(FIXTURES_DIR / "synthetic_accessions.csv")
