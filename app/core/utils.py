"""Utility functions for reproducibility and helpers."""
import numpy as np
import pandas as pd
from typing import Optional

# Minimum subgroup sample size below which fairness verdicts are statistically
# unreliable. Mirrors the F2 fix applied to the MCP server (2026-09-18/19
# webapp-parity remediation) so the Streamlit demo cannot show a confident
# PASS/FAIL on a handful of rows.
MIN_SUBGROUP_SAMPLE_SIZE = 30

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy and pandas."""
    np.random.seed(seed)

def load_sample_data(path: str = "data/sample_datasets/hr_candidates.csv") -> pd.DataFrame:
    """Load the sample HR candidates dataset."""
    return pd.read_csv(path)