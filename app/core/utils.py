"""Utility functions for reproducibility and helpers."""
import numpy as np
import pandas as pd
from typing import Optional

def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across numpy and pandas."""
    np.random.seed(seed)

def load_sample_data(path: str = "data/sample_datasets/hr_candidates.csv") -> pd.DataFrame:
    """Load the sample HR candidates dataset."""
    return pd.read_csv(path)