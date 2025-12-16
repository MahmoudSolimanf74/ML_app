# File: backend/app/utils.py
from typing import List
import pandas as pd


def validate_csv_columns(df: pd.DataFrame, required_columns: List[str]) -> List[str]:
    """
    Validate that `df` contains all `required_columns`.
    Returns a list of missing columns (empty list if none missing).
    """
    existing = list(df.columns)
    missing = [col for col in required_columns if col not in existing]
    return missing