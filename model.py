# model.py
import os
import joblib
import pandas as pd
import numpy as np
import re
from typing import Any

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # بيرجع لـ Model_app
PIPE_PATH = r"G:\programing\Model_app\ML\model\pipe.pkl"
COLUMNS_PATH = r"G:\programing\Model_app\ML\model\columns.pkl"

pipe = joblib.load(PIPE_PATH)       # Pipeline كامل: preprocessing + model
model_columns = joblib.load(COLUMNS_PATH)


def normalize_cpu(cpu: str) -> str:
    """Normalize CPU strings for the model."""
    if not isinstance(cpu, str):
        return ""
    s = cpu.strip()
    if s == "":
        return ""
    # Remove GHz patterns
    s = re.sub(r"(?i)\b\d+(?:\.\d+)?\s*ghz\b", "", s)
    s = re.sub(r"(?i)ghz", "", s)
    # Remove stray punctuation
    s = re.sub(r"[\.,;()\[\]]+", "", s)
    # Collapse multiple spaces
    s = re.sub(r"\s+", " ", s).strip()
    return s


def bool_to_int(value: Any) -> int:
    """Convert a variety of boolean-like values to 1/0."""
    if isinstance(value, bool):
        return 1 if value else 0
    if value is None:
        return 0
    if isinstance(value, (int, float)):
        try:
            return 1 if int(value) != 0 else 0
        except Exception:
            return 0
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "1", "yes", "y", "t"):
            return 1
        if v in ("false", "0", "no", "n", "f"):
            return 0
    return 0

def predict_single(input_data: dict) -> int:
    """Predict a single instance."""
    df = pd.DataFrame([input_data])

    # Convert boolean columns
    for col in ("touch_screen", "IPS_display"):
        df[col] = df.get(col, 0).apply(bool_to_int) if col in df else 0

    # Normalize CPU column
    df["CPU"] = df.get("CPU", "").apply(normalize_cpu)

    # Capitalize string columns before passing to the model
    STRING_COLUMNS = {"Company", "TypeName", "GPU_brand", "OS"}
    for col in STRING_COLUMNS:
        if col in df:
            df[col] = df[col].astype(str).str.capitalize()
    
    # Ensure all model columns exist
    for col in model_columns:
        if col not in df:
            df[col] = "" if col in STRING_COLUMNS else 0

    # Reorder columns
    df = df[model_columns]

    # Predict and exponentiate
    prediction_log = pipe.predict(df)
    prediction = np.exp(prediction_log)

    return int(round(prediction[0]))



def predict_batch(df: pd.DataFrame) -> list[int]:
    """Predict multiple instances at once."""
    STRING_COLUMNS = {"Company", "TypeName", "CPU", "GPU_brand", "OS"}
    for col in model_columns:
        if col not in df:
            df[col] = "" if col in STRING_COLUMNS else 0
    df = df[model_columns]
    predictions_log = pipe.predict(df)
    predictions = np.exp(predictions_log)
    return [int(round(p)) for p in predictions]
