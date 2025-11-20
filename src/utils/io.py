from __future__ import annotations
import os
import pandas as pd
import joblib
from .paths import path_join

def save_csv(df: pd.DataFrame, rel_path: str) -> str:
    path = path_join(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    return path

def load_csv(rel_path: str) -> pd.DataFrame:
    path = path_join(rel_path)
    return pd.read_csv(path)

def save_artifact(obj, rel_path: str) -> str:
    path = path_join(rel_path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)
    return path

def load_artifact(rel_path: str):
    path = path_join(rel_path)
    return joblib.load(path)
