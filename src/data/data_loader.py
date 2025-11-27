from __future__ import annotations
import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config.config import load_dataset_config
from src.utils.io import save_csv


def export_sklearn_to_csv() -> str:
    """Charge le dataset Breast Cancer de sklearn et l'exporte en CSV brut."""
    data = load_breast_cancer()
    data_frame = pd.DataFrame(data.data, columns=data.feature_names)
    data_frame["target"] = data.target
    return save_csv(data_frame, "data/raw/breast_cancer.csv")


def load_dataset() -> pd.DataFrame:
    """Charge le dataset."""
    config = load_dataset_config()
    
    if config.use_sklearn_loader:
        path = export_sklearn_to_csv()
        return pd.read_csv(path)
    else:
        from src.utils.io import load_csv
        return load_csv(config.csv_path)
