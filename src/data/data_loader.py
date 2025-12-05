from typing import Tuple

import pandas as pd
from sklearn.datasets import load_breast_cancer

from src.config.config import Config
from src.utils.paths import DATA_RAW_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)

RAW_FILENAME = "breast-cancer.csv"


def _build_dataset_from_sklearn() -> pd.DataFrame:
    """
    Load the breast cancer dataset directly from sklearn
    and return it as a DataFrame.

    We do NOT hardcode any feature names here.
    """
    sk = load_breast_cancer(as_frame=True)
    df = sk.frame.copy()

    # Add an ID column at the beginning
    df.insert(0, "id", range(1, len(df) + 1))

    # Target column is 'target' (numeric 0/1)
    return df


def auto_detect_columns(config: Config, df: pd.DataFrame) -> None:
    """
    Automatically detect numerical and categorical feature columns
    from the DataFrame, without listing them anywhere.

    It fills config.dataset.numerical_features and
    config.dataset.categorical_features dynamically (in memory).
    """
    id_col = config.dataset.id_column
    target_col = config.dataset.target_column

    # All columns except ID and target = features
    feature_cols = [c for c in df.columns if c not in {id_col, target_col}]

    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [
        c for c in feature_cols
        if not pd.api.types.is_numeric_dtype(df[c])
    ]

    config.dataset.numerical_features = numeric_cols
    config.dataset.categorical_features = categorical_cols

    logger.info("Auto-detected numerical features: %s", numeric_cols)
    logger.info("Auto-detected categorical features: %s", categorical_cols)


def load_raw_dataset(config: Config) -> pd.DataFrame:
    """
    Load the dataset from sklearn, automatically detect the
    feature columns, and save a raw CSV copy.

    No manual CSV download is required and no column names
    are hardcoded for features.
    """
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    raw_path = DATA_RAW_DIR / RAW_FILENAME

    logger.info("Loading breast cancer dataset from sklearn...")
    df = _build_dataset_from_sklearn()

    # Automatically detect features and store them in config
    auto_detect_columns(config, df)

    # Save raw dataset for reproducibility
    df.to_csv(raw_path, index=False)
    logger.info("Raw dataset saved to %s", raw_path)

    return df


def get_features_and_target(
    config: Config,
    df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split a DataFrame into features (X) and target (y)
    using only id_column and target_column from the config.
    """
    target = config.dataset.target_column

    y = df[target]
    X = df.drop(columns=[target])

    if config.dataset.id_column and config.dataset.id_column in X.columns:
        X = X.drop(columns=[config.dataset.id_column])

    return X, y