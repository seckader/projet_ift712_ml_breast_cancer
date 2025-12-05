from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import Config
from src.utils.paths import DATA_INTERIM_DIR
from src.utils.logging import get_logger

logger = get_logger(__name__)


def create_train_test_split(
    config: Config, df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create train/test splits and save them under data/interim.
    """
    DATA_INTERIM_DIR.mkdir(parents=True, exist_ok=True)

    target_col = config.dataset.target_column
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.training.test_size,
        random_state=config.training.random_state,
        stratify=y,
    )

    train_df = X_train.copy()
    train_df[target_col] = y_train

    test_df = X_test.copy()
    test_df[target_col] = y_test

    train_path = DATA_INTERIM_DIR / "train.csv"
    test_path = DATA_INTERIM_DIR / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    logger.info("Train split saved to %s", train_path)
    logger.info("Test split saved to %s", test_path)

    return train_df, test_df
