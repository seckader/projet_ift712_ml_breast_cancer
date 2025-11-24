from __future__ import annotations
import pandas as pd
from sklearn.model_selection import train_test_split

from src.config.config import load_dataset_config, load_training_config
from src.utils.io import save_csv


def make_splits(df: pd.DataFrame) -> None:
    """Crée les fichiers train.csv et test.csv dans data/interim/."""
    dcfg = load_dataset_config()
    tcfg = load_training_config()

    X = df.drop(columns=[dcfg.target])
    y = df[dcfg.target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=tcfg.test_size,
        random_state=tcfg.random_seed,
        stratify=y,
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    save_csv(train_df, "data/interim/train.csv")
    save_csv(test_df, "data/interim/test.csv")
