from __future__ import annotations

from typing import Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from src.config.config import DatasetConfig


def build_preprocessor(
    df: pd.DataFrame, cfg: DatasetConfig
) -> Tuple[ColumnTransformer, list[str]]:
    """Construit le préprocesseur pour le jeu de données.

    Actuellement :
    - standardisation des variables numériques.

    Les colonnes numériques sont soit :
    - données explicitement dans cfg.numeric_features, soit
    - toutes les colonnes non-cible si cette liste est vide.
    """
    if cfg.numeric_features:
        numeric_features = cfg.numeric_features
    else:
        numeric_features = [c for c in df.columns if c != cfg.target]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
        ],
        remainder="drop",
    )

    return preprocessor, numeric_features
