from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config.config import DatasetConfig


def build_preprocessor(
    dataset_cfg: DatasetConfig,
    use_scaler: bool,
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that applies preprocessing to the features.
    """
    numeric_features: List[str] = dataset_cfg.numerical_features
    categorical_features: List[str] = dataset_cfg.categorical_features

    transformers = []

    if numeric_features:
        if use_scaler:
            transformers.append(
                ("num", StandardScaler(), numeric_features)
            )
        else:
            transformers.append(
                ("num", "passthrough", numeric_features)
            )

    if categorical_features:
        transformers.append(
            ("cat", "passthrough", categorical_features)
        )

    preprocessor = ColumnTransformer(transformers=transformers)

    return preprocessor


def build_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    """
    Build a sklearn Pipeline: preprocessing -> classifier.
    """
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )
