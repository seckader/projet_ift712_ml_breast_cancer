from __future__ import annotations
from dataclasses import dataclass, field
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


@dataclass
class DatasetConfig:
    target: str
    use_sklearn_loader: bool = True
    csv_path: str = "data/raw/breast_cancer.csv"
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    random_seed: int
    cv_folds: int
    test_size: float
    scorer: str
    n_jobs: int


def _load_yaml(name: str) -> dict:
    path = os.path.join(CONFIG_DIR, name)
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_dataset_config() -> DatasetConfig:
    data = _load_yaml("dataset.yaml")
    return DatasetConfig(**data)


def load_training_config() -> TrainingConfig:
    data = _load_yaml("training.yaml")
    return TrainingConfig(**data)


def load_models_config() -> dict:
    return _load_yaml("models.yaml")
