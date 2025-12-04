from __future__ import annotations
from dataclasses import dataclass, field
import os
import yaml

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "configs")


@dataclass
class DatasetConfig:
    """Configuration liée au jeu de données."""
    target: str
    use_sklearn_loader: bool = True
    csv_path: str = "data/raw/breast_cancer.csv"
    numeric_features: list[str] = field(default_factory=list)
    categorical_features: list[str] = field(default_factory=list)
    drop_columns: list[str] = field(default_factory=list)


@dataclass
class TrainingConfig:
    """Configuration globale d'entraînement et de validation."""
    random_seed: int = 42
    cv_folds: int = 5
    test_size: float = 0.2
    scorer: str = "f1_macro"
    n_jobs: int = -1


def _load_yaml(filename: str) -> dict:
    """Charge un fichier YAML à partir du dossier configs/ et retourne un dict."""
    path = os.path.join(CONFIG_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier de configuration introuvable : {path}")
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def load_dataset_config() -> DatasetConfig:
    data = _load_yaml("dataset.yaml")
    return DatasetConfig(**data)


def load_training_config() -> TrainingConfig:
    data = _load_yaml("training.yaml")
    return TrainingConfig(**data)


def load_models_config() -> dict:
    return _load_yaml("models.yaml")
