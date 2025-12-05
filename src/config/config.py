from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml

from src.utils.paths import CONFIG_DIR


@dataclass
class DatasetConfig:
    name: str
    target_column: str
    id_column: str | None
    numerical_features: list[str]
    categorical_features: list[str]


@dataclass
class TrainingConfig:
    test_size: float
    random_state: int
    scoring: str
    n_jobs: int
    refit_metric: str
    cv: Dict[str, Any]


@dataclass
class ModelConfig:
    name: str
    class_path: str
    enabled: bool
    use_scaler: bool
    hyperparameters: Dict[str, Any]


@dataclass
class ModelsConfig:
    models: Dict[str, ModelConfig]


class Config:
    """
    Main configuration loader for the project.
    Loads dataset.yaml, training.yaml, models.yaml.
    """

    def __init__(self) -> None:
        self.dataset = self._load_dataset_config()
        self.training = self._load_training_config()
        self.models = self._load_models_config()

    @staticmethod
    def _load_yaml(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_dataset_config(self) -> DatasetConfig:
        data = self._load_yaml(CONFIG_DIR / "dataset.yaml")
        return DatasetConfig(
            name=data["name"],
            target_column=data["target_column"],
            id_column=data.get("id_column"),
            numerical_features=data.get("numerical_features", []),
            categorical_features=data.get("categorical_features", []),
        )

    def _load_training_config(self) -> TrainingConfig:
        data = self._load_yaml(CONFIG_DIR / "training.yaml")
        return TrainingConfig(
            test_size=data["test_size"],
            random_state=data["random_state"],
            scoring=data["scoring"],
            n_jobs=data["n_jobs"],
            refit_metric=data["refit_metric"],
            cv=data["cv"],
        )

    def _load_models_config(self) -> ModelsConfig:
        data = self._load_yaml(CONFIG_DIR / "models.yaml")
        model_dict: Dict[str, ModelConfig] = {}

        for name, cfg in data["models"].items():
            model_dict[name] = ModelConfig(
                name=name,
                class_path=cfg["class_path"],
                enabled=cfg.get("enabled", True),
                use_scaler=cfg.get("use_scaler", False),
                hyperparameters=cfg.get("hyperparameters", {}),
            )

        return ModelsConfig(models=model_dict)
