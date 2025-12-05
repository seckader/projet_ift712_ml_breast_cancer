from __future__ import annotations

from typing import Callable, Dict, List

from src.config.config import TrainingConfig
from src.models.base_model import BaseModel
from src.models.logistic import LogisticRegressionModel
from src.models.mlp import MLPModel

ModelBuilder = Callable[[TrainingConfig, dict], BaseModel]


def _build_logistic(cfg: TrainingConfig, model_cfg: dict) -> BaseModel:
    return LogisticRegressionModel(cfg, model_cfg.get("grid", {}))


def _build_mlp(cfg: TrainingConfig, model_cfg: dict) -> BaseModel:
    return MLPModel(cfg, model_cfg.get("grid", {}))


MODEL_BUILDERS: Dict[str, ModelBuilder] = {
    "logistic": _build_logistic,
    "mlp": _build_mlp,
}


def get_available_models() -> List[str]:
    return sorted(MODEL_BUILDERS.keys())


def build_model(name: str, training_cfg: TrainingConfig, models_cfg: dict) -> BaseModel:
    if name not in MODEL_BUILDERS:
        raise ValueError(
            f"Modèle '{name}' non supporté. "
            f"Modèles disponibles : {', '.join(get_available_models())}"
        )
    if name not in models_cfg:
        raise KeyError(f"Modèle '{name}' non défini dans configs/models.yaml.")

    builder = MODEL_BUILDERS[name]
    return builder(training_cfg, models_cfg[name])
