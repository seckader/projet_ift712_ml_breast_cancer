from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline

from src.models.base_model import BaseModel
from src.config.config import TrainingConfig


@dataclass
class MLPModel(BaseModel):
    """Réseau de neurones MLP pour la classification du cancer du sein."""

    def __init__(self, training_config: TrainingConfig, param_grid: Dict[str, Any]):
        super().__init__(
            name="mlp",
            training_config=training_config,
            param_grid=param_grid,
        )

    def build_pipeline(self, preprocessor) -> Pipeline:
        clf = MLPClassifier(
            max_iter=2000,
            random_state=self.training_config.random_seed,
        )
        return Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", clf),
            ]
        )
