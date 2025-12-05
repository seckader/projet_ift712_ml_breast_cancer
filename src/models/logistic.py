from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config.config import TrainingConfig
from src.models.base_model import BaseModel


@dataclass
class LogisticRegressionModel(BaseModel):
    """Régression logistique pour la classification du cancer du sein.

    Le modèle est entraîné via :meth:`fit_with_cv`, qui effectue
    une recherche d'hyperparamètres avec validation croisée stratifiée.
    """

    def __init__(self, training_config: TrainingConfig, param_grid: Dict[str, Any]):
        super().__init__(
            name="logistic",
            training_config=training_config,
            param_grid=param_grid,
        )

    def build_pipeline(self, preprocessor) -> Pipeline:
        clf = LogisticRegression(
            max_iter=1000,
            multi_class="auto",
            n_jobs=self.training_config.n_jobs,
            random_state=self.training_config.random_seed,
        )
        pipe = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("clf", clf),
            ]
        )
        return pipe
