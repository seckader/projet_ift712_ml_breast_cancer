from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator

from src.models.base_model import BaseClassifier


@dataclass
class MLPModel(BaseClassifier):
    """
    Wrapper MLP conforme au contrat BaseClassifier.

    - `hyperparameters` contient la grille lue dans models.yaml
      (sans préfixe `classifier__`).
    - `build_estimator` construit l'estimateur de base.
    - `hyperparam_grid` ajoute le préfixe `classifier__` pour GridSearchCV.
    """

    name: str = "mlp"
    _hyperparameters: Dict[str, Any] = field(default_factory=dict)

    def __init__(self, hyperparameters: Optional[Dict[str, Any]] = None) -> None:
        self.name = "mlp"
        # Grille brute venant du YAML, ex:
        # {"hidden_layer_sizes": [...], "activation": [...], ...}
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> BaseEstimator:
        """
        Construit l'estimateur sklearn de base.

        Les paramètres qui dépendent de la recherche d'hyperparamètres
        seront réglés par GridSearchCV via `hyperparam_grid`.
        """
        return MLPClassifier(
            max_iter=2000,
            random_state=42,  # ou une seed venant de ta config globale si tu veux
        )

    def hyperparam_grid(self) -> Dict[str, Any]:
        """
        Transforme les hyperparameters bruts en grille pour le Pipeline.

        Exemple:
            _hyperparameters = {"hidden_layer_sizes": [...], "activation": [...]}
        devient
            {"classifier__hidden_layer_sizes": [...],
             "classifier__activation": [...]}
        """
        grid: Dict[str, Any] = {}
        for param, values in self._hyperparameters.items():
            grid[f"classifier__{param}"] = values
        return grid
