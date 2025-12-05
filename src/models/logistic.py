from __future__ import annotations

from typing import Any, Dict

from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseClassifier


class LogisticRegressionModel(BaseClassifier):
    """
    Implémentation conforme à BaseClassifier pour la régression logistique.
    """

    name = "logistic"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        """
        hyperparameters vient de models.yaml (section logistic.grid ou logistic.hyperparameters,
        selon comment ton Config le construit).
        On les stocke pour construire la grille dans hyperparam_grid().
        """
        self.hyperparameters = hyperparameters or {}

    def build_estimator(self) -> BaseEstimator:
        """
        Construit l'estimateur sklearn de base.
        Les hyperparamètres seront ajustés par GridSearchCV via hyperparam_grid().
        """
        return LogisticRegression(
            max_iter=1000,  
            multi_class="auto"
        )

    def hyperparam_grid(self) -> Dict[str, Any]:
        """
        Transforme le dict d'hyperparamètres (venant du YAML) en noms compatibles
        avec le Pipeline : 'classifier__C', 'classifier__penalty', etc.
        """
        return {
            f"classifier__{k}": v
            for k, v in self.hyperparameters.items()
        }
