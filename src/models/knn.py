from __future__ import annotations
from dataclasses import dataclass

from sklearn.neighbors import KNeighborsClassifier

from .base_model import BaseClassifier


@dataclass
class KNNClassifier(BaseClassifier):
    """
    Wrapper OO autour de KNeighborsClassifier de scikit-learn.
    Il fournit :
    - un nom unique (`name`)
    - une méthode pour construire l'estimateur sklearn
    - une grille d'hyperparamètres par défaut
    """

    name: str = "knn"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> KNeighborsClassifier:
        """
        Construit l'estimateur KNN non entraîné.
        La recherche d'hyperparamètres ajustera les attributs plus tard.
        """
        return KNeighborsClassifier()

    def hyperparam_grid(self) -> dict:
        """
        Grille d'hyperparamètres par défaut pour KNN.
        Elle pourra être combinée/écrasée par configs/models.yaml.
        """
        return {
            "classifier__n_neighbors": [3, 5, 7, 9],
            "classifier__weights": ["uniform", "distance"],
            "classifier__p": [1, 2],
        }
