from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.tree import DecisionTreeClassifier

from src.models.base_model import BaseClassifier


@dataclass
class DecisionTreeWrapper(BaseClassifier):
    

    name: str = "decision_tree"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> DecisionTreeClassifier:
       
        return DecisionTreeClassifier(random_state=42)

    def hyperparam_grid(self) -> dict:
       
        return self._hyperparameters
