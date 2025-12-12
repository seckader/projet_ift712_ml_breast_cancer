# decisiontree.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from sklearn.tree import DecisionTreeClassifier

from .base_model import BaseClassifier


@dataclass
class DecisionTreeWrapper(BaseClassifier):
    

    name: str = "decision_tree"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> DecisionTreeClassifier:
       
        return DecisionTreeClassifier(random_state=42)

    def hyperparam_grid(self) -> dict:
       
        return {
            "classifier__criterion": ["gini", "entropy", "log_loss"],
            "classifier__splitter": ["best", "random"],
            "classifier__max_depth": [None, 3, 5, 8, 12],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 2, 4],
            "classifier__max_features": [None, "sqrt", "log2"],
            "classifier__class_weight": [None, "balanced"],
        }
