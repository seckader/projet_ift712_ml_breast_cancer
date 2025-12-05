from abc import ABC, abstractmethod
from typing import Any, Dict

from sklearn.base import BaseEstimator


class BaseClassifier(ABC):
    """
    Common abstract base class for all classifiers in the project.
    """

    name: str

    @abstractmethod
    def build_estimator(self) -> BaseEstimator:
        """Build and return the underlying sklearn estimator."""
        raise NotImplementedError

    @abstractmethod
    def hyperparam_grid(self) -> Dict[str, Any]:
        """
        Return the hyperparameter grid to be used with GridSearchCV.
        Keys must match the Pipeline parameter names
        (e.g. 'classifier__var_smoothing').
        """
        raise NotImplementedError