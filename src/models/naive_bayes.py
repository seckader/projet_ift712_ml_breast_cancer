from typing import Any, Dict

from sklearn.naive_bayes import GaussianNB

from src.models.base_model import BaseClassifier


class NaiveBayesClassifier(BaseClassifier):
    """
    Wrapper around sklearn's GaussianNB, adapted to the project skeleton.
    """

    name = "naive_bayes"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> GaussianNB:
        """
        Build the underlying sklearn GaussianNB estimator.
        """
        return GaussianNB()

    def hyperparam_grid(self) -> Dict[str, Any]:
        """
        Return the hyperparameter grid for GridSearchCV.

        Keys must match the Pipeline step name "classifier",
        so we use "classifier__<parameter_name>".
        """
        var_smoothing_values = self._hyperparameters.get(
            "var_smoothing",
            [1.0e-9],
        )

        return {
            "classifier__var_smoothing": var_smoothing_values,
        }