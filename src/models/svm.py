from typing import Any, Dict

from sklearn.svm import SVC

from src.models.base_model import BaseClassifier


class SVMClassifier(BaseClassifier):
    """
    Wrapper around sklearn's SVC, adapted to the project skeleton.
    """

    name = "svm"

    def __init__(self, hyperparameters: Dict[str, Any] | None = None) -> None:
        """
        Parameters
        ----------
        hyperparameters : dict, optional
            Hyperparameter values coming from configs/models.yaml.
        """
        self._hyperparameters = hyperparameters or {}

    def build_estimator(self) -> SVC:
        """
        Build the underlying sklearn SVC estimator.

        We keep `probability=False` (default) for faster training.
        """
        return SVC()

    def hyperparam_grid(self) -> Dict[str, Any]:
        """
        Return the hyperparameter grid for GridSearchCV.

        Keys must match the Pipeline step name "classifier",
        so we use "classifier__<parameter_name>".
        """
        C_values = self._hyperparameters.get("C", [1.0])
        kernel_values = self._hyperparameters.get("kernel", ["rbf"])
        gamma_values = self._hyperparameters.get("gamma", ["scale"])

        return {
            "classifier__C": C_values,
            "classifier__kernel": kernel_values,
            "classifier__gamma": gamma_values,
        }