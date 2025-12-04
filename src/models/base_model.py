from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from src.config.config import TrainingConfig
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class BaseModel(ABC):
    """Classe de base pour tous les modèles de classification.

    Chaque modèle concret doit implémenter :meth:`build_pipeline`.
    Ensuite, :meth:`fit_with_cv` gère la recherche d'hyperparamètres
    avec validation croisée stratifiée.
    """

    name: str
    training_config: TrainingConfig
    param_grid: Dict[str, Any] = field(default_factory=dict)

    best_estimator_: Optional[Pipeline] = field(init=False, default=None)
    best_params_: Optional[Dict[str, Any]] = field(init=False, default=None)
    cv_results_: Optional[Dict[str, Any]] = field(init=False, default=None)

    @abstractmethod
    def build_pipeline(self, preprocessor) -> Pipeline:
        """Construit le pipeline scikit-learn (préprocesseur + estimateur)."""
        raise NotImplementedError

    def fit_with_cv(self, X, y, preprocessor) -> None:
        """Effectue une GridSearchCV avec CV stratifiée.

        Les clés de self.param_grid correspondent aux hyperparamètres
        du classifieur et seront automatiquement préfixées par ``clf__``.
        """
        pipe = self.build_pipeline(preprocessor)
        grid = {f"clf__{k}": v for k, v in self.param_grid.items()}

        cfg = self.training_config
        cv = StratifiedKFold(
            n_splits=cfg.cv_folds,
            shuffle=True,
            random_state=cfg.random_seed,
        )

        logger.info(f"[{self.name}] Début GridSearchCV avec grille={grid}")
        search = GridSearchCV(
            estimator=pipe,
            param_grid=grid,
            scoring=cfg.scorer,
            n_jobs=cfg.n_jobs,
            cv=cv,
            refit=True,
            return_train_score=True,
        )
        search.fit(X, y)

        self.best_estimator_ = search.best_estimator_
        self.best_params_ = search.best_params_
        self.cv_results_ = search.cv_results_

        logger.info(f"[{self.name}] Meilleurs hyperparamètres : {self.best_params_}")

    # Méthodes de commodité pour la prédiction -------------------------------
    def predict(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")
        return self.best_estimator_.predict(X)

    def predict_proba(self, X):
        if self.best_estimator_ is None:
            raise RuntimeError("Le modèle n'a pas encore été entraîné.")
        if not hasattr(self.best_estimator_, "predict_proba"):
            raise AttributeError("Ce modèle ne supporte pas predict_proba.")
        return self.best_estimator_.predict_proba(X)
