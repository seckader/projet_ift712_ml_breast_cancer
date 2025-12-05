from typing import Dict

from sklearn.model_selection import GridSearchCV

from src.evaluation.crossval import build_cv_splits


def build_grid_search(
    pipeline,
    param_grid: Dict,
    training_cfg,
) -> GridSearchCV:
    """
    Build a GridSearchCV instance consistent with the training configuration.
    """
    cv = build_cv_splits(training_cfg.cv)

    verbose = getattr(training_cfg, "verbose", 1)

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring=training_cfg.scoring,
        cv=cv,
        n_jobs=training_cfg.n_jobs,
        verbose=verbose,
        refit=training_cfg.refit_metric,
        return_train_score=True,
    )
    return grid