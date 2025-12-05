import importlib

import sys, os
sys.path.append(os.path.abspath("."))

import pandas as pd
from joblib import dump

from src.config.config import Config, ModelConfig
from src.data.data_loader import get_features_and_target, auto_detect_columns
from src.features.preprocessing import build_preprocessor, build_pipeline
from src.evaluation.tuning import build_grid_search
from src.utils.io import save_csv
from src.utils.logging import get_logger
from src.utils.paths import DATA_INTERIM_DIR, MODELS_ARTIFACTS_DIR, MODELS_REPORTS_DIR

logger = get_logger(__name__)


def _import_model_class(class_path: str):
    """
    Dynamically import a model class from its full class path.
    """
    module_name, class_name = class_path.rsplit(".", maxsplit=1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def train_single_model(
    config: Config,
    model_cfg: ModelConfig,
    train_df: pd.DataFrame,
) -> None:
    """
    Train a single model using GridSearchCV.
    """
    logger.info("===== Training model: %s =====", model_cfg.name)

    # 1) Auto-detect feature columns from the training data
    auto_detect_columns(config, train_df)

    # 2) Split features/target
    X, y = get_features_and_target(config, train_df)

    # 3) Build preprocessor using the detected features
    preprocessor = build_preprocessor(
        dataset_cfg=config.dataset,
        use_scaler=model_cfg.use_scaler,
    )

    ModelClass = _import_model_class(model_cfg.class_path)
    model_wrapper = ModelClass(hyperparameters=model_cfg.hyperparameters)

    estimator = model_wrapper.build_estimator()
    pipeline = build_pipeline(preprocessor, estimator)
    param_grid = model_wrapper.hyperparam_grid()

    grid_search = build_grid_search(
        pipeline=pipeline,
        param_grid=param_grid,
        training_cfg=config.training,
    )

    grid_search.fit(X, y)

    logger.info("Best params: %s", grid_search.best_params_)
    logger.info(
        "Best %s score: %.4f",
        config.training.refit_metric,
        grid_search.best_score_,
    )

    MODELS_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_ARTIFACTS_DIR / f"{model_cfg.name}_best.joblib"
    dump(grid_search.best_estimator_, model_path)
    logger.info("Best model saved to %s", model_path)

    MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    cv_results_path = MODELS_REPORTS_DIR / f"{model_cfg.name}_cv_results.csv"
    save_csv(pd.DataFrame(grid_search.cv_results_), cv_results_path)
    logger.info("CV results saved to %s", cv_results_path)


def main():
    config = Config()

    train_path = DATA_INTERIM_DIR / "train.csv"
    if not train_path.exists():
        raise FileNotFoundError("Run scripts.prepare_data first.")

    train_df = pd.read_csv(train_path)

    for model_name, model_cfg in config.models.models.items():
        if not model_cfg.enabled:
            logger.info("Model %s disabled, skipping.", model_name)
            continue
        train_single_model(config, model_cfg, train_df)


if __name__ == "__main__":
    main()
