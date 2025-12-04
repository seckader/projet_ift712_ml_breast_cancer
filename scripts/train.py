from __future__ import annotations

import argparse
import os
import sys
from typing import Iterable, List

sys.path.append(os.path.abspath("."))

from src.config.config import (
    load_dataset_config,
    load_training_config,
    load_models_config,
)
from src.features.preprocessing import build_preprocessor
from src.models.registry import build_model, get_available_models
from src.utils.io import load_csv, save_artifact
from src.utils.logging import get_logger
from src.utils.paths import path_join
from src.utils.plotting import plot_confusion_matrix, save_classification_report

logger = get_logger("train")


def run_training_for_models(model_names: Iterable[str]) -> None:
    """Entraîne et évalue une liste de modèles."""
    dataset_cfg = load_dataset_config()
    training_cfg = load_training_config()
    models_cfg = load_models_config()

    train_df = load_csv("data/interim/train.csv")
    test_df = load_csv("data/interim/test.csv")

    X_train = train_df.drop(columns=[dataset_cfg.target])
    y_train = train_df[dataset_cfg.target]
    X_test = test_df.drop(columns=[dataset_cfg.target])
    y_test = test_df[dataset_cfg.target]

    logger.info("Shape train=%s, test=%s", X_train.shape, X_test.shape)

    preprocessor, numeric_features = build_preprocessor(train_df, dataset_cfg)
    logger.info("Features numériques utilisées : %s", numeric_features)

    for name in model_names:
        logger.info("=== Entraînement du modèle '%s' ===", name)
        model = build_model(name, training_cfg, models_cfg)
        model.fit_with_cv(X_train, y_train, preprocessor)

        y_pred = model.predict(X_test)

        out_dir = path_join("models", "artifacts", name)
        os.makedirs(out_dir, exist_ok=True)

        save_artifact(model, os.path.join(out_dir, "model.joblib"))

        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrix(
            y_true=y_test,
            y_pred=y_pred,
            title=f"{name} – Breast Cancer (test)",
            out_path=cm_path,
        )

        report_path = os.path.join(out_dir, "classification_report.txt")
        target_names = [str(c) for c in sorted(y_test.unique())]
        save_classification_report(
            y_true=y_test,
            y_pred=y_pred,
            target_names=target_names,
            out_path=report_path,
        )

        logger.info(
            "Modèle '%s' entraîné. Artefacts sauvegardés dans %s", name, out_dir
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="IFT712 – Entraînement de modèles de classification."
    )
    parser.add_argument(
        "--models",
        type=str,
        default="all",
        help=(
            "Liste de modèles à entraîner, séparés par des virgules "
            f"(disponibles: {', '.join(get_available_models())}), "
            "ou 'all' pour tous."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.models == "all":
        models_to_run: List[str] = get_available_models()
    else:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    logger.info("Modèles sélectionnés : %s", models_to_run)
    run_training_for_models(models_to_run)


if __name__ == "__main__":
    main()
