from __future__ import annotations

import os
import sys

sys.path.append(os.path.abspath("."))

from src.config.config import (
    load_dataset_config,
    load_training_config,
    load_models_config,
)
from src.features.preprocessing import build_preprocessor
from src.models.logistic import LogisticRegressionModel
from src.utils.io import load_csv, save_artifact
from src.utils.logging import get_logger
from src.utils.paths import path_join
from src.utils.plotting import plot_confusion_matrix, save_classification_report

logger = get_logger("train")


def main() -> None:
    # 1) Configurations
    dataset_cfg = load_dataset_config()
    training_cfg = load_training_config()
    models_cfg = load_models_config()
    logistic_cfg = models_cfg.get("logistic")
    if logistic_cfg is None:
        raise KeyError("Section 'logistic' manquante dans configs/models.yaml")

    # 2) Chargement des données train/test
    train_df = load_csv("data/interim/train.csv")
    test_df = load_csv("data/interim/test.csv")

    X_train = train_df.drop(columns=[dataset_cfg.target])
    y_train = train_df[dataset_cfg.target]
    X_test = test_df.drop(columns=[dataset_cfg.target])
    y_test = test_df[dataset_cfg.target]

    logger.info(f"Shape train={X_train.shape}, test={X_test.shape}")

    # 3) Préprocesseur commun
    preprocessor, numeric_features = build_preprocessor(train_df, dataset_cfg)
    logger.info(f"Features numériques utilisées : {numeric_features}")

    # 4) Modèle + CV
    model = LogisticRegressionModel(
        training_config=training_cfg,
        param_grid=logistic_cfg.get("grid", {}),
    )
    model.fit_with_cv(X_train, y_train, preprocessor)

    # 5) Évaluation sur le jeu de test
    y_pred = model.predict(X_test)

    out_dir = path_join("models", "artifacts", "logistic")
    os.makedirs(out_dir, exist_ok=True)

    # Sauvegarde du modèle
    save_artifact(model, os.path.join("models", "artifacts", "logistic", "model.joblib"))

    # Figures & rapports
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        title="Régression logistique – Breast Cancer (test)",
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

    logger.info("Entraînement terminé. Artefacts sauvegardés dans %s", out_dir)


if __name__ == "__main__":
    main()
