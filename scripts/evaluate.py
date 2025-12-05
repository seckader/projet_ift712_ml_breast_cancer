"""
Evaluation script:
- Load trained pipelines from disk
- Evaluate them on the test set
- Save metrics, confusion matrix, classification report
"""

import joblib
import pandas as pd

from src.config.config import Config
from src.data.data_loader import get_features_and_target
from src.evaluation.metrics import compute_classification_metrics
from src.utils.io import save_csv, save_text
from src.utils.logging import get_logger
from src.utils.paths import (
    DATA_INTERIM_DIR,
    MODELS_ARTIFACTS_DIR,
    MODELS_REPORTS_DIR,
)
from src.utils.plotting import plot_confusion_matrix, save_classification_report

logger = get_logger(__name__)


def evaluate_model(config: Config, model_name: str, test_df: pd.DataFrame) -> None:
    """
    Evaluate a single trained model on the test set.
    """
    model_path = MODELS_ARTIFACTS_DIR / f"{model_name}_best.joblib"
    if not model_path.exists():
        logger.warning("Model %s not found at %s, skipping.", model_name, model_path)
        return

    logger.info("Evaluating model %s", model_name)

    model = joblib.load(model_path)

    X_test, y_test = get_features_and_target(config, test_df)
    y_pred = model.predict(X_test)

    metrics = compute_classification_metrics(y_test, y_pred)

    MODELS_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    txt_path = MODELS_REPORTS_DIR / f"{model_name}_test_metrics.txt"
    csv_path = MODELS_REPORTS_DIR / f"{model_name}_test_metrics.csv"

    save_text("\n".join([f"{k}: {v:.4f}" for k, v in metrics.items()]), txt_path)
    save_csv(pd.DataFrame([metrics]), csv_path)

    logger.info("Test metrics saved to %s and %s", txt_path, csv_path)

    conf_path = MODELS_REPORTS_DIR / f"{model_name}_confusion_matrix.png"
    plot_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        title=f"Confusion Matrix - {model_name}",
        out_path=str(conf_path),
    )
    logger.info("Confusion matrix saved to %s", conf_path)

    report_path = MODELS_REPORTS_DIR / f"{model_name}_classification_report.txt"
    target_names = [str(c) for c in sorted(set(y_test))]
    save_classification_report(
        y_true=y_test,
        y_pred=y_pred,
        target_names=target_names,
        out_path=str(report_path),
    )
    logger.info("Classification report saved to %s", report_path)


def main():
    config = Config()

    test_path = DATA_INTERIM_DIR / "test.csv"
    if not test_path.exists():
        raise FileNotFoundError("Run scripts.prepare_data first.")

    test_df = pd.read_csv(test_path)

    for model_name, model_cfg in config.models.models.items():
        if model_cfg.enabled:
            evaluate_model(config, model_name, test_df)


if __name__ == "__main__":
    main()
