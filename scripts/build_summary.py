from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath("."))

import re
from pathlib import Path
import pandas as pd

from src.utils.logging import get_logger
from src.utils.paths import MODELS_REPORTS_DIR

logger = get_logger(__name__)


def _infer_model_name(path: Path) -> str:
    return re.sub(r"_(cv_results|test_metrics)\.csv$", "", path.name)


def load_cv_best_scores(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(reports_dir.glob("*_cv_results.csv")):
        model = _infer_model_name(p)
        df = pd.read_csv(p)

        if "rank_test_score" in df.columns:
            best_row = df.loc[df["rank_test_score"].idxmin()]
        elif "mean_test_score" in df.columns:
            best_row = df.loc[df["mean_test_score"].idxmax()]
        else:
            logger.warning("CV file %s has no score columns, skipping.", p.name)
            continue

        rows.append({
            "model": model,
            "cv_best_score": float(best_row.get("mean_test_score", float("nan"))),
            "cv_best_params": best_row.get("params", None),
        })

    return pd.DataFrame(rows)


def load_test_metrics(reports_dir: Path) -> pd.DataFrame:
    rows = []
    for p in sorted(reports_dir.glob("*_test_metrics.csv")):
        model = _infer_model_name(p)
        df = pd.read_csv(p)
        if df.empty:
            continue
        row = df.iloc[0].to_dict()
        row["model"] = model
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    reports_dir = Path(MODELS_REPORTS_DIR)
    reports_dir.mkdir(parents=True, exist_ok=True)

    cv_df = load_cv_best_scores(reports_dir)
    test_df = load_test_metrics(reports_dir)

    if cv_df.empty and test_df.empty:
        raise RuntimeError("No per-model reports found. Run `make train` and `make evaluate` first.")

    summary = pd.merge(test_df, cv_df, on="model", how="outer")

    preferred = ["model", "accuracy", "f1_macro", "precision_macro", "recall_macro", "roc_auc",
                 "cv_best_score", "cv_best_params"]
    cols = [c for c in preferred if c in summary.columns] + [c for c in summary.columns if c not in preferred]
    summary = summary[cols].sort_values("model")

    out_path = reports_dir / "summary.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Summary written to %s", out_path)


if __name__ == "__main__":
    main()
