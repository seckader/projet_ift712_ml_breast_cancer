from sklearn.model_selection import StratifiedKFold


def build_cv_splits(cv_cfg: dict) -> StratifiedKFold:
    """
    Build a StratifiedKFold instance from the training config.
    """
    return StratifiedKFold(
        n_splits=cv_cfg["n_splits"],
        shuffle=cv_cfg.get("shuffle", True),
        random_state=cv_cfg.get("random_state", 42),
    )