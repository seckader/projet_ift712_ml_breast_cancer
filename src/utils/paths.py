from pathlib import Path

# Root of the project: .../projet_ift712_ml_breast_cancer
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuration directory (YAML files)
CONFIG_DIR = PROJECT_ROOT / "configs"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_RAW_DIR = DATA_DIR / "raw"
DATA_INTERIM_DIR = DATA_DIR / "interim"
DATA_PROCESSED_DIR = DATA_DIR / "processed"

# Models directories
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_ARTIFACTS_DIR = MODELS_DIR / "artifacts"
MODELS_REPORTS_DIR = MODELS_DIR / "reports"

# Notebooks directory (optional)
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


def project_root() -> str:
    return str(PROJECT_ROOT)


def path_join(*parts) -> str:
    return str(PROJECT_ROOT.joinpath(*parts))