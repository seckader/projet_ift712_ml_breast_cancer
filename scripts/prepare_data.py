"""
Data preparation script:
- Load the raw breast cancer dataset (from sklearn)
- Create train/test splits
"""

from src.config.config import Config
from src.data.data_loader import load_raw_dataset
from src.data.split import create_train_test_split
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    """
    Main entry point for the prepare_data script.
    """
    config = Config()

    logger.info("Preparing data for dataset '%s'.", config.dataset.name)

    df_raw = load_raw_dataset(config)
    create_train_test_split(config, df_raw)

    logger.info("Data preparation completed successfully.")


if __name__ == "__main__":
    main()
