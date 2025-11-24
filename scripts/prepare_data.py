# scripts/prepare_data.py
from __future__ import annotations
from src.data.data_loader import load_dataset
from src.data.split import make_splits
from src.utils.io import save_csv
from src.config.config import load_dataset_config
from src.utils.logging import get_logger

logger = get_logger("prepare")


def main():
    df = load_dataset()
    logger.info(f"Dataset chargé avec shape={df.shape}")

    # Sauvegarde d'une version "processed" complète
    save_csv(df, "data/processed/full.csv")

    # Création des splits train / test
    make_splits(df)
    logger.info("Fichiers data/interim/train.csv et data/interim/test.csv créés.")


if __name__ == "__main__":
    main()
