from __future__ import annotations

import sys, os
sys.path.append(os.path.abspath("."))

from src.data.data_loader import load_dataset
from src.data.split import make_splits
from src.utils.io import save_csv
from src.utils.logging import get_logger

logger = get_logger("prepare")


def main():
    data_frame = load_dataset()
    logger.info(f"Dataset chargé avec shape={data_frame.shape}")

    # Sauvegarde d'une version "processed" complète
    save_csv(data_frame, "data/processed/full.csv")

    # Création des splits train / test
    make_splits(data_frame)
    logger.info("Fichiers data/interim/train.csv et data/interim/test.csv créés.")


if __name__ == "__main__":
    main()
