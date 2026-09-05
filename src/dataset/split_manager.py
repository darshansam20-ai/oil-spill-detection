"""
Dataset Split Management (PRD Section 16).
Provides leakage-free, reproducible scene-level partitioning into Train, Validation, and Test sets.
"""
from typing import Dict, List, Tuple
from pathlib import Path
import random

from src.dataset.archive_handler import DatasetArchiveHandler
from src.utils.logger import get_logger

logger = get_logger("dataset.split_manager")


class SplitManager:
    """Manages scene-level train/validation/test partitions."""

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        self.archive_handler = DatasetArchiveHandler()

    def get_scene_splits(self) -> Dict[str, List[Tuple[Path, Path, str]]]:
        """
        Partition dataset into Train, Validation, and Test scene sets.
        Ensures zero spatial leakage by splitting strictly at the whole-scene level.
        
        Returns:
            Dict with keys 'train', 'val', 'test', each containing list of (img_path, mask_path, split_name).
        """
        all_pairs = self.archive_handler.discover_scene_pairs()
        train_raw = [p for p in all_pairs if p[2] == "train"]
        test_raw = [p for p in all_pairs if p[2] == "test"]

        # Deterministic shuffle for train/val split
        rng = random.Random(self.random_seed)
        shuffled_train = list(train_raw)
        rng.shuffle(shuffled_train)

        # 11 scenes for train (~80% of train set), 3 scenes for validation (~20% of train set)
        val_count = max(2, int(len(shuffled_train) * 0.20))
        val_pairs = shuffled_train[:val_count]
        train_pairs = shuffled_train[val_count:]
        test_pairs = test_raw

        logger.info(
            f"Scene Partitioning: Train={len(train_pairs)} scenes, "
            f"Val={len(val_pairs)} scenes, Test={len(test_pairs)} scenes."
        )

        return {
            "train": train_pairs,
            "val": val_pairs,
            "test": test_pairs,
        }
