"""
Dataset Statistics and Distribution Analysis (PRD Section 5).
Generates comprehensive statistical summaries of the oil spill dataset.
"""
from typing import Any, Dict, List
import pandas as pd
from src.dataset.archive_handler import DatasetArchiveHandler
from src.utils.logger import get_logger

logger = get_logger("dataset.statistics")


def compute_dataset_statistics() -> Dict[str, Any]:
    """
    Compute aggregate metrics across all scenes in the dataset.
    """
    handler = DatasetArchiveHandler()
    pairs = handler.discover_scene_pairs()
    
    scene_summaries = []
    total_pixels = 0
    total_oil_pixels = 0

    for img_path, mask_path, split in pairs:
        info = handler.validate_pair_integrity(img_path, mask_path)
        info["split"] = split
        scene_summaries.append(info)
        total_pixels += info["shape"][0] * info["shape"][1]
        total_oil_pixels += info["oil_pixels"]

    overall_oil_pct = (total_oil_pixels / total_pixels) * 100.0 if total_pixels > 0 else 0.0

    stats = {
        "total_scenes": len(pairs),
        "train_scenes": len([s for s in scene_summaries if s["split"] == "train"]),
        "test_scenes": len([s for s in scene_summaries if s["split"] == "test"]),
        "total_pixels": total_pixels,
        "total_oil_pixels": total_oil_pixels,
        "overall_oil_percentage": round(overall_oil_pct, 4),
        "scenes": scene_summaries,
    }

    logger.info(
        f"Dataset Summary: {stats['total_scenes']} scenes ({stats['train_scenes']} train, {stats['test_scenes']} test), "
        f"Overall Oil Percentage: {stats['overall_oil_percentage']}%"
    )
    return stats
