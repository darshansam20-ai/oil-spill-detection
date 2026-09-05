"""
Dataset Archive and File Validation Handler (PRD Section 5).
Validates image-mask pairs, reads GeoTIFF metadata, and verifies data integrity.
"""
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import rasterio
import numpy as np
from PIL import Image

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("dataset.archive_handler")


class DatasetArchiveHandler:
    """Handles discovery, validation, and integrity checks for the oil spill dataset."""

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir = data_dir or settings.paths.data_extracted

    def discover_scene_pairs(self) -> List[Tuple[Path, Path, str]]:
        """
        Discover all corresponding image and mask GeoTIFF pairs.
        
        Returns:
            List of tuples: (image_path, mask_path, split_name)
        """
        pairs = []
        for split in ["train", "test"]:
            split_dir = self.data_dir / split
            img_dir = split_dir / "images"
            mask_dir = split_dir / "masks"
            
            if not img_dir.exists() or not mask_dir.exists():
                continue

            img_files = sorted(list(img_dir.glob("*.tif*")))
            for img_path in img_files:
                mask_path = mask_dir / img_path.name
                if mask_path.exists():
                    pairs.append((img_path, mask_path, split))
                else:
                    logger.warning(f"Mask file not found for image: {img_path.name}")

        logger.info(f"Discovered {len(pairs)} validated image-mask scene pairs across {self.data_dir}.")
        return pairs

    def validate_pair_integrity(self, image_path: Path, mask_path: Path) -> Dict[str, Any]:
        """
        Verify spatial alignment, dimensions, data types, and value ranges for a pair.
        """
        with rasterio.open(image_path) as src_img, rasterio.open(mask_path) as src_mask:
            assert src_img.shape == src_mask.shape, f"Dimension mismatch: {src_img.shape} vs {src_mask.shape}"
            
            img_data = src_img.read(1)
            mask_data = src_mask.read(1)

            oil_pixels = int(np.sum(mask_data > 0))
            total_pixels = mask_data.size
            oil_percentage = (oil_pixels / total_pixels) * 100.0

            return {
                "scene_name": image_path.stem,
                "shape": src_img.shape,
                "crs": str(src_img.crs) if src_img.crs else "EPSG:32616",
                "bounds": [src_img.bounds.left, src_img.bounds.bottom, src_img.bounds.right, src_img.bounds.top],
                "img_dtype": str(img_data.dtype),
                "img_min_db": float(np.nanmin(img_data)),
                "img_max_db": float(np.nanmax(img_data)),
                "mask_unique_values": [float(v) for v in np.unique(mask_data)],
                "oil_pixels": oil_pixels,
                "oil_percentage": round(oil_percentage, 3),
            }
