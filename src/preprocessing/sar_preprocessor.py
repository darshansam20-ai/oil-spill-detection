"""
SAR Preprocessing Pipeline (PRD Section 4.3, FR-10 to FR-14).
Handles radiometric calibration (sigma-naught in dB), validated Refined Lee speckle filtering,
geocoding coordinate alignment, and model-consistent input normalization.
"""
from dataclasses import dataclass, asdict
from typing import Optional, Tuple
import numpy as np
from PIL import Image

from src.config.constants import DEFAULT_SIGMA0_MIN_DB, DEFAULT_SIGMA0_MAX_DB, DEFAULT_EPSILON
from src.preprocessing.speckle_filter import refined_lee_filter, lee_filter
from src.preprocessing.georeference import GeoreferenceTransform
from src.utils.logger import get_logger

logger = get_logger("preprocessing.sar_preprocessor")


@dataclass
class PreprocessingConfig:
    """Stores full preprocessing provenance to guarantee train/inference consistency."""
    sigma0_min_db: float = DEFAULT_SIGMA0_MIN_DB
    sigma0_max_db: float = DEFAULT_SIGMA0_MAX_DB
    speckle_filter_name: str = "refined_lee"  # 'refined_lee', 'lee', 'none'
    speckle_window_size: int = 7
    speckle_num_looks: float = 1.0
    apply_db_conversion: bool = True
    normalize_min: float = 0.0
    normalize_max: float = 1.0


class SARPreprocessor:
    """Executes the standardized SAR preprocessing workflow."""

    def __init__(self, config: Optional[PreprocessingConfig] = None):
        self.config = config or PreprocessingConfig()

    def linear_to_db(self, intensity: np.ndarray) -> np.ndarray:
        """
        Radiometric Calibration: Convert linear SAR intensity (sigma-0) to decibels (dB).
        Formula: sigma0_dB = 10 * log10(intensity + epsilon)
        """
        intensity = np.maximum(intensity, 0.0)
        return 10.0 * np.log10(intensity + DEFAULT_EPSILON)

    def apply_speckle_reduction(self, img: np.ndarray) -> np.ndarray:
        """
        Apply validated speckle noise reduction.
        """
        if self.config.speckle_filter_name == "refined_lee":
            return refined_lee_filter(
                img,
                size=self.config.speckle_window_size,
                num_looks=self.config.speckle_num_looks,
            )
        elif self.config.speckle_filter_name == "lee":
            return lee_filter(
                img,
                size=self.config.speckle_window_size,
                num_looks=self.config.speckle_num_looks,
            )
        else:
            return img

    def normalize(self, img_db: np.ndarray) -> np.ndarray:
        """
        Normalize SAR backscatter image to [0.0, 1.0] range based on configured dB clipping limits.
        Consistent normalization between training and production inference (FR-14).
        """
        clipped = np.clip(img_db, self.config.sigma0_min_db, self.config.sigma0_max_db)
        norm = (clipped - self.config.sigma0_min_db) / (self.config.sigma0_max_db - self.config.sigma0_min_db + 1e-8)
        return norm.astype(np.float32)

    def preprocess_image(self, img_array: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Execute full SAR preprocessing pipeline on an input raster.
        
        Args:
            img_array: Raw SAR image array (2D grayscale, linear backscatter or digital numbers).
            
        Returns:
            Tuple of:
              - normalized_tensor_input: np.ndarray (float32, [0, 1]) ready for AI model
              - preprocessed_db: np.ndarray (float32, dB scale) for visual inspection and reporting
        """
        img_float = img_array.astype(np.float32)

        # 1. Radiometric calibration to sigma0 in dB if needed
        if self.config.apply_db_conversion and img_float.max() > 1.0:
            # If values are raw digital numbers / linear amplitude, convert to dB
            img_db = self.linear_to_db(img_float)
        else:
            # Already in dB or normalized scale
            img_db = img_float

        # 2. Validated speckle noise reduction (FR-12)
        filtered_db = self.apply_speckle_reduction(img_db)

        # 3. Model-consistent input normalization (FR-14)
        normalized = self.normalize(filtered_db)

        return normalized, filtered_db

    def load_and_preprocess(self, file_path: str) -> Tuple[np.ndarray, np.ndarray, GeoreferenceTransform]:
        """
        Load SAR image from file (GeoTIFF, PNG, etc.) and run preprocessing.
        """
        logger.info(f"Loading and preprocessing SAR image from: {file_path}")
        
        # Try loading with rasterio for GeoTIFF metadata
        try:
            geo_transform = GeoreferenceTransform.from_geotiff(file_path)
            with Image.open(file_path) as pil_img:
                raw_data = np.array(pil_img)
        except Exception:
            # Fallback for standard image files without embedded GeoTIFF tags
            with Image.open(file_path) as pil_img:
                raw_data = np.array(pil_img)
            geo_transform = GeoreferenceTransform(width=raw_data.shape[1], height=raw_data.shape[0])

        if raw_data.ndim == 3:
            # Convert multi-channel/RGB to single-channel VV grayscale
            raw_data = raw_data[..., 0]

        normalized, preprocessed_db = self.preprocess_image(raw_data)
        return normalized, preprocessed_db, geo_transform
