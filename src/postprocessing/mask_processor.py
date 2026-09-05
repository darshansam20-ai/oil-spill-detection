"""
Mask Post-Processing Module (PRD Section 4.4, FR-18 to FR-20).
Applies configurable thresholding, morphological opening/closing,
connected-component analysis (CCA 8-connectivity), noise removal, and component labeling.
"""
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
from scipy import ndimage
from skimage.morphology import disk, opening, closing
import cv2

from src.config.constants import (
    DEFAULT_PROBABILITY_THRESHOLD,
    DEFAULT_MIN_SPILL_PIXELS,
    DEFAULT_OPENING_RADIUS,
    DEFAULT_CLOSING_RADIUS,
)
from src.utils.logger import get_logger

logger = get_logger("postprocessing.mask_processor")


@dataclass
class SpillComponent:
    """Represents an individual connected oil-spill region extracted from the mask."""
    component_id: int
    mask: np.ndarray  # 2D boolean mask for this component
    pixel_area: int
    mean_confidence: float
    peak_confidence: float
    centroid_pixel: Tuple[float, float]  # (row, col)
    bbox_pixel: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    contours: List[np.ndarray]  # OpenCV contour points [(col, row), ...]


class MaskPostProcessor:
    """Executes post-processing pipeline on raw probability maps."""

    def __init__(
        self,
        threshold: float = DEFAULT_PROBABILITY_THRESHOLD,
        min_pixels: int = DEFAULT_MIN_SPILL_PIXELS,
        opening_radius: int = DEFAULT_OPENING_RADIUS,
        closing_radius: int = DEFAULT_CLOSING_RADIUS,
    ):
        self.threshold = threshold
        self.min_pixels = min_pixels
        self.opening_radius = opening_radius
        self.closing_radius = closing_radius

    def binarize(self, prob_map: np.ndarray) -> np.ndarray:
        """Apply configurable probability threshold -> binary mask."""
        return (prob_map >= self.threshold).astype(np.uint8)

    def apply_morphology(self, binary_mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological opening (remove noise) and closing (fill holes).
        """
        mask_bool = binary_mask.astype(bool)
        if self.opening_radius > 0:
            mask_bool = opening(mask_bool, disk(self.opening_radius))
        if self.closing_radius > 0:
            mask_bool = closing(mask_bool, disk(self.closing_radius))
        return mask_bool.astype(np.uint8)

    def extract_connected_components(
        self,
        cleaned_mask: np.ndarray,
        prob_map: np.ndarray,
    ) -> List[SpillComponent]:
        """
        Perform 8-Connected Component Analysis and extract region properties.
        Filters out regions with pixel area < min_pixels.
        """
        # 8-connectivity structure
        struct = ndimage.generate_binary_structure(2, 2)
        labeled_mask, num_features = ndimage.label(cleaned_mask, structure=struct)
        logger.info(f"Connected component analysis detected {num_features} initial candidate regions.")

        components: List[SpillComponent] = []
        for label_idx in range(1, num_features + 1):
            comp_mask = (labeled_mask == label_idx)
            pixel_count = int(np.sum(comp_mask))

            # Noise / Minimum size filtering
            if pixel_count < self.min_pixels:
                continue

            # Confidence statistics within component
            comp_probs = prob_map[comp_mask]
            mean_conf = float(np.mean(comp_probs))
            peak_conf = float(np.max(comp_probs))

            # Centroid (row, col)
            row_indices, col_indices = np.where(comp_mask)
            centroid_r = float(np.mean(row_indices))
            centroid_c = float(np.mean(col_indices))

            # Bounding box in pixels: (min_r, min_c, max_r, max_c)
            bbox = (
                int(np.min(row_indices)),
                int(np.min(col_indices)),
                int(np.max(row_indices)),
                int(np.max(col_indices)),
            )

            # Vector contour extraction using OpenCV
            comp_uint8 = comp_mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(comp_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            components.append(
                SpillComponent(
                    component_id=label_idx,
                    mask=comp_mask,
                    pixel_area=pixel_count,
                    mean_confidence=mean_conf,
                    peak_confidence=peak_conf,
                    centroid_pixel=(centroid_r, centroid_c),
                    bbox_pixel=bbox,
                    contours=[c.squeeze(1) for c in contours if len(c) >= 3],
                )
            )

        logger.info(f"Post-processing retained {len(components)} spill regions after noise filtering (min_pixels={self.min_pixels}).")
        return components

    def process(self, prob_map: np.ndarray) -> Tuple[np.ndarray, List[SpillComponent]]:
        """
        Execute full post-processing pipeline:
        1. Binarize at threshold
        2. Morphological opening & closing
        3. Connected component analysis & noise filtering
        4. Contour extraction
        
        Returns:
            Tuple of:
              - final_binary_mask: np.ndarray (uint8, 0 or 1)
              - components: List of SpillComponent objects
        """
        binary = self.binarize(prob_map)
        cleaned = self.apply_morphology(binary)
        components = self.extract_connected_components(cleaned, prob_map)

        # Reconstruct cleaned binary mask containing only valid components
        final_mask = np.zeros_like(cleaned, dtype=np.uint8)
        for comp in components:
            final_mask[comp.mask] = 1

        return final_mask, components
