"""
Large-Scene Tiling and Sliding-Window Extraction (PRD Section 4.4, FR-15).
Extracts overlapping patches from large Sentinel-1 SAR imagery with configurable patch size and overlap.
"""
from dataclasses import dataclass
from typing import Generator, List, Tuple
import numpy as np


@dataclass
class PatchTile:
    """Represents a single extracted tile with coordinate metadata."""
    patch: np.ndarray
    y1: int
    y2: int
    x1: int
    x2: int
    orig_h: int
    orig_w: int
    is_padded: bool


class ImageTiler:
    """Sliding-window patch extractor for large SAR scenes."""

    def __init__(self, patch_size: int = 256, overlap: int = 64):
        """
        Args:
            patch_size: Square tile dimension (e.g., 256 or 512).
            overlap: Overlap in pixels between adjacent patches.
        """
        assert patch_size > overlap, f"Patch size ({patch_size}) must be greater than overlap ({overlap})"
        self.patch_size = patch_size
        self.overlap = overlap
        self.stride = patch_size - overlap

    def extract_patches(self, img: np.ndarray) -> Generator[PatchTile, None, None]:
        """
        Generate sliding-window patches covering the entire image.
        Pads the image if dimensions are not exact multiples of stride.
        """
        H, W = img.shape[:2]
        
        # Calculate padding needed to cover the entire image
        pad_h = (self.stride - ((H - self.patch_size) % self.stride)) % self.stride if H > self.patch_size else max(0, self.patch_size - H)
        pad_w = (self.stride - ((W - self.patch_size) % self.stride)) % self.stride if W > self.patch_size else max(0, self.patch_size - W)
        
        padded_img = np.pad(img, ((0, pad_h), (0, pad_w)), mode="reflect") if (pad_h > 0 or pad_w > 0) else img
        padded_h, padded_w = padded_img.shape[:2]

        y_steps = range(0, padded_h - self.patch_size + 1, self.stride)
        x_steps = range(0, padded_w - self.patch_size + 1, self.stride)

        for y in y_steps:
            for x in x_steps:
                y2 = y + self.patch_size
                x2 = x + self.patch_size
                patch = padded_img[y:y2, x:x2]
                yield PatchTile(
                    patch=patch,
                    y1=y,
                    y2=y2,
                    x1=x,
                    x2=x2,
                    orig_h=H,
                    orig_w=W,
                    is_padded=(pad_h > 0 or pad_w > 0),
                )

    def get_patch_grid_dimensions(self, H: int, W: int) -> Tuple[int, int, int, int]:
        """
        Compute total padded dimensions and number of patch steps.
        """
        pad_h = (self.stride - ((H - self.patch_size) % self.stride)) % self.stride if H > self.patch_size else max(0, self.patch_size - H)
        pad_w = (self.stride - ((W - self.patch_size) % self.stride)) % self.stride if W > self.patch_size else max(0, self.patch_size - W)
        padded_h = H + pad_h
        padded_w = W + pad_w
        num_y = len(range(0, padded_h - self.patch_size + 1, self.stride))
        num_x = len(range(0, padded_w - self.patch_size + 1, self.stride))
        return padded_h, padded_w, num_y, num_x
