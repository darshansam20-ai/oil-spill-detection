"""
Patch Stitcher and Seamless Probability Reconstruction (PRD Section 4.4, FR-15).
Reconstructs full-scene probability maps from overlapping tiles using 2D Gaussian/Hann distance blending.
"""
from typing import List, Optional
import numpy as np
from src.tiling.tiler import PatchTile


def create_2d_blend_window(size: int, window_type: str = "gaussian") -> np.ndarray:
    """
    Generate a 2D weight window that tapers towards tile edges to ensure seamless blending.
    """
    if window_type == "hann":
        w1d = np.hanning(size)
    elif window_type == "gaussian":
        sigma = size / 4.0
        x = np.arange(size) - (size - 1) / 2.0
        w1d = np.exp(-0.5 * (x / sigma) ** 2)
    elif window_type == "linear":
        w1d = 1.0 - np.abs(np.linspace(-1, 1, size))
    else:  # Constant / Mean
        w1d = np.ones(size)

    w2d = np.outer(w1d, w1d)
    w2d = np.maximum(w2d, 1e-4)  # Prevent zero weights
    return w2d.astype(np.float32)


class PatchStitcher:
    """Reconstructs full-scene probability maps from overlapping tile predictions."""

    def __init__(self, patch_size: int = 256, blend_window: str = "gaussian"):
        self.patch_size = patch_size
        self.blend_window_type = blend_window
        self.weight_window = create_2d_blend_window(patch_size, blend_window)

    def stitch_patches(
        self,
        predicted_patches: List[np.ndarray],
        tiles: List[PatchTile],
        orig_h: int,
        orig_w: int,
        padded_h: int,
        padded_w: int,
    ) -> np.ndarray:
        """
        Merge overlapping predicted probability patches into a full-scene probability map.
        
        Args:
            predicted_patches: List of 2D numpy arrays (floats in [0.0, 1.0]), each (patch_size, patch_size)
            tiles: List of corresponding PatchTile coordinate metadata objects
            orig_h: Original unpadded scene height
            orig_w: Original unpadded scene width
            padded_h: Total padded canvas height
            padded_w: Total padded canvas width
            
        Returns:
            Full-scene reconstructed probability map of shape (orig_h, orig_w) with values in [0.0, 1.0].
        """
        accum_prob = np.zeros((padded_h, padded_w), dtype=np.float32)
        accum_weight = np.zeros((padded_h, padded_w), dtype=np.float32)

        for patch_pred, tile in zip(predicted_patches, tiles):
            y1, y2, x1, x2 = tile.y1, tile.y2, tile.x1, tile.x2
            # Handle possible batch dimension
            if patch_pred.ndim == 3 and patch_pred.shape[0] == 1:
                patch_pred = patch_pred[0]

            accum_prob[y1:y2, x1:x2] += patch_pred * self.weight_window
            accum_weight[y1:y2, x1:x2] += self.weight_window

        # Normalize by accumulated weights
        valid_weights = accum_weight > 0
        full_prob = np.zeros_like(accum_prob)
        full_prob[valid_weights] = accum_prob[valid_weights] / accum_weight[valid_weights]

        # Crop back to original scene dimensions
        cropped_prob = full_prob[:orig_h, :orig_w]
        return np.clip(cropped_prob, 0.0, 1.0)
