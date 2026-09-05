"""
Unit tests for Large-Scene Tiling and Stitching (FR-15).
"""
import numpy as np
import pytest

from src.tiling.tiler import ImageTiler
from src.tiling.stitcher import PatchStitcher, create_2d_blend_window


def test_tiler_and_stitcher_reconstruction():
    # Synthetic scene 600x700 pixels
    H, W = 600, 700
    scene = np.random.uniform(0.1, 0.9, (H, W)).astype(np.float32)

    patch_size = 256
    overlap = 64
    tiler = ImageTiler(patch_size=patch_size, overlap=overlap)
    stitcher = PatchStitcher(patch_size=patch_size, blend_window="gaussian")

    padded_h, padded_w, _, _ = tiler.get_patch_grid_dimensions(H, W)
    tiles = list(tiler.extract_patches(scene))
    
    assert len(tiles) > 0
    for tile in tiles:
        assert tile.patch.shape == (patch_size, patch_size)

    # Simulate identity predictor (prediction == input patch)
    predicted_patches = [t.patch for t in tiles]

    reconstructed = stitcher.stitch_patches(
        predicted_patches=predicted_patches,
        tiles=tiles,
        orig_h=H,
        orig_w=W,
        padded_h=padded_h,
        padded_w=padded_w,
    )

    assert reconstructed.shape == (H, W)
    # The reconstructed image should be virtually identical to original scene (smooth blending)
    mae = np.mean(np.abs(reconstructed - scene))
    assert mae < 0.05, f"Reconstruction error too high: MAE={mae}"


def test_2d_blend_window_properties():
    size = 128
    win = create_2d_blend_window(size=size, window_type="gaussian")
    assert win.shape == (size, size)
    assert win[size // 2, size // 2] > win[0, 0]  # Center weight > corner weight
    assert win.min() > 0.0
