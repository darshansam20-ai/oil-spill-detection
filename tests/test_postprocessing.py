"""
Unit tests for Mask Post-Processing and CCA (FR-18 to FR-20).
"""
import numpy as np
import pytest

from src.postprocessing.mask_processor import MaskPostProcessor


def test_postprocessor_binarization_and_morphology():
    processor = MaskPostProcessor(threshold=0.5, min_pixels=10, opening_radius=1, closing_radius=1)

    # Synthetic probability map: background ~0.1, spill blob ~0.9
    prob_map = np.ones((100, 100), dtype=np.float32) * 0.1
    # Add a 20x20 spill region
    prob_map[30:50, 30:50] = 0.95
    # Add 2 single-pixel noisy spikes
    prob_map[10, 10] = 0.99
    prob_map[80, 80] = 0.99

    final_mask, components = processor.process(prob_map)

    assert final_mask.shape == (100, 100)
    # The 2 noisy single-pixel spikes should be filtered out by opening/min_pixels
    assert final_mask[10, 10] == 0
    assert final_mask[80, 80] == 0
    # The main 20x20 spill should be retained as a valid component
    assert len(components) == 1
    comp = components[0]
    assert comp.pixel_area > 300
    assert comp.mean_confidence > 0.90
    assert len(comp.contours) > 0
