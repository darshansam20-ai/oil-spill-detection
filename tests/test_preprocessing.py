"""
Unit tests for SAR Preprocessing and Speckle Filtering (FR-10 to FR-14).
"""
import numpy as np
import pytest

from src.preprocessing.sar_preprocessor import SARPreprocessor, PreprocessingConfig
from src.preprocessing.speckle_filter import refined_lee_filter, lee_filter
from src.preprocessing.georeference import GeoreferenceTransform


def test_linear_to_db_calibration():
    preprocessor = SARPreprocessor()
    linear_val = np.array([[1.0, 10.0], [100.0, 0.0]], dtype=np.float32)
    db_val = preprocessor.linear_to_db(linear_val)

    assert np.isclose(db_val[0, 0], 0.0, atol=0.1)     # 10 * log10(1) = 0 dB
    assert np.isclose(db_val[0, 1], 10.0, atol=0.1)    # 10 * log10(10) = 10 dB
    assert np.isclose(db_val[1, 0], 20.0, atol=0.1)    # 10 * log10(100) = 20 dB


def test_speckle_filters_output_shape_and_range():
    # Synthetic SAR patch with speckle noise
    np.random.seed(42)
    clean_signal = np.ones((64, 64), dtype=np.float32) * (-15.0)  # -15 dB sea surface
    # Add dark oil slick patch in center
    clean_signal[20:44, 20:44] = -28.0  # -28 dB dark slick
    noise = np.random.normal(0.0, 3.0, (64, 64)).astype(np.float32)
    noisy_sar = clean_signal + noise

    filtered_lee = lee_filter(noisy_sar, size=7, num_looks=1.0)
    filtered_ref_lee = refined_lee_filter(noisy_sar, size=7, num_looks=1.0)

    assert filtered_lee.shape == (64, 64)
    assert filtered_ref_lee.shape == (64, 64)
    assert np.std(filtered_ref_lee) < np.std(noisy_sar)  # Reduced variance


def test_input_normalization_range():
    config = PreprocessingConfig(sigma0_min_db=-30.0, sigma0_max_db=0.0)
    preprocessor = SARPreprocessor(config)

    db_img = np.array([[-35.0, -15.0], [5.0, -30.0]], dtype=np.float32)
    norm = preprocessor.normalize(db_img)

    assert norm.min() >= 0.0
    assert norm.max() <= 1.0
    assert norm[0, 0] == 0.0   # Clipped at -30 dB
    assert norm[1, 0] == 1.0   # Clipped at 0 dB
    assert np.isclose(norm[0, 1], 0.5, atol=1e-3)  # -15 dB is halfway between -30 and 0


def test_georeference_transform_roundtrip():
    geo = GeoreferenceTransform(bounds=[-91.0, 27.0, -90.0, 28.0], width=1000, height=1000)
    
    # Center pixel (500, 500)
    lon, lat = geo.pixel_to_geo(500, 500)
    assert -91.0 < lon < -90.0
    assert 27.0 < lat < 28.0

    r, c = geo.geo_to_pixel(lon, lat)
    assert abs(r - 500) <= 1
    assert abs(c - 500) <= 1
