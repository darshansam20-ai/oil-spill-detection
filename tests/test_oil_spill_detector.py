"""
Unit and Integration Tests for Sentinel1OilSpillDetector.
Tests detection pipeline in both modes:
  1. Without metadata (pure image -> pixel-level bounding boxes)
  2. With metadata (metadata provided -> AOI, Lat/Lon, Date/Time resolved)
"""
import json
import pytest
import numpy as np
from pathlib import Path
from PIL import Image

from src.inference.oil_spill_detector import Sentinel1OilSpillDetector, DetectionResult


@pytest.fixture(scope="module")
def detector():
    return Sentinel1OilSpillDetector(threshold=0.50, min_spill_pixels=10)


@pytest.fixture
def synthetic_sar_image(tmp_path):
    """Create a temporary synthetic SAR image with a bright/dark oil-like patch."""
    img_array = (np.random.rand(256, 256) * 50).astype(np.uint8)
    # Simulate a low backscatter oil spill slick in center
    img_array[80:120, 80:120] = 5
    
    img_path = tmp_path / "synthetic_sar.png"
    Image.fromarray(img_array).save(img_path)
    return img_path


def test_detector_without_metadata(detector, synthetic_sar_image, tmp_path):
    out_img = tmp_path / "out_no_meta.png"
    out_json = tmp_path / "out_no_meta.json"

    result = detector.detect(
        image=synthetic_sar_image,
        metadata=None,
        output_image_path=out_img,
        save_json=out_json,
    )

    assert isinstance(result, DetectionResult)
    assert result.has_metadata is False
    assert result.aoi is None
    assert result.acquisition_date is None
    assert result.acquisition_time is None
    assert Path(result.output_image_path).exists()
    assert out_json.exists()

    with open(out_json, "r") as f:
        data = json.load(f)
    assert data["has_metadata"] is False
    assert "metadata" not in data


def test_detector_with_metadata(detector, synthetic_sar_image, tmp_path):
    out_img = tmp_path / "out_with_meta.png"
    out_json = tmp_path / "out_with_meta.json"

    meta_dict = {
        "date": "2026-09-04",
        "time": "14:30:00 UTC",
        "aoi": [-89.5, 28.2, -88.7, 28.9],
        "satellite": "Sentinel-1A",
    }

    result = detector.detect(
        image=synthetic_sar_image,
        metadata=meta_dict,
        output_image_path=out_img,
        save_json=out_json,
    )

    assert isinstance(result, DetectionResult)
    assert result.has_metadata is True
    assert result.acquisition_date == "2026-09-04"
    assert result.acquisition_time == "14:30:00 UTC"
    assert result.aoi is not None
    assert result.aoi["min_longitude"] == -89.5
    assert result.aoi["min_latitude"] == 28.2
    assert Path(result.output_image_path).exists()
    assert out_json.exists()

    with open(out_json, "r") as f:
        data = json.load(f)
    assert data["has_metadata"] is True
    assert data["metadata"]["acquisition_date"] == "2026-09-04"
    assert data["metadata"]["acquisition_time"] == "14:30:00 UTC"
    assert data["metadata"]["aoi"]["min_longitude"] == -89.5


def test_detector_with_metadata_json_file(detector, synthetic_sar_image, tmp_path):
    meta_path = tmp_path / "input_meta.json"
    meta_dict = {
        "datetime": "2020-02-24T18:00:00Z",
        "aoi": [-90.0, 27.5, -89.0, 28.5],
    }
    with open(meta_path, "w") as f:
        json.dump(meta_dict, f)

    result = detector.detect(
        image=synthetic_sar_image,
        metadata=meta_path,
        output_image_path=tmp_path / "out_json_file.png",
    )

    assert result.has_metadata is True
    assert result.acquisition_date == "2020-02-24"
    assert "18:00:00" in result.acquisition_time
