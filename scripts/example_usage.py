"""
Example Python usage script for Sentinel-1 Oil Spill Detection Model.
Demonstrates both modes:
  1. Without metadata (pure image input -> image output with bounding boxes)
  2. With metadata (metadata provided -> image output with bounding boxes + AOI, Lat/Lon, Date/Time)
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.oil_spill_detector import Sentinel1OilSpillDetector


def main():
    # 1. Initialize the Detector once
    detector = Sentinel1OilSpillDetector(threshold=0.50, min_spill_pixels=50)

    # Sample test image
    sample_image = "data/extracted/test/images/2018_12_19_d.tif"

    print("\n" + "=" * 80)
    print("DEMO 1: Running Model WITHOUT Metadata (Image Only -> Bounding Boxes)")
    print("=" * 80)
    
    result_no_meta = detector.detect(
        image=sample_image,
        metadata=None,  # No metadata supplied
        output_image_path="data/outputs/demo_without_metadata.png",
        save_json="data/outputs/demo_without_metadata.json",
    )
    result_no_meta.print_summary()

    print("\n" + "=" * 80)
    print("DEMO 2: Running Model WITH Metadata (AOI, Lat/Lon, Date/Time Enabled)")
    print("=" * 80)
    
    # Metadata can be a dictionary or a path to a JSON file
    sample_metadata = {
        "date": "2018-12-19",
        "time": "06:15:22 UTC",
        "aoi": [-89.50, 28.20, -88.70, 28.90],  # [min_lon, min_lat, max_lon, max_lat]
        "satellite": "Sentinel-1A",
    }

    result_with_meta = detector.detect(
        image=sample_image,
        metadata=sample_metadata,
        output_image_path="data/outputs/demo_with_metadata.png",
        save_json="data/outputs/demo_with_metadata.json",
    )
    result_with_meta.print_summary()

    print("\n[OK] Both demonstrations completed successfully.")
    print(f"Output (Without Metadata): {result_no_meta.output_image_path}")
    print(f"Output (With Metadata):    {result_with_meta.output_image_path}")


if __name__ == "__main__":
    main()
