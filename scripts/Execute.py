"""
Sentinel-1 Oil Spill Detection Execution Script.
Supports single image execution or batch processing an entire folder.
"""
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.oil_spill_detector import Sentinel1OilSpillDetector


def run_single_image_example():
    """Demonstrates running detection on a single SAR image."""
    print("\n" + "=" * 75)
    print("           SINGLE IMAGE OIL SPILL DETECTION DEMO           ")
    print("=" * 75)

    # 1. Initialize detector
    detector = Sentinel1OilSpillDetector(threshold=0.50, min_spill_pixels=50)

    # 2. Specify input image path
    image_path = "data/extracted/test/images/2018_12_19_d.tif"

    # 3. Optional metadata (Set to None if no metadata is available)
    metadata = {
        "date": "2018-12-19",
        "time": "06:15:22 UTC",
        "aoi": [-89.50, 28.20, -88.70, 28.90],  # [min_lon, min_lat, max_lon, max_lat]
        "satellite": "Sentinel-1A",
    }

    # 4. Run detection
    result = detector.detect(
        image=image_path,
        metadata=metadata,
        output_image_path="data/outputs/single_spill_detection.png",
        save_json="data/outputs/single_spill_detection.json",
    )

    # 5. Print summary
    result.print_summary()

    # 6. Access detection attributes in Python
    print(f"[Result] Output Image: {result.output_image_path}")
    print(f"[Result] Spills Found: {result.spills_detected}")
    for s in result.spills[:5]:  # print first 5 spills
        print(f" -> Spill #{s.spill_id}: Conf={s.peak_confidence:.2f}, BBox(pixels)={s.bbox_pixel}")


def run_batch_folder_example():
    """Demonstrates processing all SAR images in a folder."""
    print("\n" + "=" * 75)
    print("           BATCH FOLDER OIL SPILL DETECTION DEMO           ")
    print("=" * 75)

    images_dir = Path("data/extracted/test/images")
    image_files = list(images_dir.glob("*.tif")) + list(images_dir.glob("*.png"))

    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images to process in {images_dir}.\n")

    # Initialize model once and reuse across images
    detector = Sentinel1OilSpillDetector(threshold=0.50, min_spill_pixels=50)

    out_dir = Path("data/outputs/batch_results")
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in image_files[:2]:  # Example: run first 2 images
        print(f"[*] Processing: {img_path.name} ...")
        out_image = out_dir / f"{img_path.stem}_detected.png"
        
        # Run detection (Image only, no metadata)
        res = detector.detect(
            image=img_path,
            metadata=None,
            output_image_path=out_image,
        )
        print(f"    -> Detected {res.spills_detected} spills | Saved: {res.output_image_path}")


if __name__ == "__main__":
    # Run the single image detection
    run_single_image_example()

    # Uncomment below if you want to batch process an entire directory:
    # run_batch_folder_example()
