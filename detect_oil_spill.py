"""
Sentinel-1 SAR Oil Spill Detection Command-Line Tool.

Usage Examples:
    # 1. Pure image input (no metadata -> produces image with bounding boxes only):
    python detect_oil_spill.py --image data/extracted/test/images/2018_12_19_d.tif

    # 2. Image with metadata JSON file:
    python detect_oil_spill.py --image data/extracted/test/images/2018_12_19_d.tif --metadata data/sample_metadata.json

    # 3. Image with metadata provided directly via CLI flags:
    python detect_oil_spill.py --image data/extracted/test/images/2018_12_19_d.tif --date "2018-12-19" --time "06:15:22 UTC" --aoi "-89.5,28.2,-88.7,28.9" --output output.png --save-json result.json
"""
import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.oil_spill_detector import Sentinel1OilSpillDetector


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sentinel-1 SAR Oil Spill Detection & Bounding Box Extraction Tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Required Input
    parser.add_argument(
        "--image", "-i",
        type=str,
        required=True,
        help="Path to Sentinel-1 SAR input image (.tif, .tiff, .png, .jpg, etc.)",
    )
    
    # Optional Metadata Inputs
    parser.add_argument(
        "--metadata", "-m",
        type=str,
        default=None,
        help="Path to metadata JSON file or raw JSON string with AOI, Date, Time, etc.",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Acquisition Date (e.g. '2026-09-04' or '2018-12-19')",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="Acquisition Time (e.g. '14:30:00 UTC')",
    )
    parser.add_argument(
        "--aoi",
        type=str,
        default=None,
        help="Area of Interest Bounding Box as 'min_lon,min_lat,max_lon,max_lat' (e.g. '-90.5,28.0,-89.5,29.0')",
    )
    
    # Model Hyperparameters
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.50,
        help="Oil spill probability threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=50,
        help="Minimum connected spill area in pixels to filter noise",
    )
    
    # Outputs
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="File path to save the annotated output image (.png, .jpg)",
    )
    parser.add_argument(
        "--save-json", "-j",
        type=str,
        default=None,
        help="File path to save detection details and metadata as JSON",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Custom path to trained PyTorch model checkpoint (.pt)",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Open the output annotated image automatically after processing",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Prepare Metadata Dictionary if provided
    metadata_dict = None
    if args.metadata:
        meta_str_or_path = args.metadata.strip()
        if meta_str_or_path.startswith("{"):
            try:
                metadata_dict = json.loads(meta_str_or_path)
            except Exception as e:
                print(f"[Warning] Failed to parse JSON string: {e}")
        else:
            meta_path = Path(meta_str_or_path)
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        metadata_dict = json.load(f)
                except Exception as e:
                    print(f"[Warning] Failed to read metadata file '{meta_path}': {e}")
            else:
                print(f"[Warning] Metadata file not found: {meta_path}")

    if metadata_dict is None:
        metadata_dict = {}

    # Overlay CLI metadata flags if provided
    if args.date:
        metadata_dict["date"] = args.date
    if args.time:
        metadata_dict["time"] = args.time
    if args.aoi:
        try:
            parts = [float(x.strip()) for x in args.aoi.split(",")]
            if len(parts) == 4:
                metadata_dict["aoi"] = parts
            else:
                print("[Warning] --aoi flag must contain exactly 4 comma-separated numbers (min_lon,min_lat,max_lon,max_lat)")
        except ValueError:
            print("[Warning] Failed to parse --aoi numbers.")

    # If dict is completely empty and no flags passed, set metadata to None
    final_metadata = metadata_dict if metadata_dict else None

    # 2. Initialize Model
    print("\n[*] Initializing Sentinel-1 Oil Spill Detection Model...")
    detector = Sentinel1OilSpillDetector(
        checkpoint_path=args.checkpoint,
        threshold=args.threshold,
        min_spill_pixels=args.min_pixels,
    )

    # 3. Run Detection
    print(f"[*] Processing Image: {args.image}")
    result = detector.detect(
        image=args.image,
        metadata=final_metadata,
        output_image_path=args.output,
        threshold=args.threshold,
        min_spill_pixels=args.min_pixels,
        save_json=args.save_json,
    )

    # 4. Print Summary Report
    result.print_summary()

    # 5. Open image if requested
    if args.show and result.output_image_path and os.path.exists(result.output_image_path):
        try:
            os.system(f'start "" "{Path(result.output_image_path).resolve()}"')
        except Exception:
            pass


if __name__ == "__main__":
    main()
