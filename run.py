"""
Integrated Sentinel-1 SAR Oil Spill Detection & AIS Maritime Vessel Correlation System.

Unified Entry Point:
Automatically executes the sequential pipeline:
  1. Sentinel-1 SAR Deep Learning Oil Spill Detection (ConvNeXt-Tiny + U-Net)
  2. Adapter Layer (Extracts Spill Epicenter, Formats ISO Timestamp, Configures AOI)
  3. AIS Maritime Vessel Proximity & Trajectory Correlation (Global Fishing Watch API & Haversine Tracking)
  4. Generates Visual Detection Image, Interactive HTML Vessel Map, Ranked CSV/JSON, and Full Incident Report.

Usage Examples:
    # 1. Standard Run with Satellite Image and Metadata JSON:
    python run.py --image data/extracted/test/images/2018_12_19_d.tif --metadata data/sample_metadata.json

    # 2. Image with explicit CLI metadata flags:
    python run.py --image data/extracted/test/images/2018_12_19_d.tif --date "2018-12-19" --time "06:15:22 UTC" --aoi "-89.5,28.2,-88.7,28.9"

    # 3. Pure Image input (auto-infers and correlates):
    python run.py --image data/extracted/test/images/2018_12_19_d.tif

    # 4. Open results automatically:
    python run.py --image data/extracted/test/images/2018_12_19_d.tif --metadata data/sample_metadata.json --show
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

from src.pipeline.end_to_end_pipeline import EndToEndPipeline


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Sentinel-1 SAR Oil Spill Detection & AIS Maritime Vessel Tracking Integrated Pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required Satellite Input
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
        help="Acquisition Date (e.g. '2018-12-19' or '2026-09-04')",
    )
    parser.add_argument(
        "--time",
        type=str,
        default=None,
        help="Acquisition Time (e.g. '06:15:22 UTC')",
    )
    parser.add_argument(
        "--aoi",
        type=str,
        default=None,
        help="Area of Interest Bounding Box as 'min_lon,min_lat,max_lon,max_lat'",
    )

    # Spatial Coordinate Overrides
    parser.add_argument(
        "--lat",
        type=float,
        default=None,
        help="Override spill epicenter latitude for AIS search",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=None,
        help="Override spill epicenter longitude for AIS search",
    )

    # Hyperparameters
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.50,
        help="Oil spill detection probability threshold (0.0 to 1.0)",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=50,
        help="Minimum connected spill area in pixels to filter noise",
    )
    parser.add_argument(
        "--radius", "-r",
        type=float,
        default=20.0,
        help="AIS vessel search radius in kilometers around spill epicenter",
    )
    parser.add_argument(
        "--spill-id",
        type=int,
        default=None,
        help="Specific spill ID to correlate with AIS (defaults to highest confidence spill)",
    )

    # Paths & Environment
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="output",
        help="Directory to save all output artifacts",
    )
    parser.add_argument(
        "--checkpoint", "-c",
        type=str,
        default=None,
        help="Custom path to trained PyTorch oil spill model checkpoint (.pt)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Global Fishing Watch (GFW) API JWT token",
    )
    parser.add_argument(
        "--show", "-s",
        action="store_true",
        help="Automatically open the interactive AIS map and annotated detection image upon completion",
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    # 1. Parse Metadata Input
    metadata_dict = {}
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

    # Overlay CLI flags
    if args.date:
        metadata_dict["date"] = args.date
    if args.time:
        metadata_dict["time"] = args.time
    if args.aoi:
        try:
            parts = [float(x.strip()) for x in args.aoi.split(",")]
            if len(parts) == 4:
                metadata_dict["aoi"] = parts
        except ValueError:
            print("[Warning] Failed to parse --aoi numbers.")

    final_metadata = metadata_dict if metadata_dict else None

    # 2. Initialize End-to-End Pipeline
    print("\n" + "=" * 80)
    print("      INITIALIZING INTEGRATED OIL SPILL DETECTION & AIS CORRELATION SYSTEM      ")
    print("=" * 80)
    pipeline = EndToEndPipeline(
        checkpoint_path=args.checkpoint,
        detection_threshold=args.threshold,
        min_spill_pixels=args.min_pixels,
        ais_token=args.token,
        ais_search_radius_km=args.radius,
    )

    # 3. Execute Pipeline
    print(f"\n[*] Processing Sentinel-1 SAR Image: {args.image}")
    report = pipeline.run(
        image_path=args.image,
        metadata=final_metadata,
        output_dir=args.output_dir,
        spill_id_to_correlate=args.spill_id,
        search_radius_km=args.radius,
        override_lat=args.lat,
        override_lon=args.lon,
        override_time=args.date if args.date and not args.time else None,
    )

    # 4. Display Comprehensive Console Summary
    report.print_comprehensive_summary()

    # 5. Open Artifacts if requested
    if args.show:
        if report.interactive_map_path and os.path.exists(report.interactive_map_path):
            try:
                os.system(f'start "" "{Path(report.interactive_map_path).resolve()}"')
            except Exception:
                pass
        if report.annotated_image_path and os.path.exists(report.annotated_image_path):
            try:
                os.system(f'start "" "{Path(report.annotated_image_path).resolve()}"')
            except Exception:
                pass


if __name__ == "__main__":
    main()
