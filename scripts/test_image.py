"""
Manual Model Testing & Visual Verification Script.
Allows running inference on any arbitrary input image (GeoTIFF, TIFF, PNG, JPG)
and generates a high-resolution visual comparison and detection summary.

Usage:
    python scripts/test_image.py --image data/extracted/test/images/2018_09_26.tif
    python scripts/test_image.py --image data/extracted/test/images/2018_12_19_d.tif --threshold 0.50
    python scripts/test_image.py --image path/to/my_image.png --output data/outputs/my_result.png
"""
import argparse
import sys
import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")  # Safe backend for headless saving
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.inference.predictor import OilSpillPredictor
from src.preprocessing.sar_preprocessor import SARPreprocessor
from src.postprocessing.mask_processor import MaskPostProcessor
from src.utils.logger import get_logger

logger = get_logger("scripts.test_image")


def run_manual_test(
    image_path: str,
    threshold: float = 0.50,
    min_pixels: int = 50,
    output_plot_path: str = None,
    checkpoint_path: str = None,
    show_window: bool = False,
):
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    print("\n" + "=" * 75)
    print("        [SAR OIL SPILL DETECTION] MANUAL MODEL TESTER        ")
    print("=" * 75)
    print(f" [*] Target Image:        {img_file.resolve()}")
    print(f" [*] Detection Threshold: {threshold:.2f}")
    print(f" [*] Min Component Size:  {min_pixels} pixels")
    if checkpoint_path:
        print(f" [*] Custom Checkpoint:   {checkpoint_path}")

    # 1. Initialize Modules
    start_time = time.time()
    preprocessor = SARPreprocessor()
    predictor = OilSpillPredictor(checkpoint_path=Path(checkpoint_path) if checkpoint_path else None)
    postprocessor = MaskPostProcessor(threshold=threshold, min_pixels=min_pixels)

    # 2. Load & Preprocess
    print("\n[Step 1/3] Loading and preprocessing SAR image...")
    t0 = time.time()
    norm_img, db_img, geo_transform = preprocessor.load_and_preprocess(str(img_file))
    H, W = norm_img.shape[:2]
    print(f"           - Image Resolution: {W} x {H} pixels")
    print(f"           - Preprocessing time: {time.time() - t0:.2f}s")

    # 3. Model Inference
    print("\n[Step 2/3] Running neural network sliding-window inference...")
    t1 = time.time()
    prob_map = predictor.predict_scene(norm_img)
    inf_time = time.time() - t1
    print(f"           - Inference time: {inf_time:.2f}s")
    print(f"           - Probability range: min={prob_map.min():.4f}, max={prob_map.max():.4f}, mean={prob_map.mean():.4f}")

    # 4. Post-processing & Connected Components
    print("\n[Step 3/3] Post-processing probability map and extracting spill contours...")
    binary_mask, components = postprocessor.process(prob_map)
    total_spill_pixels = int(np.sum(binary_mask))
    pixel_spacing_m = geo_transform.pixel_width if hasattr(geo_transform, "pixel_width") and geo_transform.pixel_width else 10.0
    pixel_area_km2 = (pixel_spacing_m * pixel_spacing_m) / 1e6
    total_area_km2 = total_spill_pixels * pixel_area_km2

    print(f"           - Total Spill Pixels: {total_spill_pixels:,} (approx {total_area_km2:.3f} sq km)")
    print(f"           - Detected Spill Count: {len(components)}")

    # Print Detailed Component Summary Table
    if components:
        print("\n" + "-" * 90)
        print(f"{'ID':<4} | {'Area (px)':<10} | {'Area (km2)':<12} | {'Mean Conf':<10} | {'Peak Conf':<10} | {'Centroid (Row,Col)':<20} | {'Bounding Box (r0,c0,r1,c1)':<24}")
        print("-" * 90)
        for c in components:
            comp_area_km2 = c.pixel_area * pixel_area_km2
            centroid_str = f"({c.centroid_pixel[0]:.1f}, {c.centroid_pixel[1]:.1f})"
            bbox_str = f"({c.bbox_pixel[0]}, {c.bbox_pixel[1]}, {c.bbox_pixel[2]}, {c.bbox_pixel[3]})"
            print(f"{c.component_id:<4} | {c.pixel_area:<10} | {comp_area_km2:<12.4f} | {c.mean_confidence:<10.3f} | {c.peak_confidence:<10.3f} | {centroid_str:<20} | {bbox_str:<24}")
        print("-" * 90)
    else:
        print("\n [*] Clean Scene: No oil spills detected above the threshold.")

    # 5. Generate Visual Diagnostic Plot
    print("\n[*] Generating high-resolution 4-panel diagnostic visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 14), dpi=120)
    fig.patch.set_facecolor('#0f172a')  # Modern dark theme

    # Panel 1: Normalized SAR Backscatter Input
    ax1 = axes[0, 0]
    ax1.set_facecolor('#0f172a')
    im1 = ax1.imshow(norm_img, cmap='gray', aspect='auto')
    ax1.set_title(f"1. Input SAR Backscatter ({W}x{H})", color='white', fontsize=12, fontweight='bold', pad=10)
    ax1.tick_params(colors='gray', labelsize=8)
    for spine in ax1.spines.values():
        spine.set_color('#334155')
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.ax.tick_params(colors='white', labelsize=8)
    cbar1.set_label('Normalized Intensity [0-1]', color='white', fontsize=9)

    # Panel 2: Predicted Oil Spill Probability Heatmap
    ax2 = axes[0, 1]
    ax2.set_facecolor('#0f172a')
    im2 = ax2.imshow(prob_map, cmap='inferno', vmin=0.0, vmax=1.0, aspect='auto')
    ax2.set_title("2. Model Probability Heatmap", color='white', fontsize=12, fontweight='bold', pad=10)
    ax2.tick_params(colors='gray', labelsize=8)
    for spine in ax2.spines.values():
        spine.set_color('#334155')
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.ax.tick_params(colors='white', labelsize=8)
    cbar2.set_label('Spill Probability', color='white', fontsize=9)

    # Panel 3: Cleaned Binary Spill Mask
    ax3 = axes[1, 0]
    ax3.set_facecolor('#0f172a')
    im3 = ax3.imshow(binary_mask, cmap='Blues_r', vmin=0, vmax=1, aspect='auto')
    ax3.set_title(f"3. Binary Spill Mask (Threshold >= {threshold:.2f})", color='white', fontsize=12, fontweight='bold', pad=10)
    ax3.tick_params(colors='gray', labelsize=8)
    for spine in ax3.spines.values():
        spine.set_color('#334155')
    cbar3 = plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04, ticks=[0, 1])
    cbar3.ax.set_yticklabels(['Water/Background', 'Oil Spill'])
    cbar3.ax.tick_params(colors='white', labelsize=8)

    # Panel 4: Full Overlay with Detections, Contours & Bounding Boxes
    ax4 = axes[1, 1]
    ax4.set_facecolor('#0f172a')
    ax4.imshow(norm_img, cmap='gray', aspect='auto')
    
    # Semi-transparent red overlay for detected spill mask
    overlay_rgb = np.zeros((H, W, 4), dtype=np.float32)
    overlay_rgb[binary_mask == 1] = [1.0, 0.15, 0.15, 0.45]  # Translucent red
    ax4.imshow(overlay_rgb, aspect='auto')

    # Draw bounding boxes, centroids, and labels
    for c in components:
        min_r, min_c, max_r, max_c = c.bbox_pixel
        box_w = max_c - min_c
        box_h = max_r - min_r
        
        # Bounding box
        rect = patches.Rectangle(
            (min_c, min_r), box_w, box_h,
            linewidth=1.8, edgecolor='#00ffcc', facecolor='none', linestyle='--'
        )
        ax4.add_patch(rect)
        
        # Centroid marker
        ax4.plot(c.centroid_pixel[1], c.centroid_pixel[0], marker='x', markersize=7, color='#ff0055', markeredgewidth=2)

        # Label tag
        tag_text = f"Spill #{c.component_id}\nConf: {c.peak_confidence:.2f}"
        ax4.text(
            min_c, max(0, min_r - 8), tag_text,
            color='black', fontsize=7, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#00ffcc', alpha=0.85, edgecolor='none')
        )

    ax4.set_title(f"4. Detections Overlay ({len(components)} Spills Detected)", color='white', fontsize=12, fontweight='bold', pad=10)
    ax4.tick_params(colors='gray', labelsize=8)
    for spine in ax4.spines.values():
        spine.set_color('#334155')

    plt.suptitle(
        f"SAR Oil Spill Analysis - Scene: {img_file.name}\n"
        f"Spill Pixels: {total_spill_pixels:,} | Approx Area: {total_area_km2:.3f} sq km | Total Time: {time.time() - start_time:.2f}s",
        color='white', fontsize=14, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    # Save output plot
    if not output_plot_path:
        out_dir = settings.paths.data_outputs / "test_visualizations"
        out_dir.mkdir(parents=True, exist_ok=True)
        output_plot_path = out_dir / f"{img_file.stem}_test_result.png"
    else:
        output_plot_path = Path(output_plot_path)
        output_plot_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(str(output_plot_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
    plt.close(fig)

    print(f"\n [OK] Diagnostic Visualization Saved Successfully:")
    print(f"      -> {output_plot_path.resolve()}")
    print("=" * 75 + "\n")

    if show_window:
        try:
            import os
            os.system(f'start "" "{output_plot_path.resolve()}"')
        except Exception:
            pass

    return {
        "image_path": str(img_file),
        "output_plot": str(output_plot_path),
        "spills_detected": len(components),
        "total_spill_pixels": total_spill_pixels,
        "total_area_km2": total_area_km2,
        "inference_time_sec": inf_time,
        "components": components,
    }


def main():
    parser = argparse.ArgumentParser(description="Test Oil Spill Model on Single Input Image")
    parser.add_argument("--image", "-i", type=str, required=True, help="Path to input SAR image (.tif, .png, .jpg, etc.)")
    parser.add_argument("--threshold", "-t", type=float, default=0.50, help="Spill probability threshold (0.0 to 1.0, default: 0.50)")
    parser.add_argument("--min-pixels", "-m", type=int, default=50, help="Minimum connected spill area in pixels (default: 50)")
    parser.add_argument("--output", "-o", type=str, default=None, help="Path to save diagnostic PNG image")
    parser.add_argument("--checkpoint", "-c", type=str, default=None, help="Path to model checkpoint (.pt)")
    parser.add_argument("--show", "-s", action="store_true", help="Open generated visualization automatically")
    args = parser.parse_args()

    run_manual_test(
        image_path=args.image,
        threshold=args.threshold,
        min_pixels=args.min_pixels,
        output_plot_path=args.output,
        checkpoint_path=args.checkpoint,
        show_window=args.show,
    )


if __name__ == "__main__":
    main()
