"""
Standalone Sentinel-1 SAR Oil Spill Detection Pipeline.
Processes Sentinel-1 SAR images to detect oil spills, draws bounding boxes,
and resolves geospatial metadata (AOI, Lat/Lon, Date/Time) when provided.
"""
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import json
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from src.config.settings import settings
from src.inference.predictor import OilSpillPredictor
from src.preprocessing.sar_preprocessor import SARPreprocessor
from src.preprocessing.georeference import GeoreferenceTransform
from src.postprocessing.mask_processor import MaskPostProcessor, SpillComponent
from src.geospatial.area_calculator import calculate_geometry_area
from shapely.geometry import Polygon, MultiPolygon, box
from shapely.validation import make_valid
from src.utils.logger import get_logger

logger = get_logger("inference.oil_spill_detector")


@dataclass
class OilSpillDetection:
    """Represents a single detected oil spill region."""
    spill_id: int
    pixel_area: int
    mean_confidence: float
    peak_confidence: float
    bbox_pixel: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    centroid_pixel: Tuple[float, float]    # (row, col)
    
    # Metadata fields (populated ONLY when metadata is available)
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    geo_bbox: Optional[Dict[str, float]] = None  # min_lat, min_lon, max_lat, max_lon
    area_km2: Optional[float] = None
    area_hectares: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "spill_id": self.spill_id,
            "pixel_area": self.pixel_area,
            "mean_confidence": round(self.mean_confidence, 4),
            "peak_confidence": round(self.peak_confidence, 4),
            "bbox_pixel": {
                "min_row": self.bbox_pixel[0],
                "min_col": self.bbox_pixel[1],
                "max_row": self.bbox_pixel[2],
                "max_col": self.bbox_pixel[3],
                "width_px": self.bbox_pixel[3] - self.bbox_pixel[1],
                "height_px": self.bbox_pixel[2] - self.bbox_pixel[0],
            },
            "centroid_pixel": {
                "row": round(self.centroid_pixel[0], 2),
                "col": round(self.centroid_pixel[1], 2),
            },
        }
        if self.latitude is not None and self.longitude is not None:
            data["location"] = {
                "latitude": round(self.latitude, 6),
                "longitude": round(self.longitude, 6),
            }
        if self.geo_bbox is not None:
            data["geo_bbox"] = {
                "min_latitude": round(self.geo_bbox["min_lat"], 6),
                "min_longitude": round(self.geo_bbox["min_lon"], 6),
                "max_latitude": round(self.geo_bbox["max_lat"], 6),
                "max_longitude": round(self.geo_bbox["max_lon"], 6),
            }
        if self.area_km2 is not None:
            data["estimated_area_km2"] = round(self.area_km2, 4)
            data["estimated_area_hectares"] = round(self.area_hectares, 2)
        return data


@dataclass
class DetectionResult:
    """Complete result object returned by Sentinel1OilSpillDetector."""
    image_path: str
    output_image_path: Optional[str]
    has_metadata: bool
    spills_detected: int
    spills: List[OilSpillDetection]
    total_spill_pixels: int
    image_dimensions: Tuple[int, int]  # (height, width)
    
    # Metadata fields (None if metadata is not provided)
    aoi: Optional[Dict[str, Any]] = None
    acquisition_date: Optional[str] = None
    acquisition_time: Optional[str] = None
    total_area_km2: Optional[float] = None
    metadata_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        res = {
            "image_path": self.image_path,
            "output_image_path": self.output_image_path,
            "has_metadata": self.has_metadata,
            "spills_detected": self.spills_detected,
            "image_dimensions": {
                "height": self.image_dimensions[0],
                "width": self.image_dimensions[1],
            },
            "total_spill_pixels": self.total_spill_pixels,
            "spills": [s.to_dict() for s in self.spills],
        }
        if self.has_metadata:
            res["metadata"] = {
                "aoi": self.aoi,
                "acquisition_date": self.acquisition_date,
                "acquisition_time": self.acquisition_time,
                "total_spill_area_km2": round(self.total_area_km2, 4) if self.total_area_km2 is not None else None,
                "metadata_source": self.metadata_source,
            }
        return res

    def print_summary(self):
        """Print a user-friendly console report."""
        print("\n" + "=" * 75)
        print("         SENTINEL-1 SAR OIL SPILL DETECTION REPORT          ")
        print("=" * 75)
        print(f" Input Image:       {self.image_path}")
        print(f" Output Image:      {self.output_image_path or 'Not saved'}")
        print(f" Image Dimensions:  {self.image_dimensions[1]} x {self.image_dimensions[0]} (WxH)")
        print(f" Metadata Mode:     {'METADATA PROVIDED' if self.has_metadata else 'IMAGE-ONLY (No metadata)'}")
        
        if self.has_metadata:
            print(f" Acquisition Date:  {self.acquisition_date or 'N/A'}")
            print(f" Acquisition Time:  {self.acquisition_time or 'N/A'}")
            if self.aoi:
                print(f" Area of Interest:  Min(Lat: {self.aoi.get('min_latitude')}, Lon: {self.aoi.get('min_longitude')}) -> Max(Lat: {self.aoi.get('max_latitude')}, Lon: {self.aoi.get('max_longitude')})")
            if self.total_area_km2 is not None:
                print(f" Total Spill Area:  {self.total_area_km2:.4f} sq km ({self.total_area_km2 * 100:.2f} ha)")

        print(f" Spills Detected:   {self.spills_detected}")
        print(f" Total Spill Pixels:{self.total_spill_pixels:,}")
        print("-" * 75)

        if self.spills_detected > 0:
            if self.has_metadata:
                print(f"{'ID':<4} | {'Confidence':<10} | {'Area (km2)':<12} | {'Lat, Lon':<24} | {'Pixel BBox (r0,c0,r1,c1)':<24}")
                print("-" * 75)
                for s in self.spills:
                    lat_lon = f"{s.latitude:.5f}, {s.longitude:.5f}" if s.latitude is not None else "N/A"
                    area_str = f"{s.area_km2:.4f} km2" if s.area_km2 is not None else f"{s.pixel_area} px"
                    bbox_str = f"({s.bbox_pixel[0]}, {s.bbox_pixel[1]}, {s.bbox_pixel[2]}, {s.bbox_pixel[3]})"
                    print(f"{s.spill_id:<4} | {s.peak_confidence:<10.3f} | {area_str:<12} | {lat_lon:<24} | {bbox_str:<24}")
            else:
                print(f"{'ID':<4} | {'Confidence':<10} | {'Area (px)':<12} | {'Centroid (Row,Col)':<22} | {'Pixel BBox (r0,c0,r1,c1)':<24}")
                print("-" * 75)
                for s in self.spills:
                    centroid_str = f"({s.centroid_pixel[0]:.1f}, {s.centroid_pixel[1]:.1f})"
                    bbox_str = f"({s.bbox_pixel[0]}, {s.bbox_pixel[1]}, {s.bbox_pixel[2]}, {s.bbox_pixel[3]})"
                    print(f"{s.spill_id:<4} | {s.peak_confidence:<10.3f} | {s.pixel_area:<12} | {centroid_str:<22} | {bbox_str:<24}")
        else:
            print(" [*] Clean Scene: No oil spills detected above threshold.")
        print("=" * 75 + "\n")


class Sentinel1OilSpillDetector:
    """
    Production-ready Sentinel-1 SAR Oil Spill Detector.
    
    Can be initialized once and reused for multiple images.
    Supports GeoTIFF, TIFF, PNG, JPG image formats with optional metadata parsing.
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        threshold: float = 0.50,
        min_spill_pixels: int = 50,
        device: Optional[str] = None,
    ):
        self.threshold = threshold
        self.min_spill_pixels = min_spill_pixels
        self.preprocessor = SARPreprocessor()
        
        ckpt = Path(checkpoint_path) if checkpoint_path else settings.paths.checkpoints_dir / "best_model.pt"
        self.predictor = OilSpillPredictor(checkpoint_path=ckpt, device=device)
        self.postprocessor = MaskPostProcessor(threshold=threshold, min_pixels=min_spill_pixels)
        logger.info(f"Sentinel1OilSpillDetector initialized (threshold={threshold}, min_pixels={min_spill_pixels})")

    def _parse_metadata(
        self,
        image_path: Union[str, Path],
        metadata_input: Optional[Union[Dict[str, Any], str, Path]] = None,
        geo_transform_from_file: Optional[GeoreferenceTransform] = None,
    ) -> Tuple[bool, Optional[GeoreferenceTransform], Optional[Dict[str, Any]], Optional[str], Optional[str], Optional[str]]:
        """
        Parse metadata from user-provided dict/JSON or embedded GeoTIFF tags.
        Returns:
            (has_metadata, geo_transform, aoi_dict, date_str, time_str, source_description)
        """
        user_meta: Dict[str, Any] = {}
        if metadata_input is not None:
            if isinstance(metadata_input, (str, Path)):
                meta_path = Path(metadata_input)
                if meta_path.exists():
                    try:
                        with open(meta_path, "r", encoding="utf-8") as f:
                            user_meta = json.load(f)
                    except Exception as e:
                        logger.warning(f"Failed to parse metadata JSON file: {e}")
            elif isinstance(metadata_input, dict):
                user_meta = metadata_input

        # Check if user provided explicit date/time/aoi
        date_str = (
            user_meta.get("date")
            or user_meta.get("acquisition_date")
            or user_meta.get("time_stamp")
        )
        time_str = (
            user_meta.get("time")
            or user_meta.get("acquisition_time")
        )
        datetime_raw = (
            user_meta.get("datetime")
            or user_meta.get("timestamp")
            or user_meta.get("acquisition_datetime")
        )
        if datetime_raw and (not date_str or not time_str):
            try:
                dt_obj = datetime.fromisoformat(str(datetime_raw).replace("Z", "+00:00"))
                if not date_str:
                    date_str = dt_obj.strftime("%Y-%m-%d")
                if not time_str:
                    time_str = dt_obj.strftime("%H:%M:%S UTC")
            except Exception:
                if not date_str:
                    date_str = str(datetime_raw)

        # Check for user-provided AOI / Bounds
        geo_transform = None
        aoi_dict = None
        source_desc = None

        aoi_input = user_meta.get("aoi") or user_meta.get("bbox") or user_meta.get("bounds") or user_meta.get("coordinates")
        if aoi_input is not None:
            # Parse bounds: [min_lon, min_lat, max_lon, max_lat] or dict
            bounds_list = None
            if isinstance(aoi_input, (list, tuple)) and len(aoi_input) == 4:
                # Expect [min_lon, min_lat, max_lon, max_lat]
                bounds_list = [float(x) for x in aoi_input]
            elif isinstance(aoi_input, dict):
                min_lon = aoi_input.get("min_lon") or aoi_input.get("min_longitude") or aoi_input.get("west")
                min_lat = aoi_input.get("min_lat") or aoi_input.get("min_latitude") or aoi_input.get("south")
                max_lon = aoi_input.get("max_lon") or aoi_input.get("max_longitude") or aoi_input.get("east")
                max_lat = aoi_input.get("max_lat") or aoi_input.get("max_latitude") or aoi_input.get("north")
                if all(v is not None for v in [min_lon, min_lat, max_lon, max_lat]):
                    bounds_list = [float(min_lon), float(min_lat), float(max_lon), float(max_lat)]

            if bounds_list:
                geo_transform = GeoreferenceTransform(
                    bounds=bounds_list,
                    crs=user_meta.get("crs", "EPSG:4326"),
                )
                aoi_dict = {
                    "min_longitude": bounds_list[0],
                    "min_latitude": bounds_list[1],
                    "max_longitude": bounds_list[2],
                    "max_latitude": bounds_list[3],
                }
                source_desc = "User-Provided Metadata"

        # If no user-provided transform, check if GeoTIFF had valid georeference
        if geo_transform is None and geo_transform_from_file is not None and not getattr(geo_transform_from_file, "is_fallback", False):
            try:
                b = geo_transform_from_file.get_bounds()
                if b and len(b) == 4 and not (b[0] == -90.0 and b[1] == 28.0):
                    geo_transform = geo_transform_from_file
                    aoi_dict = {
                        "min_longitude": round(b[0], 6),
                        "min_latitude": round(b[1], 6),
                        "max_longitude": round(b[2], 6),
                        "max_latitude": round(b[3], 6),
                    }
                    source_desc = "GeoTIFF Embedded Metadata"
            except Exception:
                pass

        # Determine if metadata was provided
        has_metadata = bool(user_meta or (geo_transform is not None and source_desc == "GeoTIFF Embedded Metadata"))

        if has_metadata and not source_desc:
            source_desc = "User-Provided Metadata"

        return has_metadata, geo_transform, aoi_dict, date_str, time_str, source_desc

    def detect(
        self,
        image: Union[str, Path, np.ndarray, Image.Image],
        metadata: Optional[Union[Dict[str, Any], str, Path]] = None,
        output_image_path: Optional[Union[str, Path]] = None,
        threshold: Optional[float] = None,
        min_spill_pixels: Optional[int] = None,
        save_json: Optional[Union[str, Path]] = None,
    ) -> DetectionResult:
        """
        Run oil spill detection on input SAR image.
        
        Args:
            image: Path to SAR image file or numpy array / PIL Image.
            metadata: Optional metadata dict or path to JSON file containing date/time/aoi.
            output_image_path: Path to save the annotated output image (optional).
            threshold: Custom probability threshold (default: self.threshold).
            min_spill_pixels: Minimum spill area in pixels (default: self.min_spill_pixels).
            save_json: Optional path to export detection result as JSON.
            
        Returns:
            DetectionResult object with detections and conditional metadata.
        """
        th = threshold if threshold is not None else self.threshold
        min_px = min_spill_pixels if min_spill_pixels is not None else self.min_spill_pixels

        # 1. Load image and determine file path / array
        image_path_str = str(image) if isinstance(image, (str, Path)) else "in_memory_array"
        geo_transform_from_file = None

        if isinstance(image, (str, Path)):
            img_file = Path(image)
            if not img_file.exists():
                raise FileNotFoundError(f"Input SAR image not found at: {image}")
            norm_img, _, geo_transform_from_file = self.preprocessor.load_and_preprocess(str(img_file))
        elif isinstance(image, Image.Image):
            raw_arr = np.array(image)
            if raw_arr.ndim == 3:
                raw_arr = raw_arr[..., 0]
            norm_img, _ = self.preprocessor.preprocess_image(raw_arr)
        elif isinstance(image, np.ndarray):
            raw_arr = image
            if raw_arr.ndim == 3:
                raw_arr = raw_arr[..., 0]
            norm_img, _ = self.preprocessor.preprocess_image(raw_arr)
        else:
            raise ValueError(f"Unsupported image input type: {type(image)}")

        H, W = norm_img.shape[:2]

        # 2. Resolve Metadata
        has_metadata, geo_transform, aoi_dict, date_str, time_str, source_desc = self._parse_metadata(
            image_path=image_path_str,
            metadata_input=metadata,
            geo_transform_from_file=geo_transform_from_file,
        )

        if geo_transform is not None:
            geo_transform.width = W
            geo_transform.height = H

        # 3. Model Inference (Sliding window over full scene)
        prob_map = self.predictor.predict_scene(norm_img)

        # 4. Post-processing & Connected Components
        postproc = MaskPostProcessor(threshold=th, min_pixels=min_px)
        binary_mask, components = postproc.process(prob_map)
        total_spill_pixels = int(np.sum(binary_mask))

        # 5. Extract Detections
        detections: List[OilSpillDetection] = []
        total_area_km2 = 0.0

        for comp in components:
            spill = OilSpillDetection(
                spill_id=comp.component_id,
                pixel_area=comp.pixel_area,
                mean_confidence=comp.mean_confidence,
                peak_confidence=comp.peak_confidence,
                bbox_pixel=comp.bbox_pixel,
                centroid_pixel=comp.centroid_pixel,
            )

            # If metadata / georeference is active, calculate real-world coordinates and area
            if geo_transform is not None:
                # Geographic Centroid
                c_lon, c_lat = geo_transform.pixel_to_geo(row=comp.centroid_pixel[0], col=comp.centroid_pixel[1])
                spill.latitude = round(c_lat, 6)
                spill.longitude = round(c_lon, 6)

                # Geographic Bounding Box
                min_r, min_c, max_r, max_c = comp.bbox_pixel
                lon1, lat1 = geo_transform.pixel_to_geo(row=min_r, col=min_c)
                lon2, lat2 = geo_transform.pixel_to_geo(row=max_r, col=max_c)
                spill.geo_bbox = {
                    "min_lat": min(lat1, lat2),
                    "min_lon": min(lon1, lon2),
                    "max_lat": max(lat1, lat2),
                    "max_lon": max(lon1, lon2),
                }

                # Geodesic Area
                try:
                    poly_coords = []
                    for pt in comp.contours[0] if comp.contours else []:
                        c, r = float(pt[0]), float(pt[1])
                        lon_pt, lat_pt = geo_transform.pixel_to_geo(row=r, col=c)
                        poly_coords.append((lon_pt, lat_pt))
                    if len(poly_coords) >= 3:
                        poly = Polygon(poly_coords)
                        if not poly.is_valid:
                            poly = make_valid(poly)
                        area_km2, _ = calculate_geometry_area(poly)
                        spill.area_km2 = area_km2
                        spill.area_hectares = area_km2 * 100.0
                        total_area_km2 += area_km2
                except Exception:
                    # Fallback approximation from pixel resolution
                    pixel_w = getattr(geo_transform, "pixel_width", 10.0) or 10.0
                    area_km2 = (comp.pixel_area * (pixel_w ** 2)) / 1e6
                    spill.area_km2 = area_km2
                    spill.area_hectares = area_km2 * 100.0
                    total_area_km2 += area_km2

            detections.append(spill)

        # 6. Generate Highlighting & Annotated Output Image
        saved_out_path = None
        if output_image_path or isinstance(image, (str, Path)):
            if not output_image_path:
                out_dir = settings.paths.data_outputs / "detections"
                out_dir.mkdir(parents=True, exist_ok=True)
                stem = Path(image_path_str).stem if image_path_str != "in_memory_array" else "detection"
                output_image_path = out_dir / f"{stem}_spill_detected.png"
            else:
                output_image_path = Path(output_image_path)
                output_image_path.parent.mkdir(parents=True, exist_ok=True)

            self._render_annotated_image(
                norm_img=norm_img,
                binary_mask=binary_mask,
                detections=detections,
                has_metadata=has_metadata,
                date_str=date_str,
                time_str=time_str,
                aoi_dict=aoi_dict,
                total_area_km2=total_area_km2 if geo_transform is not None else None,
                output_path=output_image_path,
            )
            saved_out_path = str(output_image_path)

        # 7. Construct Result Object
        result = DetectionResult(
            image_path=image_path_str,
            output_image_path=saved_out_path,
            has_metadata=has_metadata,
            spills_detected=len(detections),
            spills=detections,
            total_spill_pixels=total_spill_pixels,
            image_dimensions=(H, W),
            aoi=aoi_dict if has_metadata else None,
            acquisition_date=date_str if has_metadata else None,
            acquisition_time=time_str if has_metadata else None,
            total_area_km2=total_area_km2 if (has_metadata and geo_transform is not None) else None,
            metadata_source=source_desc if has_metadata else None,
        )

        if save_json:
            json_path = Path(save_json)
            json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(result.to_dict(), f, indent=2)
            logger.info(f"Saved detection result JSON to: {json_path}")

        return result

    def _render_annotated_image(
        self,
        norm_img: np.ndarray,
        binary_mask: np.ndarray,
        detections: List[OilSpillDetection],
        has_metadata: bool,
        date_str: Optional[str],
        time_str: Optional[str],
        aoi_dict: Optional[Dict[str, float]],
        total_area_km2: Optional[float],
        output_path: Path,
    ):
        """
        Draw clean, high-visibility bounding boxes and highlighting on the SAR scene.
        """
        H, W = norm_img.shape[:2]
        fig, ax = plt.subplots(figsize=(12, 10), dpi=150)
        fig.patch.set_facecolor('#0b1120')  # Dark theme
        ax.set_facecolor('#0b1120')

        # 1. Base grayscale SAR Image
        ax.imshow(norm_img, cmap='gray', aspect='auto')

        # 2. Semi-transparent Oil Spill Highlight Mask (Translucent Red / Orange)
        overlay = np.zeros((H, W, 4), dtype=np.float32)
        overlay[binary_mask == 1] = [1.0, 0.15, 0.20, 0.55]
        ax.imshow(overlay, aspect='auto')

        # 3. Draw Bounding Boxes, Centroids, and Badges for each Spill
        for spill in detections:
            min_r, min_c, max_r, max_c = spill.bbox_pixel
            box_w = max_c - min_c
            box_h = max_r - min_r

            # Neon Cyan/Green Bounding Box
            rect = patches.Rectangle(
                (min_c, min_r), box_w, box_h,
                linewidth=2.2,
                edgecolor='#00ffcc',
                facecolor='none',
                linestyle='-',
            )
            ax.add_patch(rect)

            # Centroid Marker
            ax.plot(
                spill.centroid_pixel[1],
                spill.centroid_pixel[0],
                marker='+',
                markersize=9,
                color='#ff0055',
                markeredgewidth=2.2,
            )

            # Spill Label Tag
            if has_metadata and spill.latitude is not None:
                tag = f"Spill #{spill.spill_id}\nConf: {spill.peak_confidence:.2f}\nLat: {spill.latitude:.4f}\nLon: {spill.longitude:.4f}"
            else:
                tag = f"Spill #{spill.spill_id}\nConf: {spill.peak_confidence:.2f}\nArea: {spill.pixel_area}px"

            ax.text(
                min_c,
                max(0, min_r - 6),
                tag,
                color='black',
                fontsize=7.5,
                fontweight='bold',
                bbox=dict(
                    boxstyle='round,pad=0.3',
                    facecolor='#00ffcc',
                    alpha=0.92,
                    edgecolor='none',
                ),
            )

        ax.tick_params(colors='gray', labelsize=8)
        for spine in ax.spines.values():
            spine.set_color('#1e293b')

        # 4. Title & Header Info Banner
        if has_metadata:
            meta_line = f"Date: {date_str or 'N/A'} | Time: {time_str or 'N/A'}"
            if aoi_dict:
                meta_line += f" | AOI: [{aoi_dict['min_longitude']:.2f}, {aoi_dict['min_latitude']:.2f}, {aoi_dict['max_longitude']:.2f}, {aoi_dict['max_latitude']:.2f}]"
            if total_area_km2 is not None:
                meta_line += f" | Area: {total_area_km2:.3f} km²"

            plt.title(
                f"Sentinel-1 SAR Oil Spill Detection ({len(detections)} Spills Detected)\n{meta_line}",
                color='white',
                fontsize=11,
                fontweight='bold',
                pad=12,
            )
        else:
            plt.title(
                f"Sentinel-1 SAR Oil Spill Detection ({len(detections)} Spills Detected) [Image Only]",
                color='white',
                fontsize=11,
                fontweight='bold',
                pad=12,
            )

        plt.tight_layout()
        plt.savefig(str(output_path), dpi=150, facecolor=fig.get_facecolor(), edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        logger.info(f"Annotated detection image saved to: {output_path}")
