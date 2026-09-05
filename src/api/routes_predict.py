"""
Production Prediction & Inference API Router.
Handles satellite image upload, metadata parsing, and sequential execution:
  SAR Deep Learning Detector -> Adapter -> AIS Vessel Correlator -> Incident Report.
"""
import base64
import gc
import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
import torch

from src.pipeline.end_to_end_pipeline import EndToEndPipeline, EndToEndIncidentReport
from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("api.routes_predict")

router = APIRouter(tags=["Inference & Prediction"])

# Supported file extensions & limits
ALLOWED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
MAX_FILE_SIZE_BYTES = 100 * 1024 * 1024  # 100 MB

import threading

# Thread-safe Singleton pipeline instance
_pipeline_instance: Optional[EndToEndPipeline] = None
_pipeline_lock = threading.Lock()


def get_pipeline() -> EndToEndPipeline:
    """Get or initialize singleton EndToEndPipeline in memory (thread-safe)."""
    global _pipeline_instance
    if _pipeline_instance is None:
        with _pipeline_lock:
            if _pipeline_instance is None:
                logger.info("Initializing in-memory EndToEndPipeline singleton...")
                checkpoint_path = os.getenv("MODEL_CHECKPOINT_PATH", str(settings.paths.checkpoints_dir / "best_model.pt"))
                threshold = float(os.getenv("DETECTION_THRESHOLD", "0.50"))
                min_pixels = int(os.getenv("MIN_SPILL_PIXELS", "50"))
                ais_radius = float(os.getenv("AIS_SEARCH_RADIUS_KM", "20.0"))
                ais_token = os.getenv("GFW_API_TOKEN") or os.getenv("AIS_TOKEN") or None
                device = os.getenv("DEVICE", "cpu")

                _pipeline_instance = EndToEndPipeline(
                    checkpoint_path=checkpoint_path,
                    detection_threshold=threshold,
                    min_spill_pixels=min_pixels,
                    ais_token=ais_token,
                    ais_search_radius_km=ais_radius,
                    device=device,
                )
                logger.info("EndToEndPipeline singleton loaded successfully.")
    return _pipeline_instance


def encode_file_to_base64(file_path: Optional[Union[str, Path]], mime_type: str = "image/png") -> Optional[str]:
    """Read a local file and convert it into a base64 Data URI string."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
            return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        logger.warning(f"Failed to encode file {file_path} to base64: {e}")
        return None


def read_text_file(file_path: Optional[Union[str, Path]]) -> Optional[str]:
    """Read a text/html file into a string."""
    if not file_path or not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        logger.warning(f"Failed to read file {file_path}: {e}")
        return None


@router.api_route("/health", methods=["GET", "HEAD"])
@router.api_route("/api/health", methods=["GET", "HEAD"])
def health_check() -> Dict[str, Any]:
    """Lightweight liveness & readiness probe for the ML service."""
    ckpt_path = Path(os.getenv("MODEL_CHECKPOINT_PATH", str(settings.paths.checkpoints_dir / "best_model.pt")))
    is_loaded = _pipeline_instance is not None
    return {
        "status": "online",
        "service": settings.app_name,
        "version": settings.app_version,
        "model_version": settings.model_version,
        "device": str(_pipeline_instance.detector.predictor.device) if is_loaded else "cpu",
        "cuda_available": torch.cuda.is_available(),
        "checkpoint_loaded": ckpt_path.exists(),
        "pipeline_initialized": is_loaded,
    }


@router.post("/predict")
@router.post("/api/predict")
async def run_prediction_pipeline(
    image: UploadFile = File(..., description="Sentinel-1 SAR image (.tif, .tiff, .png, .jpg)"),
    metadata: Optional[str] = Form(None, description="Optional JSON string containing metadata (aoi, date, time, etc.)"),
    date: Optional[str] = Form(None, description="Acquisition Date (e.g. '2018-12-19')"),
    time: Optional[str] = Form(None, description="Acquisition Time (e.g. '06:15:22 UTC')"),
    aoi: Optional[str] = Form(None, description="Area of interest bounding box as 'min_lon,min_lat,max_lon,max_lat'"),
    lat: Optional[float] = Form(None, description="Override epicenter latitude"),
    lon: Optional[float] = Form(None, description="Override epicenter longitude"),
    threshold: Optional[float] = Form(0.50, description="Detection probability threshold (0.0 - 1.0)"),
    min_pixels: Optional[int] = Form(50, description="Minimum connected spill pixels"),
    search_radius_km: Optional[float] = Form(20.0, description="AIS vessel search radius in km"),
) -> JSONResponse:
    """
    Unified Production Endpoint:
    Processes a Sentinel-1 SAR satellite image through the 3-stage sequential pipeline:
    1. SAR Oil Spill Deep Learning Segmentation (ConvNeXt-Tiny + U-Net)
    2. Adapter Layer (Extracts Spill Epicenter, Formats ISO Timestamp, Resolves AOI)
    3. AIS Maritime Vessel Proximity & Trajectory Correlation
    """
    # 1. Validate file extension
    filename = image.filename or "uploaded_sar_image.tif"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file format '{file_ext}'. Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # 2. Parse metadata
    meta_dict: Dict[str, Any] = {}
    if metadata:
        meta_str = metadata.strip()
        if meta_str:
            try:
                meta_dict = json.loads(meta_str)
            except Exception as e:
                logger.warning(f"Failed to parse metadata JSON form field: {e}")

    if date:
        meta_dict["date"] = date.strip()
    if time:
        meta_dict["time"] = time.strip()
    if aoi:
        try:
            parts = [float(x.strip()) for x in aoi.split(",")]
            if len(parts) == 4:
                meta_dict["aoi"] = parts
        except ValueError:
            logger.warning(f"Failed to parse AOI string: {aoi}")

    # 3. Create temporary workspace for isolated inference execution
    temp_dir = Path(tempfile.mkdtemp(prefix="sar_predict_"))
    try:
        temp_input_path = temp_dir / filename
        file_size = 0
        with open(temp_input_path, "wb") as f_out:
            while chunk := await image.read(1024 * 1024):  # 1MB chunks
                file_size += len(chunk)
                if file_size > MAX_FILE_SIZE_BYTES:
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail=f"File exceeds maximum allowed size of {MAX_FILE_SIZE_BYTES / (1024*1024):.0f} MB.",
                    )
                f_out.write(chunk)

        logger.info(f"Received upload: {filename} ({file_size / (1024*1024):.2f} MB)")

        # 4. Execute Pipeline
        pipeline = get_pipeline()
        
        # Override threshold dynamically if specified
        if threshold is not None and 0.0 <= threshold <= 1.0:
            pipeline.detector.threshold = threshold
            if hasattr(pipeline.detector, "postprocessor"):
                pipeline.detector.postprocessor.threshold = threshold
        if min_pixels is not None and min_pixels > 0:
            pipeline.detector.min_spill_pixels = min_pixels
            if hasattr(pipeline.detector, "postprocessor"):
                pipeline.detector.postprocessor.min_pixels = min_pixels

        out_dir = temp_dir / "output"
        out_dir.mkdir(parents=True, exist_ok=True)

        report: EndToEndIncidentReport = pipeline.run(
            image_path=temp_input_path,
            metadata=meta_dict if meta_dict else None,
            output_dir=out_dir,
            search_radius_km=search_radius_km,
            override_lat=lat,
            override_lon=lon,
            override_time=date if date and not time else None,
        )

        # 5. Package results and encode artifacts
        annotated_b64 = encode_file_to_base64(report.annotated_image_path, "image/png")
        map_html_content = read_text_file(report.interactive_map_path)
        ranking_json = [v.to_dict() for v in report.ais_result.ranking] if report.ais_result.ranking else []

        response_data = {
            "status": "success",
            "incident_id": report.incident_id,
            "timestamp": report.timestamp,
            "filename": filename,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "oil_spill_detection": {
                "spills_detected": report.detection_result.spills_detected,
                "total_spill_pixels": report.detection_result.total_spill_pixels,
                "total_area_km2": round(report.detection_result.total_area_km2, 4) if report.detection_result.total_area_km2 is not None else None,
                "has_metadata": report.detection_result.has_metadata,
                "image_dimensions": report.detection_result.image_dimensions,
                "spills": [s.to_dict() for s in report.detection_result.spills],
            },
            "adapter_payload": report.adapter_payload.to_dict(),
            "ais_vessel_correlation": {
                "total_vessels_detected": report.ais_result.total_vessels_detected,
                "data_source": report.ais_result.data_source,
                "search_radius_km": report.ais_result.search_radius_km,
                "epicenter": {
                    "latitude": report.ais_result.spill_latitude,
                    "longitude": report.ais_result.spill_longitude,
                },
                "detection_time": report.ais_result.detection_time,
                "ranked_vessels": ranking_json,
                "primary_suspect": ranking_json[0] if ranking_json else None,
            },
            "artifacts": {
                "annotated_image_data_uri": annotated_b64,
                "interactive_map_html": map_html_content,
                "full_report": report.to_dict(),
            },
        }

        return JSONResponse(content=response_data, status_code=status.HTTP_200_OK)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Inference pipeline execution error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": "Error occurred during model inference pipeline execution.",
                "detail": str(e),
            },
        )
    finally:
        # Cleanup temporary files safely and free memory
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass
        gc.collect()
