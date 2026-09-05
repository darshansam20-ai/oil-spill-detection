"""
API Routes for Satellite Scene Management, Ingestion, and Direct Image Upload.
"""
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel

from src.config.constants import SceneStatus, Polarization, OrbitDirection
from src.config.settings import settings
from src.storage.models import SatelliteScene
from src.storage.repository import repo
from src.ingestion.stac_client import STACDiscoveryClient
from src.worker.queue_worker import get_pipeline_worker

router = APIRouter(prefix="/api/scenes", tags=["Scenes"])
stac_client = STACDiscoveryClient()


class IngestLocalSceneRequest(BaseModel):
    scene_id: str
    file_path: str
    acquisition_time: Optional[datetime] = None
    polarization: str = Polarization.VV.value
    orbit_direction: str = OrbitDirection.ASCENDING.value


@router.get("", response_model=List[SatelliteScene])
def list_scenes(status: Optional[str] = None, limit: int = Query(50, ge=1, le=200)):
    """List satellite scenes with optional status filtering."""
    return repo.list_scenes(status=status, limit=limit)


@router.get("/{scene_id}", response_model=SatelliteScene)
def get_scene(scene_id: str):
    """Retrieve metadata for a specific satellite scene."""
    scene = repo.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    return scene


@router.post("/ingest-local", response_model=SatelliteScene)
def ingest_local_scene(req: IngestLocalSceneRequest):
    """Register and ingest a local Sentinel-1 SAR scene file."""
    scene = stac_client.create_scene_from_local_file(
        scene_id=req.scene_id,
        file_path=req.file_path,
        acquisition_time=req.acquisition_time,
        polarization=req.polarization,
        orbit_direction=req.orbit_direction,
    )
    saved = repo.create_or_update_scene(scene)
    return saved


@router.post("/upload-and-analyze")
async def upload_and_analyze_image(
    file: UploadFile = File(...),
    custom_scene_id: Optional[str] = Form(None),
    threshold: Optional[float] = Form(0.50),
    acquisition_time: Optional[str] = Form(None),
) -> Dict[str, Any]:
    """
    Upload an arbitrary SAR image file (GeoTIFF, TIFF, PNG, JPG) and immediately execute
    the full end-to-end Oil Spill Detection & Geospatial Event Generation Pipeline.
    """
    upload_dir = settings.paths.data_extracted / "uploaded"
    upload_dir.mkdir(parents=True, exist_ok=True)

    # Determine unique scene ID and target filename
    original_stem = Path(file.filename).stem
    safe_stem = "".join([c if c.isalnum() or c in "_-" else "_" for c in original_stem])
    timestamp_str = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scene_id = custom_scene_id or f"UPLOAD_{safe_stem}_{timestamp_str}"
    
    file_ext = Path(file.filename).suffix or ".tif"
    target_path = upload_dir / f"{scene_id}{file_ext}"

    # Save uploaded file bytes to disk
    with open(target_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    acq_dt = datetime.utcnow()
    if acquisition_time:
        try:
            acq_dt = datetime.fromisoformat(acquisition_time.replace("Z", "+00:00"))
        except Exception:
            acq_dt = datetime.utcnow()

    # Register in database
    scene = stac_client.create_scene_from_local_file(
        scene_id=scene_id,
        file_path=str(target_path),
        acquisition_time=acq_dt,
    )
    repo.create_or_update_scene(scene)

    # Run the full pipeline with requested threshold
    worker = get_pipeline_worker()
    if threshold is not None:
        worker.postprocessor.threshold = float(threshold)
        worker.geospatializer.threshold = float(threshold)

    result = worker.process_scene(scene_id=scene_id, force=True)
    return result
