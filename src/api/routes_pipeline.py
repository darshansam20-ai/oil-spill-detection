"""
API Routes for Pipeline Execution and Processing Management.
"""
from typing import Optional
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel

from src.worker.queue_worker import get_pipeline_worker
from src.storage.repository import repo

router = APIRouter(prefix="/api/pipeline", tags=["Pipeline"])


class ProcessSceneRequest(BaseModel):
    scene_id: str
    force: bool = False


@router.post("/process-scene")
def trigger_scene_processing(req: ProcessSceneRequest, background_tasks: BackgroundTasks):
    """Trigger the full end-to-end oil spill detection pipeline on a scene."""
    scene = repo.get_scene(req.scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{req.scene_id}' not found.")

    # Execute synchronously or in background
    res = get_pipeline_worker().process_scene(scene_id=req.scene_id, force=req.force)
    return res


@router.get("/status/{scene_id}")
def get_scene_processing_status(scene_id: str):
    """Check processing lifecycle state of a scene."""
    scene = repo.get_scene(scene_id)
    if not scene:
        raise HTTPException(status_code=404, detail=f"Scene '{scene_id}' not found.")
    
    mask = repo.get_spill_mask(scene_id)
    events = repo.list_events(scene_id=scene_id)
    
    return {
        "scene_id": scene.scene_id,
        "status": scene.status.value,
        "error_message": scene.error_message,
        "processed_at": scene.processed_at,
        "spills_detected": len(events),
        "mask_available": mask is not None,
    }
