"""
API Routes for System Status, Health, and Model Provenance.
"""
from typing import Any, Dict
import torch
from fastapi import APIRouter

from src.config.constants import AUDIT_DISCLAIMER, CURRENT_MODEL_VERSION, MODEL_ARCHITECTURE
from src.config.settings import settings
from src.storage.repository import repo

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/status")
def get_system_status() -> Dict[str, Any]:
    """Retrieve system health, database status, and CUDA device info."""
    scenes = repo.list_scenes(limit=500)
    events = repo.list_events(limit=500)
    alerts = repo.list_alerts(limit=500)

    cuda_available = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_available else "CPU"

    return {
        "status": "HEALTHY",
        "app_name": settings.app_name,
        "app_version": settings.app_version,
        "model_architecture": MODEL_ARCHITECTURE,
        "model_version": CURRENT_MODEL_VERSION,
        "hardware": {
            "cuda_available": cuda_available,
            "device_name": device_name,
        },
        "stats": {
            "total_scenes": len(scenes),
            "total_events": len(events),
            "total_alerts": len(alerts),
        },
        "audit_disclaimer": AUDIT_DISCLAIMER,
    }
