"""
API Routes for Oil Spill Events and GeoJSON Layers.
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from src.storage.models import OilSpillEvent
from src.storage.repository import repo
from src.geospatial.event_builder import build_geojson_feature_collection

router = APIRouter(prefix="/api/events", tags=["Spill Events"])


@router.get("", response_model=List[OilSpillEvent])
def list_events(
    scene_id: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
    limit: int = Query(100, ge=1, le=500),
):
    """List detected oil spill events."""
    return repo.list_events(scene_id=scene_id, min_confidence=min_confidence, limit=limit)


@router.get("/geojson", response_model=Dict[str, Any])
def get_events_geojson(
    scene_id: Optional[str] = None,
    min_confidence: Optional[float] = Query(None, ge=0.0, le=1.0),
):
    """Retrieve detected oil spill events as a standard GeoJSON FeatureCollection."""
    events = repo.list_events(scene_id=scene_id, min_confidence=min_confidence)
    return build_geojson_feature_collection(events)


@router.get("/{event_id}", response_model=OilSpillEvent)
def get_event_detail(event_id: str):
    """Retrieve full details for a specific Oil Spill Event ID."""
    event = repo.get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail=f"Event '{event_id}' not found.")
    return event
