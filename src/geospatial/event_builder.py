"""
GeoJSON FeatureCollection Builder for Oil Spill Events.
Formats detected spill events into standard RFC 7946 GeoJSON representations.
"""
from typing import Any, Dict, List
from src.storage.models import OilSpillEvent


def build_geojson_feature(event: OilSpillEvent) -> Dict[str, Any]:
    """
    Format a single OilSpillEvent as a GeoJSON Feature.
    """
    return {
        "type": "Feature",
        "id": event.event_id,
        "geometry": event.polygon,
        "bbox": event.bounding_box,
        "properties": {
            "event_id": event.event_id,
            "scene_id": event.scene_id,
            "timestamp": event.timestamp.isoformat(),
            "centroid": [event.centroid_lon, event.centroid_lat],
            "area_km2": event.area_km2,
            "area_m2": event.area_m2,
            "confidence": event.confidence,
            "peak_confidence": event.peak_confidence,
            "model_version": event.model_version,
            "threshold": event.threshold,
            "status": event.status.value,
        },
    }


def build_geojson_feature_collection(events: List[OilSpillEvent]) -> Dict[str, Any]:
    """
    Format a list of OilSpillEvent records into a GeoJSON FeatureCollection.
    """
    return {
        "type": "FeatureCollection",
        "features": [build_geojson_feature(e) for e in events],
    }
