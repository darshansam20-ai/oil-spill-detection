"""
Unit tests for FastAPI REST Endpoints.
"""
from datetime import datetime
import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.storage.models import SatelliteScene, OilSpillEvent
from src.storage.repository import repo

client = TestClient(app)


def test_system_status_endpoint():
    response = client.get("/api/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "HEALTHY"
    assert "ConvNeXt" in data["model_architecture"]
    assert "audit_disclaimer" in data


def test_scene_endpoints():
    scene = SatelliteScene(
        scene_id="API_TEST_SCENE",
        acquisition_time=datetime(2020, 1, 1),
    )
    repo.create_or_update_scene(scene)

    resp_list = client.get("/api/scenes")
    assert resp_list.status_code == 200
    scenes = resp_list.json()
    assert any(s["scene_id"] == "API_TEST_SCENE" for s in scenes)

    resp_single = client.get("/api/scenes/API_TEST_SCENE")
    assert resp_single.status_code == 200
    assert resp_single.json()["scene_id"] == "API_TEST_SCENE"


def test_event_and_geojson_endpoints():
    event = OilSpillEvent(
        event_id="OSE-API-TEST",
        scene_id="API_TEST_SCENE",
        timestamp=datetime(2020, 1, 1),
        centroid_lat=28.0,
        centroid_lon=-90.0,
        polygon={"type": "Polygon", "coordinates": [[[-90, 28], [-89.9, 28], [-89.9, 28.1], [-90, 28.1], [-90, 28]]]},
        bounding_box=[-90, 28, -89.9, 28.1],
        area_km2=1.5,
        area_m2=1_500_000,
        confidence=0.88,
        peak_confidence=0.95,
        model_version="v1.0.0",
        threshold=0.5,
    )
    repo.save_oil_spill_events([event])

    resp_events = client.get("/api/events")
    assert resp_events.status_code == 200
    assert any(e["event_id"] == "OSE-API-TEST" for e in resp_events.json())

    resp_geojson = client.get("/api/events/geojson")
    assert resp_geojson.status_code == 200
    fc = resp_geojson.json()
    assert fc["type"] == "FeatureCollection"
    assert len(fc["features"]) >= 1
