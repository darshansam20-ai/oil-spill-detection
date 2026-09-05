"""
Unit tests for Ingestion and Scene Filtering (FR-01 to FR-09).
"""
from datetime import datetime
import pytest
from shapely.geometry import box

from src.config.constants import SceneStatus, Polarization, OrbitDirection
from src.ingestion.stac_client import STACDiscoveryClient
from src.ingestion.scene_filter import SceneFilter
from src.ingestion.status_tracker import StatusTracker
from src.ingestion.idempotency import IdempotencyManager
from src.storage.models import SatelliteScene
from src.storage.repository import DatabaseRepository


@pytest.fixture
def test_repo(tmp_path):
    db_file = tmp_path / "test_ingestion.db"
    return DatabaseRepository(db_path=db_file)


def test_stac_local_scene_creation():
    client = STACDiscoveryClient()
    scene = client.create_scene_from_local_file(
        scene_id="S1A_TEST_001",
        file_path="/tmp/test.tif",
        bbox=[-91.0, 27.0, -90.0, 28.0],
    )
    assert scene.scene_id == "S1A_TEST_001"
    assert scene.status == SceneStatus.DISCOVERED
    assert scene.geometry is not None
    assert scene.geometry["type"] == "Polygon"


def test_scene_filtering_polarization():
    filter_obj = SceneFilter(required_polarization="VV")
    
    valid_scene = SatelliteScene(
        scene_id="S1A_VV",
        acquisition_time=datetime(2020, 5, 1),
        polarization="VV",
    )
    invalid_scene = SatelliteScene(
        scene_id="S1A_HH",
        acquisition_time=datetime(2020, 5, 1),
        polarization="HH",
    )
    
    assert filter_obj.filter_scene(valid_scene) is True
    assert filter_obj.filter_scene(invalid_scene) is False


def test_scene_filtering_spatial_aoi():
    # AOI covering Gulf of Mexico [-92 to -88 lon, 26 to 29 lat]
    aoi = box(-92.0, 26.0, -88.0, 29.0)
    filter_obj = SceneFilter(aoi_polygon=aoi)

    inside_scene = SatelliteScene(
        scene_id="INSIDE",
        acquisition_time=datetime(2020, 5, 1),
        geometry={"type": "Polygon", "coordinates": [[[-90, 27], [-89, 27], [-89, 28], [-90, 28], [-90, 27]]]},
    )
    outside_scene = SatelliteScene(
        scene_id="OUTSIDE",
        acquisition_time=datetime(2020, 5, 1),
        geometry={"type": "Polygon", "coordinates": [[[-120, 30], [-119, 30], [-119, 31], [-120, 31], [-120, 30]]]},
    )

    assert filter_obj.filter_scene(inside_scene) is True
    assert filter_obj.filter_scene(outside_scene) is False


def test_status_tracker_and_idempotency(test_repo):
    tracker = StatusTracker(test_repo)
    idempotency = IdempotencyManager(test_repo)

    scene = SatelliteScene(
        scene_id="S1A_IDEM_TEST",
        acquisition_time=datetime(2020, 6, 1),
        status=SceneStatus.DISCOVERED,
    )
    test_repo.create_or_update_scene(scene)

    assert idempotency.should_process("S1A_IDEM_TEST") is True

    # Transition to COMPLETED
    tracker.transition("S1A_IDEM_TEST", SceneStatus.COMPLETED)
    assert tracker.get_status("S1A_IDEM_TEST") == SceneStatus.COMPLETED
    assert idempotency.is_already_processed("S1A_IDEM_TEST") is True
    assert idempotency.should_process("S1A_IDEM_TEST", force=False) is False
    assert idempotency.should_process("S1A_IDEM_TEST", force=True) is True
