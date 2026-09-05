"""
Full End-to-End Integration Test on Sentinel-1 Oil Spill Scene.
"""
from pathlib import Path
from datetime import datetime
import pytest

from src.config.settings import settings
from src.dataset.archive_handler import DatasetArchiveHandler
from src.ingestion.stac_client import STACDiscoveryClient
from src.storage.repository import repo
from src.worker.queue_worker import PipelineWorker


def test_full_pipeline_execution():
    handler = DatasetArchiveHandler()
    pairs = handler.discover_scene_pairs()
    assert len(pairs) > 0, "No dataset scene pairs found."

    # Pick first available scene
    img_path, mask_path, _ = pairs[0]
    scene_id = img_path.stem

    stac = STACDiscoveryClient()
    scene = stac.create_scene_from_local_file(
        scene_id=scene_id,
        file_path=str(img_path),
        acquisition_time=datetime(2018, 12, 7, 12, 0),
    )
    repo.create_or_update_scene(scene)

    worker = PipelineWorker()
    res = worker.process_scene(scene_id=scene_id, force=True)

    assert res["status"] in ["COMPLETED", "NO_SPILL_DETECTED"]
    assert "report_html" in res
    assert "report_json" in res
    assert Path(res["report_html"]).exists()
    assert Path(res["report_json"]).exists()
