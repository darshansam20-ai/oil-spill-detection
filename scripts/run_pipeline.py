import argparse
import sys
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings
from src.dataset.archive_handler import DatasetArchiveHandler
from src.ingestion.stac_client import STACDiscoveryClient
from src.storage.repository import repo
from src.worker.queue_worker import PipelineWorker
from src.utils.logger import get_logger

logger = get_logger("scripts.run_pipeline")


def ingest_all_local_scenes():
    """Discover all local dataset scenes and register them in database."""
    handler = DatasetArchiveHandler()
    pairs = handler.discover_scene_pairs()
    stac = STACDiscoveryClient()

    for img_path, _, _ in pairs:
        scene_id = img_path.stem
        # Extract date from filename if format YYYY_MM_DD or YYYYMMDD
        raw_name = scene_id.replace("_", "").replace("b", "").replace("d", "").replace("e", "").replace("f", "")
        try:
            acq_time = datetime.strptime(raw_name[:8], "%Y%m%d")
        except Exception:
            acq_time = datetime(2019, 1, 1)

        scene = stac.create_scene_from_local_file(
            scene_id=scene_id,
            file_path=str(img_path),
            acquisition_time=acq_time,
        )
        repo.create_or_update_scene(scene)

    logger.info(f"Ingested {len(pairs)} scenes into database.")


def main():
    parser = argparse.ArgumentParser(description="Run SAR Oil Spill Detection Pipeline")
    parser.add_argument("--all", action="store_true", help="Process all ingested scenes")
    parser.add_argument("--scene-id", type=str, help="Process specific scene by ID")
    parser.add_argument("--force", action="store_true", help="Force re-processing")
    args = parser.parse_args()

    # Ensure scenes are registered in database
    ingest_all_local_scenes()

    worker = PipelineWorker()

    if args.scene_id:
        res = worker.process_scene(args.scene_id, force=args.force)
        print(f"\nResult for {args.scene_id}: {res['status']} | Spills: {res.get('spills_detected', 0)} | Alerts: {res.get('alerts_triggered', 0)}")
    elif args.all:
        scenes = repo.list_scenes()
        for s in scenes:
            res = worker.process_scene(s.scene_id, force=args.force)
            print(f"Scene {s.scene_id:20s} -> {res['status']} ({res.get('spills_detected', 0)} spills, {res.get('alerts_triggered', 0)} alerts)")
    else:
        print("Please specify --scene-id <ID> or --all to execute pipeline.")


if __name__ == "__main__":
    main()
