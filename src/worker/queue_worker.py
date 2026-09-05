"""
Asynchronous Queue & Pipeline Worker Architecture (PRD Section 5 & 18).
Executes the end-to-end SAR Oil-Spill Detection Pipeline with retry mechanisms,
failure tracking, and database synchronization.
"""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
from PIL import Image

from src.alerting.alert_engine import AlertEngine
from src.config.constants import SceneStatus, ReviewStatus, CURRENT_MODEL_VERSION
from src.config.settings import settings
from src.geospatial.geospatializer import SpillGeospatializer
from src.inference.predictor import OilSpillPredictor
from src.ingestion.idempotency import IdempotencyManager
from src.ingestion.status_tracker import StatusTracker
from src.postprocessing.mask_processor import MaskPostProcessor
from src.preprocessing.sar_preprocessor import SARPreprocessor
from src.reporting.report_generator import InvestigationReportGenerator
from src.storage.models import SatelliteScene, SpillMask, OilSpillEvent, Alert
from src.storage.repository import DatabaseRepository, repo
from src.utils.logger import get_logger

logger = get_logger("worker.queue_worker")


class PipelineWorker:
    """Executes the complete Oil-Spill detection workflow on a Sentinel-1 scene."""

    def __init__(
        self,
        repository: Optional[DatabaseRepository] = None,
        predictor: Optional[OilSpillPredictor] = None,
    ):
        self.repo = repository or repo
        self.status_tracker = StatusTracker(self.repo)
        self.idempotency = IdempotencyManager(self.repo)
        
        self.preprocessor = SARPreprocessor()
        self.predictor = predictor or OilSpillPredictor()
        self.postprocessor = MaskPostProcessor(
            threshold=settings.postprocessing.probability_threshold,
            min_pixels=settings.postprocessing.min_spill_pixels,
            opening_radius=settings.postprocessing.opening_radius,
            closing_radius=settings.postprocessing.closing_radius,
        )
        self.geospatializer = SpillGeospatializer(
            model_version=self.predictor.model_version,
            threshold=settings.postprocessing.probability_threshold,
        )
        self.alert_engine = AlertEngine(
            min_confidence=settings.alerting.confidence_threshold,
            min_area_km2=settings.alerting.area_km2_threshold,
        )
        self.report_generator = InvestigationReportGenerator(self.repo)

    def process_scene(self, scene_id: str, force: bool = False) -> Dict[str, Any]:
        """
        Execute end-to-end pipeline on a satellite scene.
        """
        logger.info(f"=== Starting Processing Pipeline for Scene: {scene_id} ===")
        scene = self.repo.get_scene(scene_id)
        if not scene:
            raise ValueError(f"Scene {scene_id} not found in database.")

        if not self.idempotency.should_process(scene_id, force=force):
            return {"status": "SKIPPED", "scene_id": scene_id, "message": "Already processed."}

        try:
            # 1. INGESTION / VALIDATION
            self.status_tracker.transition(scene_id, SceneStatus.INGESTED)
            if not scene.local_path or not Path(scene.local_path).exists():
                raise FileNotFoundError(f"Scene raster file not found at: {scene.local_path}")

            # 2. PREPROCESSING (FR-10 to FR-14)
            self.status_tracker.transition(scene_id, SceneStatus.PREPROCESSING)
            norm_img, preprocessed_db, geo_transform = self.preprocessor.load_and_preprocess(scene.local_path)

            # 3. AI INFERENCE (FR-15 to FR-17)
            self.status_tracker.transition(scene_id, SceneStatus.INFERENCE)
            prob_map = self.predictor.predict_scene(norm_img)

            # 4. MASK POST-PROCESSING (FR-18 to FR-20)
            self.status_tracker.transition(scene_id, SceneStatus.POSTPROCESSING)
            binary_mask, components = self.postprocessor.process(prob_map)

            # Save mask & probability raster artifacts
            output_mask_path = settings.paths.data_outputs / f"{scene_id}_mask.png"
            output_prob_path = settings.paths.data_outputs / f"{scene_id}_prob.png"
            Image.fromarray(binary_mask * 255).save(output_mask_path)
            Image.fromarray((prob_map * 255).astype(np.uint8)).save(output_prob_path)

            spill_mask_record = SpillMask(
                scene_id=scene_id,
                mask_path=str(output_mask_path),
                prob_map_path=str(output_prob_path),
                threshold=settings.postprocessing.probability_threshold,
                model_version=self.predictor.model_version,
                num_spills_detected=len(components),
            )
            self.repo.save_spill_mask(spill_mask_record)

            # 5. GEOSPATIAL EVENT GENERATION (FR-21 to FR-25)
            self.status_tracker.transition(scene_id, SceneStatus.GEOSPATIALIZED)
            events = self.geospatializer.geospatial_events_from_components(
                components=components,
                geo_transform=geo_transform,
                scene_id=scene_id,
                acquisition_time=scene.acquisition_time,
            )
            if events:
                self.repo.save_oil_spill_events(events)

            # 6. ALERTING (FR-33)
            alerts = self.alert_engine.evaluate_events(events)
            for alert in alerts:
                self.repo.save_alert(alert)

            # 7. INVESTIGATION REPORTING (FR-35, FR-36)
            html_report, json_report = self.report_generator.save_report(
                scene=scene,
                events=events,
                alerts=alerts,
                raw_img=norm_img,
                prob_map=prob_map,
                binary_mask=binary_mask,
            )

            # 8. FINAL STATUS
            final_status = SceneStatus.COMPLETED if events else SceneStatus.NO_SPILL_DETECTED
            self.status_tracker.transition(scene_id, final_status)

            logger.info(f"=== Successfully Completed Pipeline for {scene_id}: {len(events)} spills, {len(alerts)} alerts ===")
            return {
                "scene_id": scene_id,
                "status": final_status.value,
                "spills_detected": len(events),
                "alerts_triggered": len(alerts),
                "events": [e.dict() for e in events],
                "alerts": [a.dict() for a in alerts],
                "report_html": str(html_report),
                "report_json": str(json_report),
            }

        except Exception as e:
            logger.error(f"Pipeline failure for scene {scene_id}: {str(e)}", exc_info=True)
            self.status_tracker.transition(scene_id, SceneStatus.FAILED_PROCESSING, error_message=str(e))
            return {
                "scene_id": scene_id,
                "status": SceneStatus.FAILED_PROCESSING.value,
                "error": str(e),
            }


_pipeline_worker_instance: Optional[PipelineWorker] = None


def get_pipeline_worker() -> PipelineWorker:
    """Lazy initialization of PipelineWorker singleton."""
    global _pipeline_worker_instance
    if _pipeline_worker_instance is None:
        _pipeline_worker_instance = PipelineWorker()
    return _pipeline_worker_instance


class AsyncSceneWorkerPool:
    """Multi-threaded background worker pool for concurrent scene processing."""

    def __init__(self, max_workers: int = 2):
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._worker: Optional[PipelineWorker] = None

    @property
    def worker(self) -> PipelineWorker:
        if self._worker is None:
            self._worker = get_pipeline_worker()
        return self._worker

    def submit_scene_job(self, scene_id: str, force: bool = False):
        """Submit scene processing job to background pool."""
        return self.executor.submit(self.worker.process_scene, scene_id, force)

