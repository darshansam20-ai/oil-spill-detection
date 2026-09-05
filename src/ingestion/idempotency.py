"""
Idempotency and duplicate processing prevention (FR-04).
Ensures scenes are only processed once unless explicitly forced.
"""
from typing import Optional
from src.config.constants import SceneStatus
from src.storage.repository import DatabaseRepository, repo
from src.utils.id_generator import generate_scene_hash
from src.utils.logger import get_logger

logger = get_logger("ingestion.idempotency")


class IdempotencyManager:
    """Guarantees duplicate-prevention and idempotent pipeline execution."""

    def __init__(self, repository: Optional[DatabaseRepository] = None):
        self.repo = repository or repo

    def is_already_processed(self, scene_id: str) -> bool:
        """
        Check if a scene has already completed processing.
        
        Returns:
            True if scene is already COMPLETED or NO_SPILL_DETECTED.
        """
        scene = self.repo.get_scene(scene_id)
        if not scene:
            return False
        return scene.status in [SceneStatus.COMPLETED, SceneStatus.NO_SPILL_DETECTED]

    def should_process(self, scene_id: str, force: bool = False) -> bool:
        """
        Determine whether a scene should be dispatched for processing.
        """
        if force:
            logger.info(f"Force re-processing requested for scene {scene_id}.")
            return True
        if self.is_already_processed(scene_id):
            logger.info(f"Scene {scene_id} already processed. Skipping duplicate execution.")
            return False
        return True
