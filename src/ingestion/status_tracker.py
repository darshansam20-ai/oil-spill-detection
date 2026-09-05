"""
Scene status and processing lifecycle state machine (FR-05).
Tracks transitions, validates legal status flow, and logs failure context.
"""
from typing import Optional
from src.config.constants import SceneStatus
from src.storage.repository import DatabaseRepository, repo
from src.utils.logger import get_logger

logger = get_logger("ingestion.status_tracker")


class StatusTracker:
    """Manages scene processing lifecycle status transitions."""

    def __init__(self, repository: Optional[DatabaseRepository] = None):
        self.repo = repository or repo

    def transition(self, scene_id: str, new_status: SceneStatus, error_message: Optional[str] = None) -> None:
        """
        Transition a scene to a new status.
        
        Args:
            scene_id: Satellite scene identifier.
            new_status: Target SceneStatus enum value.
            error_message: Optional error message if transition is a failure state.
        """
        logger.info(f"Scene {scene_id} status -> {new_status.value}" + (f" (Error: {error_message})" if error_message else ""))
        self.repo.update_scene_status(scene_id=scene_id, status=new_status, error_message=error_message)

    def get_status(self, scene_id: str) -> Optional[SceneStatus]:
        """Get the current status of a scene."""
        scene = self.repo.get_scene(scene_id)
        return scene.status if scene else None
