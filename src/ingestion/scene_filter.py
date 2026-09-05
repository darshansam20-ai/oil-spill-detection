"""
Scene Filtering module (PRD Section 4.2, FR-06 to FR-09).
Applies Area of Interest (AOI), temporal, orbit/pass, polarization, and product metadata validation.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from shapely.geometry import shape, box, Polygon
from src.config.constants import Polarization, OrbitDirection
from src.storage.models import SatelliteScene
from src.utils.logger import get_logger

logger = get_logger("ingestion.scene_filter")


class SceneFilter:
    """Filters candidate Sentinel-1 scenes against configured spatial and temporal rules."""

    def __init__(
        self,
        aoi_polygon: Optional[Polygon] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        required_polarization: str = Polarization.VV.value,
        required_orbit_direction: str = OrbitDirection.ANY.value,
    ):
        self.aoi_polygon = aoi_polygon
        self.start_date = start_date
        self.end_date = end_date
        self.required_polarization = required_polarization
        self.required_orbit_direction = required_orbit_direction

    def filter_scene(self, scene: SatelliteScene) -> bool:
        """
        Evaluate all filter criteria for a scene.
        
        Returns:
            True if scene passes all filters, False otherwise.
        """
        # 1. Product Metadata Validation (FR-09)
        if not scene.scene_id:
            logger.warning(f"Scene validation failed: Missing scene_id.")
            return False

        # 2. Polarization Filtering (FR-08)
        if self.required_polarization and self.required_polarization != "ANY":
            if self.required_polarization not in scene.polarization:
                logger.info(f"Scene {scene.scene_id} rejected by polarization filter: {scene.polarization} != {self.required_polarization}")
                return False

        # 3. Orbit Direction Filtering (FR-08)
        if self.required_orbit_direction and self.required_orbit_direction != OrbitDirection.ANY.value:
            if scene.orbit_direction != self.required_orbit_direction:
                logger.info(f"Scene {scene.scene_id} rejected by orbit filter: {scene.orbit_direction} != {self.required_orbit_direction}")
                return False

        # 4. Temporal Filtering (FR-07)
        if self.start_date and scene.acquisition_time < self.start_date:
            logger.info(f"Scene {scene.scene_id} rejected by start date filter: {scene.acquisition_time} < {self.start_date}")
            return False
        if self.end_date and scene.acquisition_time > self.end_date:
            logger.info(f"Scene {scene.scene_id} rejected by end date filter: {scene.acquisition_time} > {self.end_date}")
            return False

        # 5. Spatial / AOI Filtering (FR-06)
        if self.aoi_polygon and scene.geometry:
            scene_geom = shape(scene.geometry)
            if not self.aoi_polygon.intersects(scene_geom):
                logger.info(f"Scene {scene.scene_id} rejected: Does not intersect configured AOI.")
                return False

        return True

    def filter_batch(self, scenes: List[SatelliteScene]) -> List[SatelliteScene]:
        """Filter a list of scenes, returning only those that pass criteria."""
        passed = [s for s in scenes if self.filter_scene(s)]
        logger.info(f"Filtered {len(scenes)} scenes -> {len(passed)} passed criteria.")
        return passed
