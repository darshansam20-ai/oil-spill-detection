"""
Geospatial Event Generation Module (PRD Section 4.5, FR-21 to FR-25).
Transforms segmentation pixel contours into geographic Polygons, calculates geodesic surface area,
determines centroid and bounding box, and creates unique Oil Spill Event records.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import json
import numpy as np
from shapely.geometry import Polygon, MultiPolygon, mapping
from shapely.validation import make_valid
from shapely.ops import transform as shapely_transform

from src.config.constants import CURRENT_MODEL_VERSION
from src.geospatial.area_calculator import calculate_geometry_area
from src.postprocessing.mask_processor import SpillComponent
from src.preprocessing.georeference import GeoreferenceTransform
from src.storage.models import OilSpillEvent
from src.utils.id_generator import generate_event_id
from src.utils.logger import get_logger

logger = get_logger("geospatial.geospatializer")


class SpillGeospatializer:
    """Transforms image-space spill regions into validated geographic OilSpillEvent objects."""

    def __init__(self, model_version: str = CURRENT_MODEL_VERSION, threshold: float = 0.5):
        self.model_version = model_version
        self.threshold = threshold

    def create_event_from_component(
        self,
        component: SpillComponent,
        geo_transform: GeoreferenceTransform,
        scene_id: str,
        acquisition_time: datetime,
        index: int = 0,
    ) -> Optional[OilSpillEvent]:
        """
        Convert a SpillComponent into an auditable OilSpillEvent.
        """
        if not component.contours or len(component.contours) == 0:
            return None

        # Convert pixel contour coordinates [(col, row), ...] to geographic [(lon, lat), ...]
        polygons = []
        for contour in component.contours:
            if len(contour) < 3:
                continue
            geo_coords = []
            for pt in contour:
                c, r = float(pt[0]), float(pt[1])
                lon, lat = geo_transform.pixel_to_geo(row=r, col=c)
                geo_coords.append((lon, lat))
            
            # Ensure closed ring
            if geo_coords[0] != geo_coords[-1]:
                geo_coords.append(geo_coords[0])

            try:
                poly = Polygon(geo_coords)
                if not poly.is_valid:
                    poly = make_valid(poly)
                if not poly.is_empty and poly.area > 0:
                    polygons.append(poly)
            except Exception as e:
                logger.warning(f"Error creating polygon from contour: {e}")

        if not polygons:
            return None

        # Combine into single Polygon or MultiPolygon
        if len(polygons) == 1:
            geom = polygons[0]
        else:
            geom = MultiPolygon(polygons)
            if not geom.is_valid:
                geom = make_valid(geom)

        # 1. Centroid (FR-22)
        c_r, c_c = component.centroid_pixel
        c_lon, c_lat = geo_transform.pixel_to_geo(row=c_r, col=c_c)

        # 2. Bounding Box [min_lon, min_lat, max_lon, max_lat] (FR-22)
        min_r, min_c, max_r, max_c = component.bbox_pixel
        min_lon, max_lat = geo_transform.pixel_to_geo(row=min_r, col=min_c)
        max_lon, min_lat = geo_transform.pixel_to_geo(row=max_r, col=max_c)
        bbox = [
            round(min(min_lon, max_lon), 6),
            round(min(min_lat, max_lat), 6),
            round(max(min_lon, max_lon), 6),
            round(max(min_lat, max_lat), 6),
        ]

        # 3. Geodesic Area Calculation (FR-23)
        area_km2, area_m2 = calculate_geometry_area(geom)

        # 4. Generate Unique Event ID (FR-25)
        event_id = generate_event_id(scene_id=scene_id, timestamp=acquisition_time, index=index)

        return OilSpillEvent(
            event_id=event_id,
            scene_id=scene_id,
            timestamp=acquisition_time,
            centroid_lat=round(c_lat, 6),
            centroid_lon=round(c_lon, 6),
            polygon=mapping(geom),
            bounding_box=bbox,
            area_km2=round(area_km2, 4),
            area_m2=round(area_m2, 2),
            confidence=round(component.mean_confidence, 4),
            peak_confidence=round(component.peak_confidence, 4),
            model_version=self.model_version,
            threshold=self.threshold,
        )

    def geospatial_events_from_components(
        self,
        components: List[SpillComponent],
        geo_transform: GeoreferenceTransform,
        scene_id: str,
        acquisition_time: datetime,
    ) -> List[OilSpillEvent]:
        """
        Transform all detected spill components into geographic OilSpillEvent records.
        """
        events = []
        for idx, comp in enumerate(components):
            evt = self.create_event_from_component(
                component=comp,
                geo_transform=geo_transform,
                scene_id=scene_id,
                acquisition_time=acquisition_time,
                index=idx,
            )
            if evt is not None:
                events.append(evt)

        logger.info(f"Generated {len(events)} geographic OilSpillEvent records for scene {scene_id}.")
        return events
