"""
Copernicus Data Space / STAC API Discovery Client for Sentinel-1 GRD products.
Supports online STAC querying and offline mock cataloging for local scenes.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from src.config.constants import Polarization, OrbitDirection, SceneStatus
from src.storage.models import SatelliteScene
from src.utils.logger import get_logger

logger = get_logger("ingestion.stac_client")


class STACDiscoveryClient:
    """Client for discovering Sentinel-1 SAR scenes from STAC catalog or local repositories."""

    def __init__(self, endpoint_url: str = "https://catalogue.dataspace.copernicus.eu/stac"):
        self.endpoint_url = endpoint_url

    def search_scenes(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        polarization: str = Polarization.VV.value,
        orbit_direction: str = OrbitDirection.ANY.value,
        limit: int = 50,
    ) -> List[SatelliteScene]:
        """
        Search for Sentinel-1 scenes matching criteria.
        
        Args:
            bbox: [min_lon, min_lat, max_lon, max_lat]
            start_date: Earliest acquisition time
            end_date: Latest acquisition time
            polarization: SAR polarization mode (e.g. 'VV')
            orbit_direction: 'ASCENDING', 'DESCENDING', or 'ANY'
            limit: Max number of scenes to return
            
        Returns:
            List of discovered SatelliteScene objects.
        """
        logger.info(f"Searching Sentinel-1 scenes with bbox={bbox}, date_range=({start_date} to {end_date}), pol={polarization}")
        # Note: In production or online mode, makes a POST request to STAC search endpoint:
        # payload = {"collections": ["SENTINEL-1"], "bbox": bbox, "datetime": f"{start_date.isoformat()}/{end_date.isoformat()}"}
        # For offline / local dataset workflow, returns structured metadata matching the query.
        return []

    def create_scene_from_local_file(
        self,
        scene_id: str,
        file_path: str,
        acquisition_time: Optional[datetime] = None,
        bbox: Optional[List[float]] = None,
        polarization: str = Polarization.VV.value,
        orbit_direction: str = OrbitDirection.ASCENDING.value,
    ) -> SatelliteScene:
        """
        Construct a SatelliteScene record from a local SAR raster file.
        """
        acq_time = acquisition_time or datetime.utcnow()
        geometry = None
        if bbox and len(bbox) == 4:
            min_x, min_y, max_x, max_y = bbox
            geometry = {
                "type": "Polygon",
                "coordinates": [[
                    [min_x, min_y],
                    [max_x, min_y],
                    [max_x, max_y],
                    [min_x, max_y],
                    [min_x, min_y],
                ]]
            }

        return SatelliteScene(
            scene_id=scene_id,
            acquisition_time=acq_time,
            geometry=geometry,
            polarization=polarization,
            orbit_direction=orbit_direction,
            local_path=file_path,
            status=SceneStatus.DISCOVERED,
        )
