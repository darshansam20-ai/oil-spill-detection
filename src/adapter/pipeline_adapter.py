"""
Pipeline Adapter Layer.
Bridges the semantic gap between Sentinel-1 Oil Spill Detection output and AIS Vessel Correlation input.

Functionality:
1. Receives `DetectionResult` from `Sentinel1OilSpillDetector`.
2. Evaluates detected oil spill components and identifies the primary epicenter (highest confidence / largest area).
3. Normalizes date and time strings into standard ISO-8601 timestamps (`YYYY-MM-DDTHH:MM:SSZ`).
4. Determines the spatial search radius (in km) dynamically or via configuration.
5. Provides fallback mechanisms with clear informational notices when running in image-only mode without geographic metadata.
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from src.inference.oil_spill_detector import DetectionResult, OilSpillDetection
from src.utils.logger import get_logger

logger = get_logger("adapter.pipeline_adapter")


@dataclass
class AISInputPayload:
    """Exact structured input expected by the AIS Vessel Correlator."""
    spill_latitude: float
    spill_longitude: float
    detection_time: str  # ISO-8601 format: YYYY-MM-DDTHH:MM:SSZ
    search_radius_km: float
    spill_id: Optional[int] = None
    spill_area_km2: Optional[float] = None
    confidence: Optional[float] = None
    is_fallback_coordinates: bool = False
    notes: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "spill_latitude": round(self.spill_latitude, 6),
            "spill_longitude": round(self.spill_longitude, 6),
            "detection_time": self.detection_time,
            "search_radius_km": self.search_radius_km,
            "spill_id": self.spill_id,
            "spill_area_km2": round(self.spill_area_km2, 4) if self.spill_area_km2 is not None else None,
            "confidence": round(self.confidence, 4) if self.confidence is not None else None,
            "is_fallback_coordinates": self.is_fallback_coordinates,
            "notes": self.notes,
        }


class PipelineAdapter:
    """
    Production-grade adapter that transforms DetectionResult into AISInputPayload.
    """

    def __init__(
        self,
        default_search_radius_km: float = 20.0,
        default_fallback_lat: float = 18.90,
        default_fallback_lon: float = 72.50,
    ):
        self.default_search_radius_km = default_search_radius_km
        self.default_fallback_lat = default_fallback_lat
        self.default_fallback_lon = default_fallback_lon

    def _format_iso_timestamp(
        self,
        date_str: Optional[str],
        time_str: Optional[str],
    ) -> str:
        """
        Convert date string (e.g. '2018-12-19') and time string (e.g. '06:15:22 UTC')
        into standard ISO-8601 string: 'YYYY-MM-DDTHH:MM:SSZ'.
        """
        if not date_str:
            return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

        # Clean date string
        clean_date = date_str.strip().split("T")[0]

        # Clean time string
        clean_time = "12:00:00"
        if time_str:
            raw_t = time_str.strip().upper().replace("UTC", "").replace("Z", "").strip()
            parts = raw_t.split(":")
            if len(parts) == 3:
                clean_time = f"{int(parts[0]):02d}:{int(parts[1]):02d}:{int(float(parts[2])):02d}"
            elif len(parts) == 2:
                clean_time = f"{int(parts[0]):02d}:{int(parts[1]):02d}:00"

        # Combine into ISO format
        try:
            dt = datetime.fromisoformat(f"{clean_date}T{clean_time}+00:00")
            return dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except Exception:
            return f"{clean_date}T{clean_time}Z"

    def convert_detection_to_ais_input(
        self,
        detection_result: DetectionResult,
        spill_id: Optional[int] = None,
        custom_radius_km: Optional[float] = None,
        override_lat: Optional[float] = None,
        override_lon: Optional[float] = None,
        override_time: Optional[str] = None,
    ) -> AISInputPayload:
        """
        Convert DetectionResult into AISInputPayload.
        
        Args:
            detection_result: Output from Sentinel1OilSpillDetector.
            spill_id: Optional specific spill ID to correlate (defaults to highest confidence / largest spill).
            custom_radius_km: Optional custom search radius override.
            override_lat: Optional override latitude.
            override_lon: Optional override longitude.
            override_time: Optional override timestamp.
            
        Returns:
            AISInputPayload ready for AISCorrelator.
        """
        logger.info("Executing adapter conversion: DetectionResult -> AISInputPayload")

        # 1. Check if user explicitly overrode coordinates
        if override_lat is not None and override_lon is not None:
            iso_time = override_time or self._format_iso_timestamp(
                detection_result.acquisition_date,
                detection_result.acquisition_time,
            )
            return AISInputPayload(
                spill_latitude=float(override_lat),
                spill_longitude=float(override_lon),
                detection_time=iso_time,
                search_radius_km=custom_radius_km or self.default_search_radius_km,
                is_fallback_coordinates=False,
                notes="User explicitly provided coordinate overrides.",
            )

        # 2. Extract best spill from detections
        target_spill: Optional[OilSpillDetection] = None
        if detection_result.spills:
            if spill_id is not None:
                for s in detection_result.spills:
                    if s.spill_id == spill_id:
                        target_spill = s
                        break
            if target_spill is None:
                # Rank by peak confidence * area
                target_spill = max(
                    detection_result.spills,
                    key=lambda s: (s.peak_confidence, s.area_km2 or s.pixel_area),
                )

        # 3. Determine Coordinates
        if target_spill is not None and target_spill.latitude is not None and target_spill.longitude is not None:
            spill_lat = target_spill.latitude
            spill_lon = target_spill.longitude
            is_fallback = False
            notes = f"Correlating Spill #{target_spill.spill_id} (Confidence: {target_spill.peak_confidence:.3f})"
        elif detection_result.aoi is not None:
            # Center of AOI
            min_lat = detection_result.aoi.get("min_latitude", self.default_fallback_lat)
            max_lat = detection_result.aoi.get("max_latitude", self.default_fallback_lat)
            min_lon = detection_result.aoi.get("min_longitude", self.default_fallback_lon)
            max_lon = detection_result.aoi.get("max_longitude", self.default_fallback_lon)
            spill_lat = (min_lat + max_lat) / 2.0
            spill_lon = (min_lon + max_lon) / 2.0
            is_fallback = False
            notes = "Spill coordinates derived from AOI center."
        else:
            # Fallback for image-only mode without georeference
            spill_lat = self.default_fallback_lat
            spill_lon = self.default_fallback_lon
            is_fallback = True
            notes = "Image-only input without metadata; utilizing reference demonstration coordinates."
            logger.info(notes)

        # 4. Resolve Timestamp
        if override_time:
            iso_time = override_time
        else:
            iso_time = self._format_iso_timestamp(
                detection_result.acquisition_date,
                detection_result.acquisition_time,
            )

        # 5. Determine Search Radius
        radius = custom_radius_km or self.default_search_radius_km
        if target_spill and target_spill.area_km2 and target_spill.area_km2 > 50:
            # Expand radius slightly for very large regional spills
            radius = max(radius, 30.0)

        payload = AISInputPayload(
            spill_latitude=spill_lat,
            spill_longitude=spill_lon,
            detection_time=iso_time,
            search_radius_km=radius,
            spill_id=target_spill.spill_id if target_spill else None,
            spill_area_km2=target_spill.area_km2 if target_spill else None,
            confidence=target_spill.peak_confidence if target_spill else None,
            is_fallback_coordinates=is_fallback,
            notes=notes,
        )

        logger.info(
            f"Adapter conversion successful: Epicenter=(Lat: {payload.spill_latitude:.4f}, "
            f"Lon: {payload.spill_longitude:.4f}), Time={payload.detection_time}, Radius={payload.search_radius_km} km"
        )
        return payload
