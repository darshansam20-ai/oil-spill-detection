"""
Database repository and persistence layer for satellite scenes, masks, events, and alerts.
"""
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from src.config.settings import settings
from src.config.constants import SceneStatus, ReviewStatus, AlertSeverity
from src.storage.models import (
    Base,
    DBSatelliteScene,
    DBSpillMask,
    DBOilSpillEvent,
    DBAlert,
    SatelliteScene,
    SpillMask,
    OilSpillEvent,
    Alert,
)


class DatabaseRepository:
    """Thread-safe SQLite/PostgreSQL database repository."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or settings.paths.db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_engine(f"sqlite:///{self.db_path}", connect_args={"check_same_thread": False})
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.init_db()

    def init_db(self) -> None:
        """Create all database tables."""
        Base.metadata.create_all(bind=self.engine)

    def get_session(self) -> Session:
        """Get a database session."""
        return self.SessionLocal()

    # --- Satellite Scenes ---

    def create_or_update_scene(self, scene: SatelliteScene) -> SatelliteScene:
        with self.get_session() as session:
            db_scene = session.query(DBSatelliteScene).filter_by(scene_id=scene.scene_id).first()
            if not db_scene:
                db_scene = DBSatelliteScene(
                    scene_id=scene.scene_id,
                    acquisition_time=scene.acquisition_time,
                    footprint_geojson=json.dumps(scene.geometry) if scene.geometry else None,
                    polarization=scene.polarization,
                    orbit_direction=scene.orbit_direction,
                    orbit_number=scene.orbit_number,
                    source_url=scene.source_url,
                    local_path=scene.local_path,
                    status=scene.status.value,
                    error_message=scene.error_message,
                    ingested_at=scene.ingested_at,
                    processed_at=scene.processed_at,
                )
                session.add(db_scene)
            else:
                db_scene.status = scene.status.value
                db_scene.error_message = scene.error_message
                if scene.local_path:
                    db_scene.local_path = scene.local_path
                if scene.processed_at:
                    db_scene.processed_at = scene.processed_at
                if scene.geometry:
                    db_scene.footprint_geojson = json.dumps(scene.geometry)
            session.commit()
            session.refresh(db_scene)
            return self._to_pydantic_scene(db_scene)

    def get_scene(self, scene_id: str) -> Optional[SatelliteScene]:
        with self.get_session() as session:
            db_scene = session.query(DBSatelliteScene).filter_by(scene_id=scene_id).first()
            return self._to_pydantic_scene(db_scene) if db_scene else None

    def list_scenes(self, status: Optional[str] = None, limit: int = 100) -> List[SatelliteScene]:
        with self.get_session() as session:
            query = session.query(DBSatelliteScene)
            if status:
                query = query.filter_by(status=status)
            db_scenes = query.order_by(DBSatelliteScene.acquisition_time.desc()).limit(limit).all()
            return [self._to_pydantic_scene(s) for s in db_scenes]

    def update_scene_status(self, scene_id: str, status: SceneStatus, error_message: Optional[str] = None) -> None:
        with self.get_session() as session:
            db_scene = session.query(DBSatelliteScene).filter_by(scene_id=scene_id).first()
            if db_scene:
                db_scene.status = status.value
                db_scene.error_message = error_message
                if status in [SceneStatus.COMPLETED, SceneStatus.NO_SPILL_DETECTED]:
                    db_scene.processed_at = datetime.utcnow()
                session.commit()

    # --- Spill Masks ---

    def save_spill_mask(self, mask: SpillMask) -> SpillMask:
        with self.get_session() as session:
            db_mask = DBSpillMask(
                scene_id=mask.scene_id,
                mask_path=mask.mask_path,
                prob_map_path=mask.prob_map_path,
                threshold=mask.threshold,
                model_version=mask.model_version,
                num_spills_detected=mask.num_spills_detected,
                created_at=mask.created_at,
            )
            session.add(db_mask)
            session.commit()
            session.refresh(db_mask)
            return mask

    def get_spill_mask(self, scene_id: str) -> Optional[SpillMask]:
        with self.get_session() as session:
            db_mask = session.query(DBSpillMask).filter_by(scene_id=scene_id).order_by(DBSpillMask.created_at.desc()).first()
            if not db_mask:
                return None
            return SpillMask(
                scene_id=db_mask.scene_id,
                mask_path=db_mask.mask_path,
                prob_map_path=db_mask.prob_map_path,
                threshold=db_mask.threshold,
                model_version=db_mask.model_version,
                num_spills_detected=db_mask.num_spills_detected,
                created_at=db_mask.created_at,
            )

    # --- Oil Spill Events ---

    def save_oil_spill_events(self, events: List[OilSpillEvent]) -> None:
        with self.get_session() as session:
            for event in events:
                db_event = DBOilSpillEvent(
                    event_id=event.event_id,
                    scene_id=event.scene_id,
                    timestamp=event.timestamp,
                    centroid_lat=event.centroid_lat,
                    centroid_lon=event.centroid_lon,
                    polygon_geojson=json.dumps(event.polygon),
                    bbox_geojson=json.dumps(event.bounding_box),
                    area_km2=event.area_km2,
                    area_m2=event.area_m2,
                    confidence=event.confidence,
                    peak_confidence=event.peak_confidence,
                    model_version=event.model_version,
                    threshold=event.threshold,
                    status=event.status.value,
                    created_at=event.created_at,
                )
                session.merge(db_event)
            session.commit()

    def get_event(self, event_id: str) -> Optional[OilSpillEvent]:
        with self.get_session() as session:
            db_event = session.query(DBOilSpillEvent).filter_by(event_id=event_id).first()
            return self._to_pydantic_event(db_event) if db_event else None

    def list_events(self, scene_id: Optional[str] = None, min_confidence: Optional[float] = None, limit: int = 100) -> List[OilSpillEvent]:
        with self.get_session() as session:
            query = session.query(DBOilSpillEvent)
            if scene_id:
                query = query.filter_by(scene_id=scene_id)
            if min_confidence is not None:
                query = query.filter(DBOilSpillEvent.confidence >= min_confidence)
            db_events = query.order_by(DBOilSpillEvent.timestamp.desc()).limit(limit).all()
            return [self._to_pydantic_event(e) for e in db_events]

    # --- Alerts ---

    def save_alert(self, alert: Alert) -> None:
        with self.get_session() as session:
            db_alert = DBAlert(
                alert_id=alert.alert_id,
                event_id=alert.event_id,
                scene_id=alert.scene_id,
                severity=alert.severity.value,
                status=alert.status.value,
                confidence=alert.confidence,
                area_km2=alert.area_km2,
                reviewer_notes=alert.reviewer_notes,
                reviewed_by=alert.reviewed_by,
                reviewed_at=alert.reviewed_at,
                created_at=alert.created_at,
            )
            session.merge(db_alert)
            session.commit()

    def list_alerts(self, severity: Optional[str] = None, status: Optional[str] = None, limit: int = 100) -> List[Alert]:
        with self.get_session() as session:
            query = session.query(DBAlert)
            if severity:
                query = query.filter_by(severity=severity)
            if status:
                query = query.filter_by(status=status)
            db_alerts = query.order_by(DBAlert.created_at.desc()).limit(limit).all()
            return [self._to_pydantic_alert(a) for a in db_alerts]

    def update_alert_review(self, alert_id: str, status: ReviewStatus, notes: Optional[str] = None, reviewer: Optional[str] = None) -> Optional[Alert]:
        with self.get_session() as session:
            db_alert = session.query(DBAlert).filter_by(alert_id=alert_id).first()
            if not db_alert:
                return None
            db_alert.status = status.value
            if notes:
                db_alert.reviewer_notes = notes
            if reviewer:
                db_alert.reviewed_by = reviewer
            db_alert.reviewed_at = datetime.utcnow()

            # Sync with underlying event
            db_event = session.query(DBOilSpillEvent).filter_by(event_id=db_alert.event_id).first()
            if db_event:
                db_event.status = status.value

            session.commit()
            session.refresh(db_alert)
            return self._to_pydantic_alert(db_alert)

    # --- Helper converters ---

    def _to_pydantic_scene(self, s: DBSatelliteScene) -> SatelliteScene:
        return SatelliteScene(
            scene_id=s.scene_id,
            acquisition_time=s.acquisition_time,
            geometry=json.loads(s.footprint_geojson) if s.footprint_geojson else None,
            polarization=s.polarization,
            orbit_direction=s.orbit_direction,
            orbit_number=s.orbit_number,
            source_url=s.source_url,
            local_path=s.local_path,
            status=SceneStatus(s.status),
            error_message=s.error_message,
            ingested_at=s.ingested_at,
            processed_at=s.processed_at,
        )

    def _to_pydantic_event(self, e: DBOilSpillEvent) -> OilSpillEvent:
        return OilSpillEvent(
            event_id=e.event_id,
            scene_id=e.scene_id,
            timestamp=e.timestamp,
            centroid_lat=e.centroid_lat,
            centroid_lon=e.centroid_lon,
            polygon=json.loads(e.polygon_geojson),
            bounding_box=json.loads(e.bbox_geojson),
            area_km2=e.area_km2,
            area_m2=e.area_m2,
            confidence=e.confidence,
            peak_confidence=e.peak_confidence,
            model_version=e.model_version,
            threshold=e.threshold,
            status=ReviewStatus(e.status),
            created_at=e.created_at,
        )

    def _to_pydantic_alert(self, a: DBAlert) -> Alert:
        return Alert(
            alert_id=a.alert_id,
            event_id=a.event_id,
            scene_id=a.scene_id,
            severity=AlertSeverity(a.severity),
            status=ReviewStatus(a.status),
            confidence=a.confidence,
            area_km2=a.area_km2,
            reviewer_notes=a.reviewer_notes,
            reviewed_by=a.reviewed_by,
            reviewed_at=a.reviewed_at,
            created_at=a.created_at,
        )


# Singleton repository instance
repo = DatabaseRepository()
