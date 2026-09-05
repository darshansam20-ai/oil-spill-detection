"""
Core Data Models for Oil Spill Detection System (PRD Section 8 - Strictly without AIS).
Implements Pydantic and SQLAlchemy ORM models.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    DateTime,
    Text,
    Enum as SQLEnum,
    create_engine,
)
from sqlalchemy.orm import declarative_base

from src.config.constants import SceneStatus, AlertSeverity, ReviewStatus, Polarization, OrbitDirection

Base = declarative_base()


# ==========================================
# SQLAlchemy ORM Models (Persistence Layer)
# ==========================================

class DBSatelliteScene(Base):
    """Tracks each Sentinel-1 SAR input scene."""
    __tablename__ = "satellite_scenes"

    scene_id = Column(String(128), primary_key=True, index=True)
    acquisition_time = Column(DateTime, nullable=False, index=True)
    footprint_geojson = Column(Text, nullable=True)  # JSON string of GeoJSON footprint polygon
    polarization = Column(String(16), default=Polarization.VV.value)
    orbit_direction = Column(String(32), default=OrbitDirection.ASCENDING.value)
    orbit_number = Column(Integer, nullable=True)
    source_url = Column(String(512), nullable=True)
    local_path = Column(String(512), nullable=True)
    status = Column(String(32), default=SceneStatus.DISCOVERED.value, index=True)
    error_message = Column(Text, nullable=True)
    ingested_at = Column(DateTime, default=datetime.utcnow)
    processed_at = Column(DateTime, nullable=True)


class DBSpillMask(Base):
    """Stores segmentation output artifacts for a scene."""
    __tablename__ = "spill_masks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(String(128), index=True, nullable=False)
    mask_path = Column(String(512), nullable=False)
    prob_map_path = Column(String(512), nullable=False)
    threshold = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)
    num_spills_detected = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)


class DBOilSpillEvent(Base):
    """Represents a detected oil spill event."""
    __tablename__ = "oil_spill_events"

    event_id = Column(String(64), primary_key=True, index=True)
    scene_id = Column(String(128), index=True, nullable=False)
    timestamp = Column(DateTime, nullable=False, index=True)
    centroid_lat = Column(Float, nullable=False)
    centroid_lon = Column(Float, nullable=False)
    polygon_geojson = Column(Text, nullable=False)  # GeoJSON Geometry string
    bbox_geojson = Column(String(128), nullable=False)  # min_lon, min_lat, max_lon, max_lat
    area_km2 = Column(Float, nullable=False)
    area_m2 = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    peak_confidence = Column(Float, nullable=False)
    model_version = Column(String(64), nullable=False)
    threshold = Column(Float, nullable=False)
    status = Column(String(32), default=ReviewStatus.NEW.value, index=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class DBAlert(Base):
    """Alert record for human investigation workflow."""
    __tablename__ = "alerts"

    alert_id = Column(String(64), primary_key=True, index=True)
    event_id = Column(String(64), index=True, nullable=False)
    scene_id = Column(String(128), index=True, nullable=False)
    severity = Column(String(32), default=AlertSeverity.MEDIUM.value, index=True)
    status = Column(String(32), default=ReviewStatus.NEW.value, index=True)
    confidence = Column(Float, nullable=False)
    area_km2 = Column(Float, nullable=False)
    reviewer_notes = Column(Text, nullable=True)
    reviewed_by = Column(String(128), nullable=True)
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


# ==========================================
# Pydantic Schemas (API & Processing Layer)
# ==========================================

class SatelliteScene(BaseModel):
    """Pydantic schema for SatelliteScene."""
    model_config = ConfigDict(from_attributes=True)

    scene_id: str
    acquisition_time: datetime
    geometry: Optional[Dict[str, Any]] = None  # GeoJSON polygon dict
    polarization: str = Polarization.VV.value
    orbit_direction: str = OrbitDirection.ASCENDING.value
    orbit_number: Optional[int] = None
    source_url: Optional[str] = None
    local_path: Optional[str] = None
    status: SceneStatus = SceneStatus.DISCOVERED
    error_message: Optional[str] = None
    ingested_at: datetime = Field(default_factory=datetime.utcnow)
    processed_at: Optional[datetime] = None


class SpillMask(BaseModel):
    """Pydantic schema for SpillMask."""
    model_config = ConfigDict(from_attributes=True)

    scene_id: str
    mask_path: str
    prob_map_path: str
    threshold: float
    model_version: str
    num_spills_detected: int = 0
    created_at: datetime = Field(default_factory=datetime.utcnow)


class OilSpillEvent(BaseModel):
    """Pydantic schema for OilSpillEvent."""
    model_config = ConfigDict(from_attributes=True)

    event_id: str
    scene_id: str
    timestamp: datetime
    centroid_lat: float
    centroid_lon: float
    polygon: Dict[str, Any]  # GeoJSON Geometry Polygon/MultiPolygon
    bounding_box: List[float]  # [min_lon, min_lat, max_lon, max_lat]
    area_km2: float
    area_m2: float
    confidence: float
    peak_confidence: float
    model_version: str
    threshold: float
    status: ReviewStatus = ReviewStatus.NEW
    created_at: datetime = Field(default_factory=datetime.utcnow)


class Alert(BaseModel):
    """Pydantic schema for Alert."""
    model_config = ConfigDict(from_attributes=True)

    alert_id: str
    event_id: str
    scene_id: str
    severity: AlertSeverity
    status: ReviewStatus = ReviewStatus.NEW
    confidence: float
    area_km2: float
    reviewer_notes: Optional[str] = None
    reviewed_by: Optional[str] = None
    reviewed_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
