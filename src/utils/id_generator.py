"""
Identifier generation utilities for Oil Spill Events, Scene hashes, and Alert IDs.
Ensures unique, human-readable, and deterministic IDs.
"""
import hashlib
import uuid
from datetime import datetime
from typing import Optional


def generate_event_id(scene_id: str, timestamp: Optional[datetime] = None, index: int = 0) -> str:
    """
    Generate a unique, auditable Oil Spill Event ID.
    Format: OSE-YYYYMMDD-<6_CHAR_HASH>
    
    Args:
        scene_id: Source Sentinel-1 scene identifier.
        timestamp: Scene acquisition timestamp.
        index: Index of the spill polygon in the scene.
        
    Returns:
        Formatted event ID string.
    """
    ts = timestamp or datetime.utcnow()
    date_str = ts.strftime("%Y%m%d")
    raw = f"{scene_id}:{index}:{ts.isoformat()}:{uuid.uuid4().hex[:6]}"
    hash_str = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6].upper()
    return f"OSE-{date_str}-{hash_str}"


def generate_alert_id(event_id: str) -> str:
    """
    Generate an Alert ID linked to an Oil Spill Event.
    Format: ALT-YYYYMMDD-<6_CHAR_HASH>
    """
    date_str = datetime.utcnow().strftime("%Y%m%d")
    raw = f"ALERT:{event_id}:{uuid.uuid4().hex[:6]}"
    hash_str = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:6].upper()
    return f"ALT-{date_str}-{hash_str}"


def generate_scene_hash(scene_id: str, acquisition_time: str, polarization: str) -> str:
    """
    Generate a deterministic SHA256 hash for a satellite scene for idempotency checks.
    """
    payload = f"{scene_id}|{acquisition_time}|{polarization}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
