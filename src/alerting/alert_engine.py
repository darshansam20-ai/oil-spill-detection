"""
Alerting Engine (PRD Section 4.7, FR-33).
Evaluates detected oil spill events against configurable business rules (confidence, surface area),
classifies severity, and triggers alert records for human operator investigation.
"""
from typing import List, Optional
from src.config.constants import (
    AlertSeverity,
    ReviewStatus,
    CRITICAL_AREA_KM2_THRESHOLD,
    HIGH_AREA_KM2_THRESHOLD,
    MEDIUM_AREA_KM2_THRESHOLD,
    DEFAULT_ALERT_CONFIDENCE_THRESHOLD,
    DEFAULT_ALERT_AREA_KM2_THRESHOLD,
)
from src.config.settings import settings
from src.storage.models import OilSpillEvent, Alert
from src.utils.id_generator import generate_alert_id
from src.utils.logger import get_logger

logger = get_logger("alerting.alert_engine")


class AlertEngine:
    """Evaluates OilSpillEvent records and generates investigation alerts."""

    def __init__(
        self,
        min_confidence: float = DEFAULT_ALERT_CONFIDENCE_THRESHOLD,
        min_area_km2: float = DEFAULT_ALERT_AREA_KM2_THRESHOLD,
    ):
        self.min_confidence = min_confidence
        self.min_area_km2 = min_area_km2

    def determine_severity(self, area_km2: float, confidence: float) -> AlertSeverity:
        """
        Classify alert severity based on physical spill surface area and model confidence.
        """
        if area_km2 >= CRITICAL_AREA_KM2_THRESHOLD or (area_km2 >= HIGH_AREA_KM2_THRESHOLD and confidence >= 0.80):
            return AlertSeverity.CRITICAL
        elif area_km2 >= HIGH_AREA_KM2_THRESHOLD or (area_km2 >= MEDIUM_AREA_KM2_THRESHOLD and confidence >= 0.70):
            return AlertSeverity.HIGH
        elif area_km2 >= MEDIUM_AREA_KM2_THRESHOLD or confidence >= self.min_confidence:
            return AlertSeverity.MEDIUM
        else:
            return AlertSeverity.LOW

    def evaluate_event(self, event: OilSpillEvent) -> Optional[Alert]:
        """
        Evaluate an individual spill event against alert criteria.
        
        Returns:
            Alert instance if criteria met, None otherwise.
        """
        # Check if event meets threshold criteria
        meets_area = event.area_km2 >= self.min_area_km2
        meets_conf = event.confidence >= self.min_confidence

        if meets_area or meets_conf:
            severity = self.determine_severity(event.area_km2, event.confidence)
            alert = Alert(
                alert_id=generate_alert_id(event.event_id),
                event_id=event.event_id,
                scene_id=event.scene_id,
                severity=severity,
                status=ReviewStatus.NEW,
                confidence=event.confidence,
                area_km2=event.area_km2,
            )
            logger.info(f"Generated {severity.value} alert {alert.alert_id} for spill event {event.event_id} (Area: {event.area_km2:.3f} km², Conf: {event.confidence:.2f})")
            return alert
        return None

    def evaluate_events(self, events: List[OilSpillEvent]) -> List[Alert]:
        """
        Evaluate a batch of spill events, returning triggered alerts.
        """
        alerts = []
        for event in events:
            alert = self.evaluate_event(event)
            if alert:
                alerts.append(alert)
        return alerts
