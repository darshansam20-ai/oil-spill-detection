"""
Unit tests for Alerting and Investigation Reporting (FR-33 to FR-36).
"""
from datetime import datetime
import pytest

from src.config.constants import AlertSeverity, ReviewStatus
from src.alerting.alert_engine import AlertEngine
from src.alerting.workflow import HumanReviewWorkflow
from src.reporting.report_generator import InvestigationReportGenerator
from src.storage.models import SatelliteScene, OilSpillEvent
from src.storage.repository import DatabaseRepository


@pytest.fixture
def test_repo(tmp_path):
    db_file = tmp_path / "test_alerts.db"
    return DatabaseRepository(db_path=db_file)


def test_alert_engine_severity_classification():
    engine = AlertEngine(min_confidence=0.5, min_area_km2=0.1)

    event_critical = OilSpillEvent(
        event_id="OSE-CRIT",
        scene_id="SCENE_01",
        timestamp=datetime.utcnow(),
        centroid_lat=28.0,
        centroid_lon=-90.0,
        polygon={"type": "Polygon", "coordinates": []},
        bounding_box=[-90.1, 27.9, -89.9, 28.1],
        area_km2=6.5,
        area_m2=6_500_000,
        confidence=0.92,
        peak_confidence=0.99,
        model_version="v1.0",
        threshold=0.5,
    )

    alert_crit = engine.evaluate_event(event_critical)
    assert alert_crit is not None
    assert alert_crit.severity == AlertSeverity.CRITICAL


def test_human_review_workflow(test_repo):
    engine = AlertEngine()
    workflow = HumanReviewWorkflow(test_repo)

    event = OilSpillEvent(
        event_id="OSE-REVIEW-TEST",
        scene_id="SCENE_02",
        timestamp=datetime.utcnow(),
        centroid_lat=28.0,
        centroid_lon=-90.0,
        polygon={"type": "Polygon", "coordinates": []},
        bounding_box=[-90.1, 27.9, -89.9, 28.1],
        area_km2=1.2,
        area_m2=1_200_000,
        confidence=0.85,
        peak_confidence=0.92,
        model_version="v1.0",
        threshold=0.5,
    )
    test_repo.save_oil_spill_events([event])
    alert = engine.evaluate_event(event)
    test_repo.save_alert(alert)

    # Submit human confirmation
    updated = workflow.submit_review(
        alert_id=alert.alert_id,
        decision=ReviewStatus.CONFIRMED,
        reviewer_name="senior_operator",
        reviewer_notes="Verified by SAR texture analysis.",
    )

    assert updated.status == ReviewStatus.CONFIRMED
    assert updated.reviewed_by == "senior_operator"
    # Event status should also sync to CONFIRMED
    ev = test_repo.get_event("OSE-REVIEW-TEST")
    assert ev.status == ReviewStatus.CONFIRMED


def test_report_generator_html_and_json(test_repo, tmp_path):
    generator = InvestigationReportGenerator(test_repo)
    generator.reports_dir = tmp_path

    scene = SatelliteScene(
        scene_id="SCENE_RPT_TEST",
        acquisition_time=datetime(2020, 5, 12),
        polarization="VV",
    )
    event = OilSpillEvent(
        event_id="OSE-RPT-001",
        scene_id="SCENE_RPT_TEST",
        timestamp=datetime(2020, 5, 12),
        centroid_lat=28.1234,
        centroid_lon=-90.5678,
        polygon={"type": "Polygon", "coordinates": []},
        bounding_box=[-90.6, 28.1, -90.5, 28.2],
        area_km2=2.45,
        area_m2=2_450_000,
        confidence=0.89,
        peak_confidence=0.95,
        model_version="v1.0.0",
        threshold=0.5,
    )

    html_p, json_p = generator.save_report(scene=scene, events=[event])
    assert html_p.exists()
    assert json_p.exists()

    with open(html_p, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "MANDATORY AUDIT NOTICE" in html_text
    assert "SCENE_RPT_TEST" in html_text
    assert "OSE-RPT-001" in html_text
