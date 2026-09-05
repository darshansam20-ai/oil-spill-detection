"""
API Routes for Alert Management and Human Review Decisions.
"""
from typing import List, Optional
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from src.config.constants import ReviewStatus, AlertSeverity
from src.storage.models import Alert
from src.storage.repository import repo
from src.alerting.workflow import HumanReviewWorkflow

router = APIRouter(prefix="/api/alerts", tags=["Alerts & Review"])
review_workflow = HumanReviewWorkflow()


class ReviewDecisionRequest(BaseModel):
    decision: ReviewStatus
    reviewer_name: str = "operator"
    reviewer_notes: Optional[str] = None


@router.get("", response_model=List[Alert])
def list_alerts(
    severity: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = Query(50, ge=1, le=200),
):
    """List investigation alerts."""
    return repo.list_alerts(severity=severity, status=status, limit=limit)


@router.post("/{alert_id}/review", response_model=Alert)
def submit_review_decision(alert_id: str, req: ReviewDecisionRequest):
    """Submit human-in-the-loop operator review decision for an alert."""
    updated = review_workflow.submit_review(
        alert_id=alert_id,
        decision=req.decision,
        reviewer_name=req.reviewer_name,
        reviewer_notes=req.reviewer_notes,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Alert '{alert_id}' not found.")
    return updated
