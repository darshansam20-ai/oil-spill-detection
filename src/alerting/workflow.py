"""
Human-in-the-Loop Review Workflow & Audit Trail (PRD Section 5 & FR-36).
Tracks operator review decisions, timestamps, and notes on candidate oil spill alerts.
"""
from datetime import datetime
from typing import List, Optional
from src.config.constants import ReviewStatus
from src.storage.models import Alert
from src.storage.repository import DatabaseRepository, repo
from src.utils.logger import get_logger

logger = get_logger("alerting.workflow")


class HumanReviewWorkflow:
    """Manages human-in-the-loop alert review transitions and audit trails."""

    def __init__(self, repository: Optional[DatabaseRepository] = None):
        self.repo = repository or repo

    def submit_review(
        self,
        alert_id: str,
        decision: ReviewStatus,
        reviewer_name: str = "operator",
        reviewer_notes: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Record an operator's investigation decision on an alert.
        """
        logger.info(f"Operator '{reviewer_name}' marked alert {alert_id} as {decision.value}. Notes: {reviewer_notes}")
        updated_alert = self.repo.update_alert_review(
            alert_id=alert_id,
            status=decision,
            notes=reviewer_notes,
            reviewer=reviewer_name,
        )
        return updated_alert

    def get_pending_alerts(self, limit: int = 50) -> List[Alert]:
        """List alerts awaiting human operator review."""
        return self.repo.list_alerts(status=ReviewStatus.NEW.value, limit=limit)
