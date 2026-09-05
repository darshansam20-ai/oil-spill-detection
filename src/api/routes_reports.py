"""
API Routes for Investigation Reports.
Resolves both explicit report IDs and scene-based report references with automatic
timestamp matching and latest-version fallback.
"""
from pathlib import Path
from typing import List, Optional, Dict, Any
from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse, FileResponse

from src.config.settings import settings
from src.utils.logger import get_logger

logger = get_logger("api.routes_reports")
router = APIRouter(prefix="/api/reports", tags=["Reports"])


def resolve_report_path(report_id: str, extension: str = ".html") -> Optional[Path]:
    """
    Intelligently find the corresponding report file on disk.
    Supports:
      1. Exact full filename or ID (e.g., 'RPT-SCENE_01-20260830111941')
      2. Base scene ID with prefix (e.g., 'RPT-MY_MANUAL_TEST_01')
      3. Bare scene ID (e.g., 'MY_MANUAL_TEST_01')
      4. Wildcard prefix matching newest timestamped report
    """
    reports_dir = settings.paths.reports_dir
    if not reports_dir.exists():
        return None

    clean_id = report_id[:-len(extension)] if report_id.endswith(extension) else report_id

    # 1. Exact match with extension
    exact_path = reports_dir / f"{clean_id}{extension}"
    if exact_path.exists():
        return exact_path

    # 2. Check if clean_id missing 'RPT-' prefix
    if not clean_id.startswith("RPT-"):
        rpt_exact = reports_dir / f"RPT-{clean_id}{extension}"
        if rpt_exact.exists():
            return rpt_exact

    # 3. Check for wildcard matches (e.g. timestamped versions like RPT-<scene_id>-<timestamp>.html)
    patterns = [
        f"{clean_id}*{extension}",
        f"RPT-{clean_id}*{extension}",
        f"*{clean_id}*{extension}",
    ]
    for pattern in patterns:
        matches = [p for p in reports_dir.glob(pattern) if p.is_file()]
        if matches:
            # Sort by modification time descending to get latest
            matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            return matches[0]

    return None


@router.get("", response_model=List[Dict[str, Any]])
def list_reports():
    """List all generated investigation reports available on the system."""
    reports_dir = settings.paths.reports_dir
    if not reports_dir.exists():
        return []

    html_files = sorted(reports_dir.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True)
    report_list = []
    for f in html_files:
        report_list.append({
            "report_id": f.stem,
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "modified_time": f.stat().st_mtime,
            "html_url": f"/api/reports/{f.stem}/html",
            "json_url": f"/api/reports/{f.stem}/json",
        })
    return report_list


@router.get("/{report_id}/html", response_class=HTMLResponse)
def view_html_report(report_id: str):
    """View self-contained HTML investigation report."""
    html_path = resolve_report_path(report_id, extension=".html")
    if not html_path or not html_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report '{report_id}' not found. Please ensure the scene analysis has completed successfully."
        )

    with open(html_path, "r", encoding="utf-8") as f:
        return f.read()


@router.get("/{report_id}/json")
def get_json_report(report_id: str):
    """Download report metadata JSON."""
    json_path = resolve_report_path(report_id, extension=".json")
    if not json_path or not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report JSON '{report_id}' not found."
        )
    return FileResponse(json_path, media_type="application/json")


@router.get("/scene/{scene_id}/html", response_class=HTMLResponse)
def view_scene_report(scene_id: str):
    """Convenience endpoint to view latest report for a specific scene."""
    return view_html_report(scene_id)
