"""
Investigation Report Generator (PRD Section 4.7, FR-35, FR-36 & Section 14).
Produces comprehensive, auditable HTML, JSON, and PDF investigation reports
documenting SAR scene metadata, preprocessing configuration, AI model version,
detected spill metrics, and visual evidence.
"""
import json
import base64
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
import io

from src.config.constants import AUDIT_DISCLAIMER, CURRENT_MODEL_VERSION, MODEL_ARCHITECTURE
from src.config.settings import settings
from src.storage.models import SatelliteScene, OilSpillEvent, Alert
from src.storage.repository import DatabaseRepository, repo
from src.utils.logger import get_logger

logger = get_logger("reporting.report_generator")


def array_to_base64_png(arr: np.ndarray, is_mask: bool = False, is_prob: bool = False) -> str:
    """Convert numpy array to Base64 encoded PNG for self-contained HTML reports."""
    if is_mask:
        uint8_arr = (arr * 255).astype(np.uint8)
        img = Image.fromarray(uint8_arr, mode="L")
    elif is_prob:
        # Colormap approximation for probability heatmap
        uint8_arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        img = Image.fromarray(uint8_arr, mode="L")
    else:
        # Grayscale backscatter
        norm = np.clip(arr, 0.0, 1.0) if arr.max() <= 1.0 else (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
        uint8_arr = (norm * 255).astype(np.uint8)
        img = Image.fromarray(uint8_arr, mode="L")

    # Resize preview for report performance if extremely large
    if max(img.size) > 1024:
        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


class InvestigationReportGenerator:
    """Generates detailed Oil-Spill Investigation Reports."""

    def __init__(self, repository: Optional[DatabaseRepository] = None):
        self.repo = repository or repo
        self.reports_dir = settings.paths.reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report_data(
        self,
        scene: SatelliteScene,
        events: List[OilSpillEvent],
        alerts: Optional[List[Alert]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        raw_img: Optional[np.ndarray] = None,
        prob_map: Optional[np.ndarray] = None,
        binary_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Compile all scene metadata, model outputs, and evidence into report schema.
        """
        alerts = alerts or []
        pre_cfg = preprocessing_config or {
            "sigma0_min_db": settings.preprocessing.sigma0_min_db,
            "sigma0_max_db": settings.preprocessing.sigma0_max_db,
            "speckle_filter": settings.preprocessing.speckle_filter,
            "speckle_window_size": settings.preprocessing.speckle_window_size,
            "normalization": f"[{settings.preprocessing.normalize_min}, {settings.preprocessing.normalize_max}]",
        }

        report_id = f"RPT-{scene.scene_id}-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}"

        report_data = {
            "report_id": report_id,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "audit_disclaimer": AUDIT_DISCLAIMER,
            "scene": {
                "scene_id": scene.scene_id,
                "acquisition_time": scene.acquisition_time.isoformat(),
                "polarization": scene.polarization,
                "orbit_direction": scene.orbit_direction,
                "orbit_number": scene.orbit_number,
                "status": scene.status.value,
            },
            "preprocessing": pre_cfg,
            "model": {
                "architecture": MODEL_ARCHITECTURE,
                "model_version": CURRENT_MODEL_VERSION,
                "patch_size": settings.tiling.patch_size,
                "overlap": settings.tiling.overlap,
                "segmentation_threshold": settings.postprocessing.probability_threshold,
            },
            "summary": {
                "spills_detected": len(events),
                "total_area_km2": round(sum(e.area_km2 for e in events), 4),
                "max_confidence": round(max([e.confidence for e in events], default=0.0), 4),
                "alerts_triggered": len(alerts),
            },
            "events": [
                {
                    "event_id": e.event_id,
                    "centroid": {"lat": e.centroid_lat, "lon": e.centroid_lon},
                    "bounding_box": e.bounding_box,
                    "area_km2": e.area_km2,
                    "area_m2": e.area_m2,
                    "confidence": e.confidence,
                    "peak_confidence": e.peak_confidence,
                    "status": e.status.value,
                }
                for e in events
            ],
            "alerts": [
                {
                    "alert_id": a.alert_id,
                    "event_id": a.event_id,
                    "severity": a.severity.value,
                    "status": a.status.value,
                    "confidence": a.confidence,
                    "area_km2": a.area_km2,
                }
                for a in alerts
            ],
        }

        # Attach image evidence previews
        if raw_img is not None:
            report_data["evidence_sar_b64"] = array_to_base64_png(raw_img)
        if prob_map is not None:
            report_data["evidence_prob_b64"] = array_to_base64_png(prob_map, is_prob=True)
        if binary_mask is not None:
            report_data["evidence_mask_b64"] = array_to_base64_png(binary_mask, is_mask=True)

        return report_data

    def render_html_report(self, data: Dict[str, Any]) -> str:
        """Render a self-contained HTML investigation report."""
        events_html = ""
        for ev in data["events"]:
            events_html += f"""
            <tr>
                <td><code>{ev['event_id']}</code></td>
                <td>{ev['centroid']['lat']:.4f}, {ev['centroid']['lon']:.4f}</td>
                <td><strong>{ev['area_km2']:.3f} km²</strong> ({ev['area_m2']:,.0f} m²)</td>
                <td><span class="badge conf">{ev['confidence'] * 100:.1f}%</span></td>
                <td><span class="badge status-{ev['status'].lower()}">{ev['status']}</span></td>
            </tr>
            """

        alerts_html = ""
        for al in data["alerts"]:
            alerts_html += f"""
            <div class="alert-card sev-{al['severity'].lower()}">
                <strong>{al['severity']} Alert</strong> — <code>{al['alert_id']}</code>
                <p>Event: <code>{al['event_id']}</code> | Area: {al['area_km2']:.3f} km² | Confidence: {al['confidence']*100:.1f}% | Status: {al['status']}</p>
            </div>
            """

        sar_img_tag = f"<img src='data:image/png;base64,{data['evidence_sar_b64']}' alt='SAR Backscatter'/>" if "evidence_sar_b64" in data else "<p>Not available</p>"
        prob_img_tag = f"<img src='data:image/png;base64,{data['evidence_prob_b64']}' alt='Probability Map'/>" if "evidence_prob_b64" in data else "<p>Not available</p>"
        mask_img_tag = f"<img src='data:image/png;base64,{data['evidence_mask_b64']}' alt='Segmentation Mask'/>" if "evidence_mask_b64" in data else "<p>Not available</p>"

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Oil Spill Investigation Report - {data['scene']['scene_id']}</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; margin: 0; padding: 24px; background: #0f172a; color: #f8fafc; line-height: 1.5; }}
        .container {{ max-width: 1100px; margin: 0 auto; background: #1e293b; border-radius: 12px; padding: 32px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); }}
        .header {{ border-bottom: 2px solid #334155; padding-bottom: 16px; margin-bottom: 24px; display: flex; justify-content: space-between; align-items: center; }}
        h1 {{ margin: 0; font-size: 24px; color: #38bdf8; }}
        .meta {{ color: #94a3b8; font-size: 13px; }}
        .disclaimer-banner {{ background: #451a03; border-left: 4px solid #f97316; color: #ffedd5; padding: 14px 18px; border-radius: 6px; margin-bottom: 24px; font-size: 13px; font-weight: 500; }}
        .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        .grid-3 {{ display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
        .card {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; }}
        .card h3 {{ margin-top: 0; font-size: 15px; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.5px; border-bottom: 1px solid #1e293b; padding-bottom: 8px; }}
        .metric {{ font-size: 28px; font-weight: 700; color: #38bdf8; margin: 8px 0; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 12px; font-size: 14px; }}
        th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #334155; }}
        th {{ background: #0f172a; color: #94a3b8; font-weight: 600; }}
        .badge {{ padding: 3px 8px; border-radius: 4px; font-size: 12px; font-weight: 600; display: inline-block; }}
        .badge.conf {{ background: #0369a1; color: #e0f2fe; }}
        .status-new {{ background: #1e3a8a; color: #bfdbfe; }}
        .status-confirmed {{ background: #065f46; color: #a7f3d0; }}
        .status-dismissed {{ background: #374151; color: #9ca3af; }}
        .alert-card {{ padding: 12px 16px; border-radius: 6px; margin-bottom: 10px; font-size: 14px; border-left: 4px solid; }}
        .sev-critical {{ background: #450a0a; border-color: #ef4444; color: #fecaca; }}
        .sev-high {{ background: #431407; border-color: #f97316; color: #ffedd5; }}
        .sev-medium {{ background: #422006; border-color: #eab308; color: #fef08a; }}
        .sev-low {{ background: #1e293b; border-color: #38bdf8; color: #e0f2fe; }}
        .evidence-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-top: 16px; }}
        .evidence-item {{ background: #0f172a; border: 1px solid #334155; border-radius: 6px; padding: 10px; text-align: center; }}
        .evidence-item img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .evidence-item p {{ margin: 8px 0 0 0; font-size: 12px; color: #94a3b8; font-weight: 500; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div>
                <h1>Oil-Spill Investigation Report</h1>
                <div class="meta">Scene ID: <code>{data['scene']['scene_id']}</code> | Generated: {data['generated_at']}</div>
            </div>
            <div>
                <span class="badge" style="background:#0284c7; color:#fff; font-size:14px; padding:6px 12px;">Report ID: {data['report_id']}</span>
            </div>
        </div>

        <div class="disclaimer-banner">
            ⚠️ <strong>MANDATORY AUDIT NOTICE:</strong> {data['audit_disclaimer']}
        </div>

        <div class="grid-3">
            <div class="card">
                <h3>Spill Detections</h3>
                <div class="metric">{data['summary']['spills_detected']}</div>
                <div class="meta">Connected slick regions</div>
            </div>
            <div class="card">
                <h3>Total Surface Area</h3>
                <div class="metric">{data['summary']['total_area_km2']:.3f} <span style="font-size:16px;">km²</span></div>
                <div class="meta">Geodesic WGS84 calculation</div>
            </div>
            <div class="card">
                <h3>Peak Confidence</h3>
                <div class="metric">{data['summary']['max_confidence']*100:.1f}%</div>
                <div class="meta">AI model probability</div>
            </div>
        </div>

        <div class="grid-2">
            <div class="card">
                <h3>Satellite & Preprocessing Provenance</h3>
                <table>
                    <tr><td>Satellite Sensor</td><td>Sentinel-1A SAR (C-Band)</td></tr>
                    <tr><td>Polarization</td><td>{data['scene']['polarization']}</td></tr>
                    <tr><td>Orbit Direction</td><td>{data['scene']['orbit_direction']}</td></tr>
                    <tr><td>Acquisition Time</td><td>{data['scene']['acquisition_time']}</td></tr>
                    <tr><td>Radiometric Calibration</td><td>σ⁰ dB [{data['preprocessing']['sigma0_min_db']} to {data['preprocessing']['sigma0_max_db']} dB]</td></tr>
                    <tr><td>Speckle Noise Filter</td><td>{data['preprocessing']['speckle_filter']} (Window: {data['preprocessing']['speckle_window_size']}x{data['preprocessing']['speckle_window_size']})</td></tr>
                </table>
            </div>

            <div class="card">
                <h3>AI Model & Inference Configuration</h3>
                <table>
                    <tr><td>Model Architecture</td><td>{data['model']['architecture']}</td></tr>
                    <tr><td>Model Version</td><td>{data['model']['model_version']}</td></tr>
                    <tr><td>Patch Size & Overlap</td><td>{data['model']['patch_size']}x{data['model']['patch_size']} (Overlap: {data['model']['overlap']} px)</td></tr>
                    <tr><td>Segmentation Threshold</td><td>{data['model']['segmentation_threshold']}</td></tr>
                    <tr><td>Active Alerts</td><td>{data['summary']['alerts_triggered']} triggered</td></tr>
                    <tr><td>Deployment Mode</td><td>Inference Only (No Backprop)</td></tr>
                </table>
            </div>
        </div>

        <div class="card" style="margin-bottom: 24px;">
            <h3>Visual Evidence & Model Heatmaps</h3>
            <div class="evidence-grid">
                <div class="evidence-item">
                    {sar_img_tag}
                    <p>1. Preprocessed SAR Backscatter (VV)</p>
                </div>
                <div class="evidence-item">
                    {prob_img_tag}
                    <p>2. AI Model Probability Map</p>
                </div>
                <div class="evidence-item">
                    {mask_img_tag}
                    <p>3. Binary Mask & Spill Contours</p>
                </div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 24px;">
            <h3>Detected Spill Regions</h3>
            <table>
                <thead>
                    <tr>
                        <th>Event ID</th>
                        <th>Centroid (Lat, Lon)</th>
                        <th>Estimated Area</th>
                        <th>Confidence</th>
                        <th>Review Status</th>
                    </tr>
                </thead>
                <tbody>
                    {events_html if events_html else '<tr><td colspan="5" style="text-align:center;">No oil spills detected above threshold.</td></tr>'}
                </tbody>
            </table>
        </div>

        {f'<div class="card"><h3>Investigation Alerts</h3>{alerts_html}</div>' if alerts_html else ''}
    </div>
</body>
</html>
        """
        return html

    def save_report(
        self,
        scene: SatelliteScene,
        events: List[OilSpillEvent],
        alerts: Optional[List[Alert]] = None,
        preprocessing_config: Optional[Dict[str, Any]] = None,
        raw_img: Optional[np.ndarray] = None,
        prob_map: Optional[np.ndarray] = None,
        binary_mask: Optional[np.ndarray] = None,
    ) -> Tuple[Path, Path]:
        """
        Generate and persist both JSON and self-contained HTML report files.
        
        Returns:
            Tuple of (html_report_path, json_report_path).
        """
        data = self.generate_report_data(
            scene=scene,
            events=events,
            alerts=alerts,
            preprocessing_config=preprocessing_config,
            raw_img=raw_img,
            prob_map=prob_map,
            binary_mask=binary_mask,
        )

        report_id = data["report_id"]
        json_path = self.reports_dir / f"{report_id}.json"
        html_path = self.reports_dir / f"{report_id}.html"

        # Save JSON (without huge base64 strings for compact storage)
        json_data = {k: v for k, v in data.items() if not k.startswith("evidence_")}
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2)

        # Render & Save HTML
        html_content = self.render_html_report(data)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"Generated investigation reports: HTML -> {html_path}, JSON -> {json_path}")
        return html_path, json_path
