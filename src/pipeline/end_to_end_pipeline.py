"""
Unified End-to-End Pipeline Orchestrator.

Combines:
1. Sentinel-1 SAR Oil Spill Semantic Segmentation & Bounding Box Detection (User's Model)
2. Pipeline Data & Metadata Adapter Layer
3. AIS Maritime Vessel Proximity & Trajectory Correlation Engine (Friend's Model)

Provides a single programmatic interface:
`run_pipeline(image, metadata=..., ...)`
"""
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import json
import sys

from src.adapter.pipeline_adapter import PipelineAdapter, AISInputPayload
from src.config.settings import settings
from src.inference.ais_correlator import AISCorrelator, AISCorrelationResult
from src.inference.oil_spill_detector import Sentinel1OilSpillDetector, DetectionResult
from src.utils.logger import get_logger

logger = get_logger("pipeline.end_to_end")

# Ensure UTF-8 output on Windows terminals
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


@dataclass
class EndToEndIncidentReport:
    """Comprehensive Incident Assessment combining Detection and Vessel Correlation."""
    incident_id: str
    timestamp: str
    image_path: str
    detection_result: DetectionResult
    adapter_payload: AISInputPayload
    ais_result: AISCorrelationResult
    output_dir: str
    annotated_image_path: Optional[str]
    interactive_map_path: Optional[str]
    vessel_ranking_json_path: Optional[str]
    vessel_ranking_csv_path: Optional[str]
    incident_report_json_path: Optional[str]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "incident_id": self.incident_id,
            "pipeline_timestamp": self.timestamp,
            "input_sar_image": self.image_path,
            "oil_spill_detection": self.detection_result.to_dict(),
            "adapter_payload": self.adapter_payload.to_dict(),
            "ais_vessel_correlation": self.ais_result.to_dict(),
            "artifacts": {
                "annotated_detection_image": self.annotated_image_path,
                "interactive_ais_map": self.interactive_map_path,
                "vessel_ranking_json": self.vessel_ranking_json_path,
                "vessel_ranking_csv": self.vessel_ranking_csv_path,
                "full_incident_report_json": self.incident_report_json_path,
            },
        }

    def print_comprehensive_summary(self):
        """Print unified end-to-end incident summary report."""
        print("\n" + "=" * 80)
        print("          SENTINEL-1 SAR & AIS INTEGRATED INCIDENT REPORT          ")
        print("=" * 80)
        print(f" Incident ID:       {self.incident_id}")
        print(f" Input SAR Image:   {self.image_path}")
        print(f" Metadata Status:   {'Provided' if self.detection_result.has_metadata else 'Image-Only (Fallback Coordinates)'}")
        print(f" Spills Detected:   {self.detection_result.spills_detected}")
        print(f" Total Spill Area:  {self.detection_result.total_area_km2:.4f} sq km" if self.detection_result.total_area_km2 else f" Total Spill Pixels: {self.detection_result.total_spill_pixels:,} px")
        print("-" * 80)
        print(f" Target Epicenter:  Lat: {self.adapter_payload.spill_latitude:.5f}, Lon: {self.adapter_payload.spill_longitude:.5f}")
        print(f" Incident Time:     {self.adapter_payload.detection_time}")
        print(f" AIS Search Radius: {self.adapter_payload.search_radius_km:.1f} km")
        print(f" AIS Data Source:   {self.ais_result.data_source}")
        print(f" Vessels Detected:  {self.ais_result.total_vessels_detected}")
        print("-" * 80)

        if self.ais_result.ranking:
            top_suspect = self.ais_result.ranking[0]
            print(f" [!] PRIMARY SUSPECT VESSEL: '{top_suspect.ship_name}'")
            print(f"     - MMSI:             {top_suspect.mmsi}")
            print(f"     - Vessel Type:      {top_suspect.vessel_type}")
            print(f"     - Minimum Distance: {top_suspect.minimum_distance_km:.2f} km from spill epicenter")
            print(f"     - Track Points:     {top_suspect.historical_position_count} positions recorded")
            print("\n Top 5 Nearby Vessels:")
            print(f" {'Rank':<5} | {'Ship Name':<22} | {'MMSI':<12} | {'Min Distance':<15} | {'Type':<15}")
            print(" " + "-" * 76)
            for v in self.ais_result.ranking[:5]:
                print(f" {v.rank:<5} | {v.ship_name[:21]:<22} | {v.mmsi:<12} | {v.minimum_distance_km:.2f} km{' ':8} | {v.vessel_type[:14]:<15}")
        else:
            print(" [*] No AIS vessels correlated within the specified radius.")

        print("-" * 80)
        print(" Generated Artifacts:")
        print(f" [1] Annotated Image: {self.annotated_image_path or 'N/A'}")
        print(f" [2] Interactive Map: {self.interactive_map_path or 'N/A'}")
        print(f" [3] Ranking JSON:    {self.vessel_ranking_json_path or 'N/A'}")
        print(f" [4] Ranking CSV:     {self.vessel_ranking_csv_path or 'N/A'}")
        print(f" [5] Complete Report: {self.incident_report_json_path or 'N/A'}")
        print("=" * 80 + "\n")


class EndToEndPipeline:
    """
    Production-ready integrated pipeline combining:
    1. Sentinel-1 SAR Oil Spill Detector
    2. Pipeline Adapter
    3. AIS Vessel Correlator
    """

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        detection_threshold: float = 0.50,
        min_spill_pixels: int = 50,
        ais_token: Optional[str] = None,
        ais_search_radius_km: float = 20.0,
        device: Optional[str] = None,
    ):
        logger.info("Initializing Integrated Sentinel-1 & AIS Pipeline...")
        self.detector = Sentinel1OilSpillDetector(
            checkpoint_path=checkpoint_path,
            threshold=detection_threshold,
            min_spill_pixels=min_spill_pixels,
            device=device,
        )
        self.adapter = PipelineAdapter(
            default_search_radius_km=ais_search_radius_km,
        )
        self.ais_correlator = AISCorrelator(
            token=ais_token,
            search_radius_km=ais_search_radius_km,
        )
        logger.info("Integrated Pipeline initialized successfully.")

    def run(
        self,
        image_path: Union[str, Path],
        metadata: Optional[Union[Dict[str, Any], str, Path]] = None,
        output_dir: Optional[Union[str, Path]] = None,
        spill_id_to_correlate: Optional[int] = None,
        search_radius_km: Optional[float] = None,
        override_lat: Optional[float] = None,
        override_lon: Optional[float] = None,
        override_time: Optional[str] = None,
    ) -> EndToEndIncidentReport:
        """
        Execute full end-to-end pipeline:
        Image -> Oil Spill Detection -> Adapter -> AIS Vessel Correlation -> Report.
        """
        img_path = Path(image_path)
        stem = img_path.stem
        out_base_dir = Path(output_dir) if output_dir else Path("output")
        out_base_dir.mkdir(parents=True, exist_ok=True)

        detections_dir = out_base_dir / "detections"
        ais_dir = out_base_dir / "ais"
        detections_dir.mkdir(parents=True, exist_ok=True)
        ais_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"=== STEP 1/3: Running Oil Spill Detection on {img_path.name} ===")
        annotated_img_path = detections_dir / f"{stem}_spill_detected.png"
        detection_json_path = detections_dir / f"{stem}_detection.json"

        detection_result = self.detector.detect(
            image=img_path,
            metadata=metadata,
            output_image_path=annotated_img_path,
            save_json=detection_json_path,
        )

        logger.info(f"=== STEP 2/3: Adapting Model Output for AIS Module ===")
        adapter_payload = self.adapter.convert_detection_to_ais_input(
            detection_result=detection_result,
            spill_id=spill_id_to_correlate,
            custom_radius_km=search_radius_km,
            override_lat=override_lat,
            override_lon=override_lon,
            override_time=override_time,
        )

        logger.info(f"=== STEP 3/3: Running AIS Maritime Vessel Correlation ===")
        ais_result = self.ais_correlator.correlate(
            spill_lat=adapter_payload.spill_latitude,
            spill_lon=adapter_payload.spill_longitude,
            detection_time=adapter_payload.detection_time,
            search_radius_km=adapter_payload.search_radius_km,
            output_dir=ais_dir,
            base_filename=stem,
        )

        # Generate Complete Incident Report
        incident_id = f"INCIDENT-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{stem[:10]}"
        report_json_path = out_base_dir / f"{stem}_end_to_end_report.json"

        incident_report = EndToEndIncidentReport(
            incident_id=incident_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            image_path=str(img_path.resolve()),
            detection_result=detection_result,
            adapter_payload=adapter_payload,
            ais_result=ais_result,
            output_dir=str(out_base_dir.resolve()),
            annotated_image_path=str(annotated_img_path.resolve()) if annotated_img_path.exists() else None,
            interactive_map_path=str(Path(ais_result.map_file_path).resolve()) if ais_result.map_file_path else None,
            vessel_ranking_json_path=str(Path(ais_result.json_file_path).resolve()) if ais_result.json_file_path else None,
            vessel_ranking_csv_path=str(Path(ais_result.csv_file_path).resolve()) if ais_result.csv_file_path else None,
            incident_report_json_path=str(report_json_path.resolve()),
        )

        with open(report_json_path, "w", encoding="utf-8") as f:
            json.dump(incident_report.to_dict(), f, indent=2)
        logger.info(f"Saved complete end-to-end incident report to: {report_json_path}")

        return incident_report
