"""
End-to-End Pipeline Package.
Orchestrates Sentinel-1 SAR Oil Spill Detection and AIS Maritime Vessel Correlation.
"""
from src.pipeline.end_to_end_pipeline import EndToEndPipeline, EndToEndIncidentReport

__all__ = ["EndToEndPipeline", "EndToEndIncidentReport"]
