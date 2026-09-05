"""
Pipeline Adapter Layer.
Converts the output of the Sentinel-1 SAR Oil Spill Detection model into the input
required by the AIS Maritime Vessel Correlation model.
"""
from src.adapter.pipeline_adapter import PipelineAdapter, AISInputPayload

__all__ = ["PipelineAdapter", "AISInputPayload"]
