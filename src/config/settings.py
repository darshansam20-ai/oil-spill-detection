"""
Application configuration and environment settings using Pydantic.
"""
import os
from pathlib import Path
from typing import List, Optional
from pydantic import BaseModel, Field


class PreprocessingSettings(BaseModel):
    """SAR preprocessing parameters."""
    sigma0_min_db: float = -30.0
    sigma0_max_db: float = 0.0
    speckle_filter: str = "refined_lee"  # 'refined_lee', 'lee', 'none'
    speckle_window_size: int = 7
    speckle_num_looks: int = 1
    normalize_min: float = 0.0
    normalize_max: float = 1.0
    apply_db_conversion: bool = True


class TilingSettings(BaseModel):
    """Large-scene tiling parameters."""
    patch_size: int = 256
    overlap: int = 64
    stride: int = 192  # patch_size - overlap
    blend_mode: str = "gaussian"  # 'gaussian', 'hann', 'linear', 'mean'


class ModelSettings(BaseModel):
    """ConvNeXt-Tiny + U-Net AI model hyperparameters."""
    architecture: str = "convnext_tiny_unet"
    in_channels: int = 1
    num_classes: int = 1
    encoder_name: str = "convnext_tiny"
    pretrained: bool = True
    probability_threshold: float = 0.5
    device: str = "cuda"  # 'cuda' or 'cpu'


class PostprocessingSettings(BaseModel):
    """Segmentation mask post-processing parameters."""
    probability_threshold: float = 0.5
    min_spill_pixels: int = 50
    opening_radius: int = 2
    closing_radius: int = 3
    extract_polygons: bool = True
    simplify_tolerance: float = 0.0001


class AlertingSettings(BaseModel):
    """Alert threshold criteria."""
    confidence_threshold: float = 0.65
    area_km2_threshold: float = 0.20
    critical_area_threshold: float = 5.0
    high_area_threshold: float = 1.0
    medium_area_threshold: float = 0.20


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class PathSettings(BaseModel):
    """Filesystem paths for datasets, artifacts, storage and reports."""
    base_dir: Path = Field(default_factory=lambda: Path(os.getenv("BASE_DIR", str(PROJECT_ROOT))))
    data_raw: Path = Field(default_factory=lambda: Path(os.getenv("DATA_RAW", str(PROJECT_ROOT / "data" / "raw"))))
    data_extracted: Path = Field(default_factory=lambda: Path(os.getenv("DATA_EXTRACTED", str(PROJECT_ROOT / "data" / "extracted"))))
    data_processed: Path = Field(default_factory=lambda: Path(os.getenv("DATA_PROCESSED", str(PROJECT_ROOT / "data" / "processed"))))
    data_outputs: Path = Field(default_factory=lambda: Path(os.getenv("DATA_OUTPUTS", str(PROJECT_ROOT / "data" / "outputs"))))
    checkpoints_dir: Path = Field(default_factory=lambda: Path(os.getenv("CHECKPOINTS_DIR", str(PROJECT_ROOT / "artifacts" / "checkpoints"))))
    metadata_dir: Path = Field(default_factory=lambda: Path(os.getenv("METADATA_DIR", str(PROJECT_ROOT / "artifacts" / "metadata"))))
    reports_dir: Path = Field(default_factory=lambda: Path(os.getenv("REPORTS_DIR", str(PROJECT_ROOT / "artifacts" / "reports"))))
    db_path: Path = Field(default_factory=lambda: Path(os.getenv("DB_PATH", str(PROJECT_ROOT / "data" / "oil_spill_system.db"))))

    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for path in [
            self.data_raw,
            self.data_extracted,
            self.data_processed,
            self.data_outputs,
            self.checkpoints_dir,
            self.metadata_dir,
            self.reports_dir,
            self.db_path.parent,
        ]:
            try:
                path.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass


class PipelineSettings(BaseModel):
    """Global system configuration."""
    app_name: str = "Automated SAR Oil-Spill Detection & AIS Tracking System"
    app_version: str = "1.0.0"
    debug: bool = False
    model_version: str = "v1.0.0-convnext-unet"

    paths: PathSettings = Field(default_factory=PathSettings)
    preprocessing: PreprocessingSettings = Field(default_factory=PreprocessingSettings)
    tiling: TilingSettings = Field(default_factory=TilingSettings)
    model: ModelSettings = Field(default_factory=ModelSettings)
    postprocessing: PostprocessingSettings = Field(default_factory=PostprocessingSettings)
    alerting: AlertingSettings = Field(default_factory=AlertingSettings)


# Singleton global settings instance
settings = PipelineSettings()
settings.paths.ensure_directories()

