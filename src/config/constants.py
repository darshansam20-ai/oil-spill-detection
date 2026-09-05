"""
Constants and Enums for the Automated SAR Oil-Spill Detection System.
Strictly Oil-Spill Detection Scope (No AIS).
"""
from enum import Enum


class SceneStatus(str, Enum):
    """Lifecycle statuses for a Sentinel-1 satellite scene."""
    DISCOVERED = "DISCOVERED"
    FILTERED = "FILTERED"
    INGESTED = "INGESTED"
    PREPROCESSING = "PREPROCESSING"
    INFERENCE = "INFERENCE"
    POSTPROCESSING = "POSTPROCESSING"
    GEOSPATIALIZED = "GEOSPATIALIZED"
    COMPLETED = "COMPLETED"
    NO_SPILL_DETECTED = "NO_SPILL_DETECTED"
    FAILED_INGESTION = "FAILED_INGESTION"
    FAILED_PREPROCESSING = "FAILED_PREPROCESSING"
    FAILED_INFERENCE = "FAILED_INFERENCE"
    FAILED_POSTPROCESSING = "FAILED_POSTPROCESSING"
    FAILED_GEOSPATIALIZATION = "FAILED_GEOSPATIALIZATION"
    FAILED_PROCESSING = "FAILED_PROCESSING"


class AlertSeverity(str, Enum):
    """Severity levels for detected oil spill events."""
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ReviewStatus(str, Enum):
    """Human-in-the-loop review status for detected spill events."""
    NEW = "NEW"
    UNDER_REVIEW = "UNDER_REVIEW"
    CONFIRMED = "CONFIRMED"
    DISMISSED = "DISMISSED"


class Polarization(str, Enum):
    """SAR polarization modes."""
    VV = "VV"
    VH = "VH"
    HH = "HH"
    HV = "HV"
    VV_VH = "VV+VH"


class OrbitDirection(str, Enum):
    """Satellite orbit pass direction."""
    ASCENDING = "ASCENDING"
    DESCENDING = "DESCENDING"
    ANY = "ANY"


# Preprocessing and Calibration Constants
DEFAULT_SIGMA0_MIN_DB = -30.0
DEFAULT_SIGMA0_MAX_DB = 0.0
DEFAULT_EPSILON = 1e-7

# Tiling Defaults
DEFAULT_PATCH_SIZE = 256
DEFAULT_OVERLAP = 64  # 25% overlap
DEFAULT_STRIDE = 192

# Segmentation Defaults
DEFAULT_PROBABILITY_THRESHOLD = 0.5
DEFAULT_MIN_SPILL_PIXELS = 50  # Minimum connected component size in pixels
DEFAULT_OPENING_RADIUS = 2
DEFAULT_CLOSING_RADIUS = 3

# Alert Thresholds
DEFAULT_ALERT_CONFIDENCE_THRESHOLD = 0.65
DEFAULT_ALERT_AREA_KM2_THRESHOLD = 0.20  # km²
CRITICAL_AREA_KM2_THRESHOLD = 5.0
HIGH_AREA_KM2_THRESHOLD = 1.0
MEDIUM_AREA_KM2_THRESHOLD = 0.2

# Model Metadata
MODEL_ARCHITECTURE = "ConvNeXt-Tiny + U-Net"
CURRENT_MODEL_VERSION = "v1.0.0-convnext-unet"

# Mandatory Audit & Human Review Disclaimer
AUDIT_DISCLAIMER = (
    "MODEL-DETECTED POTENTIAL OIL SPILL. Generated automatically by SAR deep learning segmentation. "
    "For screening and investigative purposes only; not verified ground-truth attribution."
)
