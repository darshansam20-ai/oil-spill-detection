"""
Codebase Audit Test: Strict AIS Absence Verification (Rule 2).
Scans all source code in `src/` to assert zero occurrences of AIS, MMSI, vessel attribution,
or vessel candidate ranking logic/symbols.
"""
from pathlib import Path
import pytest

FORBIDDEN_TERMS = [
    "ais_",
    "_ais",
    "aisobservation",
    "vesselcandidate",
    "candidate_vessel",
    "mmsi",
    "vessel_association",
    "closest_point_of_approach",
    "track_reconstruction",
    "vessel_trajectory",
    "candidate_vessels",
]

SRC_DIR = Path(__file__).resolve().parent.parent / "src"


def test_codebase_strict_ais_absence():
    """Ensure zero occurrences of forbidden AIS symbols across src/ directory."""
    violations = []
    py_files = list(SRC_DIR.glob("**/*.py")) + list(SRC_DIR.glob("**/*.js")) + list(SRC_DIR.glob("**/*.html"))

    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                clean_line = line.strip().lower()
                # Skip comments explaining PRD AIS removal
                if "#" in clean_line and "no ais" in clean_line or "without ais" in clean_line or "ais removed" in clean_line:
                    continue
                if clean_line.startswith("//") or clean_line.startswith("/*") or clean_line.startswith("*"):
                    continue

                for term in FORBIDDEN_TERMS:
                    if term in clean_line:
                        violations.append(f"{file_path.name}:{line_idx} contains forbidden AIS term '{term}': {line.strip()}")

    assert len(violations) == 0, f"Found {len(violations)} AIS violations:\n" + "\n".join(violations)
