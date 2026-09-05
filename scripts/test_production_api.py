"""
End-to-End Production API Verification Suite.
Tests health checks, full prediction pipeline on sample SAR scene, and security validation.
"""
import json
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.api.app import app


def test_production_flow():
    print("=" * 80)
    print("      AEGIS-SAR PRODUCTION API END-TO-END VERIFICATION SUITE      ")
    print("=" * 80)

    client = TestClient(app)

    # 1. Test Health Check
    print("\n[TEST 1/3] Testing GET /health...")
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Health check failed: {health_resp.text}"
    health_data = health_resp.json()
    print(f" -> Status: {health_data.get('status')}")
    print(f" -> Model Version: {health_data.get('model_version')}")
    print(f" -> Device: {health_data.get('device')}")
    print(f" -> Checkpoint Loaded: {health_data.get('checkpoint_loaded')}")
    assert health_data.get("status") == "online"
    print(" ✓ [PASS] Health check verified successfully.")

    # 2. Test Security: Invalid File Type Rejection
    print("\n[TEST 2/3] Testing Security & Input Validation (Invalid File Upload)...")
    invalid_file_payload = {"image": ("test_script.sh", b"echo 'malicious payload'", "text/plain")}
    invalid_resp = client.post("/predict", files=invalid_file_payload)
    print(f" -> Response Code: {invalid_resp.status_code}")
    print(f" -> Response Detail: {invalid_resp.json().get('detail')}")
    assert invalid_resp.status_code == 400, f"Expected 400 Bad Request, got {invalid_resp.status_code}"
    print(" ✓ [PASS] Invalid file rejection verified successfully.")

    # 3. Test Full End-to-End Prediction on Valid Sentinel-1 SAR Image
    sample_image_path = PROJECT_ROOT / "data" / "extracted" / "test" / "images" / "2018_12_19_d.tif"
    sample_meta_path = PROJECT_ROOT / "data" / "sample_metadata.json"

    assert sample_image_path.exists(), f"Sample image not found at: {sample_image_path}"
    assert sample_meta_path.exists(), f"Sample metadata not found at: {sample_meta_path}"

    with open(sample_meta_path, "r", encoding="utf-8") as f:
        meta_json_str = f.read()

    print(f"\n[TEST 3/3] Testing POST /predict with Sentinel-1 SAR Scene: {sample_image_path.name}...")
    with open(sample_image_path, "rb") as img_file:
        files = {"image": (sample_image_path.name, img_file, "image/tiff")}
        data = {
            "metadata": meta_json_str,
            "threshold": "0.50",
            "search_radius_km": "20.0",
        }
        resp = client.post("/predict", files=files, data=data)

    assert resp.status_code == 200, f"Prediction failed with status {resp.status_code}: {resp.text}"
    res_data = resp.json()

    print(f" -> Incident ID: {res_data.get('incident_id')}")
    print(f" -> Status: {res_data.get('status')}")
    
    det = res_data.get("oil_spill_detection", {})
    print(f" -> Spills Detected: {det.get('spills_detected')}")
    print(f" -> Total Spill Pixels: {det.get('total_spill_pixels'):,}")
    print(f" -> Total Area: {det.get('total_area_km2')} sq km")
    assert det.get("spills_detected", 0) > 0, "Expected at least 1 detected spill"

    adapter = res_data.get("adapter_payload", {})
    print(f" -> Target Epicenter: Lat: {adapter.get('spill_latitude')}, Lon: {adapter.get('spill_longitude')}")
    print(f" -> Detection Time: {adapter.get('detection_time')}")
    assert adapter.get("spill_latitude") is not None
    assert adapter.get("spill_longitude") is not None

    ais = res_data.get("ais_vessel_correlation", {})
    print(f" -> AIS Vessels Found: {ais.get('total_vessels_detected')}")
    if ais.get("primary_suspect"):
        top = ais["primary_suspect"]
        print(f" -> Top Suspect: {top.get('ship_name')} (MMSI: {top.get('mmsi')}, Distance: {top.get('minimum_distance_km')} km)")
    assert ais.get("total_vessels_detected", 0) > 0, "Expected AIS vessels to be correlated"

    artifacts = res_data.get("artifacts", {})
    has_annotated_img = bool(artifacts.get("annotated_image_data_uri", "").startswith("data:image/png;base64,"))
    has_map_html = bool("<html" in artifacts.get("interactive_map_html", "").lower() or "<div" in artifacts.get("interactive_map_html", "").lower())
    print(f" -> Base64 Annotated Image Generated: {has_annotated_img} (Length: {len(artifacts.get('annotated_image_data_uri', ''))} chars)")
    print(f" -> Interactive Map HTML Generated: {has_map_html} (Length: {len(artifacts.get('interactive_map_html', ''))} chars)")
    assert has_annotated_img, "Expected valid base64 image data URI"
    assert has_map_html, "Expected valid HTML interactive map"

    print(" ✓ [PASS] Full Dual-Model Pipeline executed and verified successfully!")
    print("\n" + "=" * 80)
    print("             ALL VERIFICATION SUITE CHECKS PASSED (3/3)             ")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    test_production_flow()
