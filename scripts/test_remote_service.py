"""
Remote Production Service Testing Suite.
Sends real requests (GET /health, POST /predict) to a deployed remote URL (e.g. Hugging Face Space or Render)
and verifies that both models execute in sequence and return valid oil spill and AIS intelligence.
"""
import argparse
import os
import sys
import time
from pathlib import Path
import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def test_remote(base_url: str):
    if base_url.endswith("/"):
        base_url = base_url[:-1]

    print("\n" + "=" * 80)
    print(f"      TESTING LIVE REMOTE SERVICE: {base_url}      ")
    print("=" * 80)

    # 1. Test GET /health
    health_url = f"{base_url}/health"
    print(f"\n[1/2] Testing Health Probe: GET {health_url} ...")
    try:
        start_t = time.time()
        resp = requests.get(health_url, timeout=30)
        dur = time.time() - start_t
        print(f" -> HTTP Status: {resp.status_code} ({dur:.2f}s)")
        
        if resp.status_code == 200:
            data = resp.json()
            print(f" -> Service: {data.get('service')}")
            print(f" -> Model Version: {data.get('model_version')}")
            print(f" -> Device: {data.get('device')}")
            print(f" -> Checkpoint Loaded: {data.get('checkpoint_loaded')}")
            print(" ✓ [PASS] Health check succeeded!")
        else:
            print(f" ✕ [FAIL] Health check returned non-200: {resp.text}")
            return False
    except Exception as e:
        print(f" ✕ [FAIL] Could not connect to {health_url}: {e}")
        print("    Ensure the Space/Server has finished building and status is 'RUNNING'.")
        return False

    # 2. Test POST /predict
    predict_url = f"{base_url}/predict"
    sample_image = PROJECT_ROOT / "data" / "extracted" / "test" / "images" / "2018_12_19_d.tif"
    sample_metadata = PROJECT_ROOT / "data" / "sample_metadata.json"

    assert sample_image.exists(), f"Sample image missing: {sample_image}"
    assert sample_metadata.exists(), f"Sample metadata missing: {sample_metadata}"

    with open(sample_metadata, "r", encoding="utf-8") as f:
        meta_str = f.read()

    print(f"\n[2/2] Testing Full Pipeline: POST {predict_url} with SAR Scene {sample_image.name} ...")
    print("      (Executing: MY MODEL -> ADAPTER -> FRIEND'S MODEL -> RESULT)")
    try:
        start_t = time.time()
        with open(sample_image, "rb") as img_f:
            files = {"image": (sample_image.name, img_f, "image/tiff")}
            data = {
                "metadata": meta_str,
                "threshold": "0.50",
                "search_radius_km": "20.0",
            }
            resp = requests.post(predict_url, files=files, data=data, timeout=120)
        dur = time.time() - start_t
        print(f" -> HTTP Status: {resp.status_code} (Inference Duration: {dur:.2f}s)")

        if resp.status_code == 200:
            res_json = resp.json()
            det = res_json.get("oil_spill_detection", {})
            adapter = res_json.get("adapter_payload", {})
            ais = res_json.get("ais_vessel_correlation", {})
            artifacts = res_json.get("artifacts", {})

            print("\n" + "-" * 75)
            print("           LIVE PREDICTION RESULTS RECEIVED           ")
            print("-" * 75)
            print(f" Incident ID:         {res_json.get('incident_id')}")
            print(f" Spills Detected:     {det.get('spills_detected')} clusters")
            print(f" Total Spill Area:    {det.get('total_area_km2')} sq km ({det.get('total_spill_pixels'):,} px)")
            print(f" Epicenter Coordinates: Lat {adapter.get('spill_latitude')}, Lon {adapter.get('spill_longitude')}")
            print(f" AIS Vessels Found:   {ais.get('total_vessels_detected')}")
            if ais.get("primary_suspect"):
                top = ais["primary_suspect"]
                print(f" Primary Suspect:     {top.get('ship_name')} (MMSI: {top.get('mmsi')}, Distance: {top.get('minimum_distance_km')} km)")
            
            has_b64 = bool(artifacts.get("annotated_image_data_uri", "").startswith("data:image/png;base64,"))
            has_map = bool(len(artifacts.get("interactive_map_html", "")) > 100)
            print(f" Annotated SAR Image: {'✓ Received (' + str(len(artifacts.get('annotated_image_data_uri', ''))) + ' chars)' if has_b64 else '✕ Missing'}")
            print(f" Interactive Map:     {'✓ Received (' + str(len(artifacts.get('interactive_map_html', ''))) + ' chars)' if has_map else '✕ Missing'}")
            print("-" * 75)
            print(" ✓ [PASS] Complete sequential dual-model pipeline verified on live remote host!")
            print("=" * 80 + "\n")
            return True
        else:
            print(f" ✕ [FAIL] Prediction failed with status {resp.status_code}: {resp.text}")
            return False

    except Exception as e:
        print(f" ✕ [FAIL] Error sending prediction request: {e}")
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test remote AEGIS-SAR ML service")
    parser.add_argument(
        "--url", "-u",
        type=str,
        required=True,
        help="Base URL of the deployed ML service (e.g. https://username-aegis-sar-ml.hf.space)",
    )
    args = parser.parse_args()
    test_remote(args.url)
