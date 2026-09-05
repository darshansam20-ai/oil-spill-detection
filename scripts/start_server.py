import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import uvicorn
from scripts.run_pipeline import ingest_all_local_scenes

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start SAR Oil Spill Detection Web Server")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host address")
    parser.add_argument("--port", type=int, default=8000, help="Port number")
    args = parser.parse_args()

    # Pre-register scenes in database
    try:
        ingest_all_local_scenes()
    except Exception as e:
        print(f"Notice during scene cataloging: {e}")

    print("\n==================================================================")
    print(f" [*] SAR Oil-Spill Detection Server Starting on http://{args.host}:{args.port}")
    print(f" [*] Interactive Dashboard: http://{args.host}:{args.port}/")
    print(f" [*] OpenAPI Documentation: http://{args.host}:{args.port}/docs")
    print("==================================================================\n")

    uvicorn.run("src.api.app:app", host=args.host, port=args.port, reload=False)
