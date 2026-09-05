"""
Automated One-Click Hugging Face Docker Space Deployment Tool.
Uploads the complete AEGIS-SAR ML Inference Service to Hugging Face Spaces using the huggingface_hub Python SDK.
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    from huggingface_hub import HfApi, login, get_space_runtime
except ImportError:
    print("[Error] huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def deploy(repo_id: str, token: str):
    print("=" * 80)
    print("      AEGIS-SAR HUGGING FACE DOCKER SPACE DEPLOYMENT TOOL      ")
    print("=" * 80)

    api = HfApi(token=token)

    # 1. Validate Token & User Identity
    try:
        user_info = api.whoami()
        username = user_info.get("name")
        print(f"[*] Authenticated with Hugging Face as: @{username}")
    except Exception as e:
        print(f"[Error] Failed to authenticate with Hugging Face: {e}")
        sys.exit(1)

    # Format full repo_id if only space name was provided
    if "/" not in repo_id:
        full_repo_id = f"{username}/{repo_id}"
    else:
        full_repo_id = repo_id

    print(f"[*] Target Hugging Face Space: https://huggingface.co/spaces/{full_repo_id}")

    # 2. Create Space if it doesn't exist
    try:
        print(f"[*] Creating / verifying Space '{full_repo_id}' (SDK: Docker)...")
        api.create_repo(
            repo_id=full_repo_id,
            repo_type="space",
            space_sdk="docker",
            exist_ok=True,
            private=False,
        )
        print(" ✓ [OK] Space repository verified on Hugging Face.")
    except Exception as e:
        print(f"[Warning] Repository check: {e}")

    # 3. Upload Project Files (Excluding local scratch & cache)
    ignore_patterns = [
        ".venv/**",
        ".git/**",
        ".vscode/**",
        ".pytest_cache/**",
        "__pycache__/**",
        "*.pyc",
        "data/extracted/**",
        "data/raw/**",
        "data/processed/**",
        "output/**",
        "AIS MODULE.zip",
        "ais_module_extracted/**",
        "tests/**",
    ]

    print("\n[*] Uploading ML service container files & weights (including 383MB best_model.pt)...")
    print("    This may take 1-3 minutes depending on your internet connection.")
    try:
        api.upload_folder(
            folder_path=str(PROJECT_ROOT),
            repo_id=full_repo_id,
            repo_type="space",
            ignore_patterns=ignore_patterns,
            commit_message="Deploy AEGIS-SAR Production ML Service Container",
        )
        print(" ✓ [OK] All files and model checkpoints uploaded successfully!")
    except Exception as e:
        print(f"[Error] Failed to upload files: {e}")
        sys.exit(1)

    # 4. Construct Live URLs
    # Format: https://<username>-<space_name>.hf.space
    space_name_part = full_repo_id.split("/")[-1].replace("_", "-")
    user_part = full_repo_id.split("/")[0].replace("_", "-")
    live_direct_url = f"https://{user_part}-{space_name_part}.hf.space"
    space_dashboard_url = f"https://huggingface.co/spaces/{full_repo_id}"

    print("\n" + "=" * 80)
    print("                      DEPLOYMENT INITIATED                      ")
    print("=" * 80)
    print(f" Space Dashboard: {space_dashboard_url}")
    print(f" Live API URL:    {live_direct_url}")
    print("=" * 80)
    print("\n[*] Hugging Face is now building the Docker container.")
    print("    You can view real-time build logs at:")
    print(f"    {space_dashboard_url}")
    print("\n[*] Once the Space status says 'RUNNING', verify the deployment with:")
    print(f"    python scripts/test_remote_service.py --url {live_direct_url}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy AEGIS-SAR to Hugging Face Docker Space")
    parser.add_argument(
        "--token", "-t",
        type=str,
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face API Write Token (from https://huggingface.co/settings/tokens)",
    )
    parser.add_argument(
        "--space-name", "-s",
        type=str,
        default=os.getenv("HF_SPACE_NAME", "aegis-sar-ml"),
        help="Space Name or full repo_id (e.g. 'aegis-sar-ml' or 'username/aegis-sar-ml')",
    )

    args = parser.parse_args()

    if not args.token:
        print("\n" + "!" * 80)
        print(" [Action Required] Hugging Face Access Token Needed")
        print("!" * 80)
        print(" 1. Go to: https://huggingface.co/settings/tokens")
        print(" 2. Create a token with 'Write' role.")
        print(" 3. Run: python scripts/deploy_to_huggingface.py --token YOUR_HF_TOKEN --space-name aegis-sar-ml")
        print("!" * 80 + "\n")
        sys.exit(1)

    deploy(repo_id=args.space_name, token=args.token)
