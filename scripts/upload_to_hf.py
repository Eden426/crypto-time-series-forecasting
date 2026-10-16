"""
Upload trained GRU models (.keras) and scalers (.pkl) to Hugging Face Hub.

Usage:
    python scripts/upload_to_hf.py --repo-id <your-hf-username>/crypto-time-series-gru --token <hf_token>

Or using environment variables:
    set HF_REPO_ID=<your-hf-username>/crypto-time-series-gru
    set HF_TOKEN=hf_xxxxxxxxx
    python scripts/upload_to_hf.py
"""

import argparse
import os
from pathlib import Path
from huggingface_hub import HfApi, create_repo

BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"

ARTIFACTS = [
    "BTC_gru.keras",
    "ETH_gru.keras",
    "LTC_gru.keras",
    "XRP_gru.keras",
    "BTC_scaler.pkl",
    "ETH_scaler.pkl",
    "LTC_scaler.pkl",
    "XRP_scaler.pkl",
]


def main():
    parser = argparse.ArgumentParser(description="Upload Crypto GRU models and scalers to Hugging Face Hub")
    parser.add_argument(
        "--repo-id",
        default=os.getenv("HF_REPO_ID"),
        help="Hugging Face Hub repo ID, e.g. 'username/crypto-time-series-gru'",
    )
    parser.add_argument(
        "--token",
        default=os.getenv("HF_TOKEN"),
        help="Hugging Face Write access token (or set HF_TOKEN env var / run `huggingface-cli login`)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create the repository as private (default: public)",
    )
    args = parser.parse_args()

    if not args.repo_id:
        parser.error("Please provide --repo-id or set the HF_REPO_ID environment variable.")

    api = HfApi(token=args.token)

    print(f"Ensuring Hugging Face model repository exists: {args.repo_id} ...")
    create_repo(
        repo_id=args.repo_id,
        token=args.token,
        repo_type="model",
        private=args.private,
        exist_ok=True,
    )

    model_card_path = MODELS_DIR / "MODEL_CARD.md"
    if model_card_path.exists():
        print("Uploading README.md (Model Card)...")
        api.upload_file(
            path_or_fileobj=str(model_card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="model",
        )

    for filename in ARTIFACTS:
        file_path = MODELS_DIR / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Missing model artifact: {file_path}")
        print(f"Uploading {filename} ...")
        api.upload_file(
            path_or_fileobj=str(file_path),
            path_in_repo=filename,
            repo_id=args.repo_id,
            repo_type="model",
        )

    print(f"\nSuccessfully uploaded all models and scalers to: https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
