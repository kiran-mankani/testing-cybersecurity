"""Settings read from environment variables (or a .env file in the project root)."""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://127.0.0.1:27017")
MONGODB_DATABASE = os.getenv("MONGODB_DATABASE", "smart_cybersecurity")
MONGODB_TIMEOUT_MS = int(os.getenv("MONGODB_TIMEOUT_MS", "2000"))

# Relative paths are resolved against the project root.
MODEL_BUNDLE_PATH = PROJECT_ROOT / os.getenv(
    "MODEL_BUNDLE_PATH",
    "models/deployment/standard_weighted_xgboost_bundle.pkl",
)
RAW_TEST_CSV = PROJECT_ROOT / "data" / "raw" / "kdd_test.csv"
FAIR_COMPARISON_DIR = PROJECT_ROOT / "results" / "fair_comparison"

MAX_BATCH_ROWS = int(os.getenv("MAX_BATCH_ROWS", "5000"))
