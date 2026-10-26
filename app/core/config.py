"""
Configuration module for SIKUKER application.
Loads environment variables and defines paths and threshold constants.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory (SIKUKER root)
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load environment variables from .env file in project root
ENV_FILE_PATH = BASE_DIR / ".env"
load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)

# Models directory and model artifact paths
MODELS_DIR = BASE_DIR / "models"
MODEL_PATH = MODELS_DIR / "classifier_model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor_pipeline.joblib"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.json"

# Prediction Status Probability Thresholds (3-Tier Decision)
# probability >= TERIMA_THRESHOLD -> "Terima"
# PERTIMBANGAN_THRESHOLD <= probability < TERIMA_THRESHOLD -> "Pertimbangan"
# probability < PERTIMBANGAN_THRESHOLD -> "Tolak"
TERIMA_THRESHOLD: float = 0.75
PERTIMBANGAN_THRESHOLD: float = 0.45

# Google Gemini API configuration
GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()
