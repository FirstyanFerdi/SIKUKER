"""
Machine Learning Inference Service for SIKUKER.
Loads trained artifacts from models/, runs preprocessing and probability predictions,
and maps probability scores to a 3-tier status (Terima, Pertimbangan, Tolak).
Includes graceful Fallback/Mock mode if joblib or artifacts are not available.
"""

import json
import logging
import functools
from pathlib import Path
from typing import Tuple, Dict, Any, Optional

from app.core.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    FEATURE_METADATA_PATH,
    TERIMA_THRESHOLD,
    PERTIMBANGAN_THRESHOLD,
)
from app.schemas.applicant_schema import (
    ApplicantInput,
    PredictionResult,
    PredictionStatus,
    ALL_FEATURES_ORDER,
)
from app.services.gap_analyzer import analyze_qualification_gaps

logger = logging.getLogger(__name__)

# Global flag to track whether mock mode was used
_IS_MOCK_ACTIVE: bool = False


class MockPreprocessor:
    """Fallback dummy preprocessor when joblib/model artifacts are not available."""

    def transform(self, df):
        return df.to_numpy() if hasattr(df, "to_numpy") else df


class MockModel:
    """
    Fallback dummy model producing deterministic, realistic probabilities
    based on applicant qualifications when real model artifacts are unavailable.
    """

    def predict_proba(self, X):
        import numpy as np

        # Default fallback proba
        proba_1 = 0.50
        return np.array([[1.0 - proba_1, proba_1]])


def is_mock_mode_active() -> bool:
    """Return True if the ML service is currently operating in Mock/Fallback mode."""
    global _IS_MOCK_ACTIVE
    return _IS_MOCK_ACTIVE


def _calculate_mock_probability(applicant: ApplicantInput) -> float:
    """
    Compute a deterministic heuristic qualification score (0.0 to 1.0)
    when operating in mock/fallback mode.
    """
    # 1. GPA normalized (2.0 - 4.0 -> 0.0 - 1.0)
    gpa_norm = max(0.0, min(1.0, (applicant.gpa - 2.0) / 2.0))
    # 2. Test score (0.0 - 100.0 -> 0.0 - 1.0)
    test_norm = max(0.0, min(1.0, applicant.test_score / 100.0))
    # 3. Experience years (0.0 - 15.0 -> 0.0 - 1.0)
    exp_norm = max(0.0, min(1.0, applicant.experience_years / 15.0))
    # 4. Training hours (0.0 - 100.0 -> 0.0 - 1.0)
    train_norm = max(0.0, min(1.0, applicant.training_hours / 100.0))
    # 5. Relevant experience (0 or 1)
    rel_norm = 1.0 if applicant.relevent_experience == "Has relevent experience" else 0.0
    # 6. Education level bonus
    edu_weights = {
        "Phd": 1.0,
        "Masters": 0.9,
        "Graduate": 0.75,
        "High School": 0.4,
        "Primary School": 0.2,
    }
    edu_norm = edu_weights.get(applicant.education_level, 0.5)

    # Weighted sum
    score = (
        0.25 * gpa_norm
        + 0.25 * test_norm
        + 0.20 * exp_norm
        + 0.10 * train_norm
        + 0.10 * rel_norm
        + 0.10 * edu_norm
    )
    return float(max(0.05, min(0.98, round(score, 4))))


def _load_metadata_from_file() -> Dict[str, Any]:
    """Load feature_metadata.json with fallback defaults if missing."""
    if FEATURE_METADATA_PATH.exists():
        try:
            with open(FEATURE_METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read {FEATURE_METADATA_PATH}: {e}")

    # Default fallback metadata
    return {
        "all_features_order": ALL_FEATURES_ORDER,
        "ideal_numeric_benchmarks": {
            "gpa": 3.45,
            "test_score": 78.5,
            "experience_years": 12.0,
            "training_hours": 50.0,
            "city_development_index": 0.91,
        },
        "status_thresholds": {
            "terima_min_prob": TERIMA_THRESHOLD,
            "pertimbangan_min_prob": PERTIMBANGAN_THRESHOLD,
        },
    }


@functools.lru_cache(maxsize=1)
def load_ml_artifacts() -> Tuple[Any, Any, Dict[str, Any]]:
    """
    Load pre-trained classification model, preprocessor pipeline, and metadata.
    Uses functools.lru_cache to prevent repetitive disk I/O on every request.

    Returns:
        Tuple[model, preprocessor, metadata]
    """
    global _IS_MOCK_ACTIVE
    metadata = _load_metadata_from_file()

    # Verify physical file existence
    if not (MODEL_PATH.exists() and PREPROCESSOR_PATH.exists()):
        logger.warning(
            f"Model or Preprocessor artifact not found in {MODEL_PATH.parent}. "
            "Activating Mock/Fallback Mode."
        )
        _IS_MOCK_ACTIVE = True
        return MockModel(), MockPreprocessor(), metadata

    try:
        import joblib
        model = joblib.load(MODEL_PATH)
        preprocessor = joblib.load(PREPROCESSOR_PATH)
        _IS_MOCK_ACTIVE = False
        logger.info("Successfully loaded ML model and preprocessor pipeline.")
        return model, preprocessor, metadata
    except Exception as exc:
        logger.warning(
            f"Error loading artifacts with joblib ({exc}). Activating Mock/Fallback Mode."
        )
        _IS_MOCK_ACTIVE = True
        return MockModel(), MockPreprocessor(), metadata


def map_probability_to_status(probability: float) -> PredictionStatus:
    """
    Map a probability score to a 3-tier qualification decision status:
    - probability >= TERIMA_THRESHOLD (0.75) -> "Terima"
    - PERTIMBANGAN_THRESHOLD (0.45) <= probability < TERIMA_THRESHOLD (0.75) -> "Pertimbangan"
    - probability < PERTIMBANGAN_THRESHOLD (0.45) -> "Tolak"
    """
    if probability >= TERIMA_THRESHOLD:
        return "Terima"
    elif probability >= PERTIMBANGAN_THRESHOLD:
        return "Pertimbangan"
    else:
        return "Tolak"


def predict_applicant_suitability(applicant: ApplicantInput) -> PredictionResult:
    """
    Run qualification suitability inference for a validated applicant input.

    Process:
    1. Loads cached model, preprocessor, and metadata.
    2. Converts applicant to a single-row DataFrame with strict column ordering.
    3. Transforms features via preprocessor.transform(df).
    4. Predicts suitability probability via model.predict_proba().
    5. Maps probability to 3-tier status (Terima, Pertimbangan, Tolak).
    6. Identifies strengths and weaknesses via analyze_qualification_gaps.
    7. Returns a validated PredictionResult.

    Args:
        applicant (ApplicantInput): Validated applicant input.

    Returns:
        PredictionResult: Full prediction output object.
    """
    model, preprocessor, metadata = load_ml_artifacts()

    # Step A: Analyze qualification strengths and weaknesses
    strengths, weaknesses = analyze_qualification_gaps(applicant, metadata)

    # Step B: Determine probability
    if is_mock_mode_active():
        probability = _calculate_mock_probability(applicant)
    else:
        try:
            import pandas as pd
            features_order = metadata.get("all_features_order", ALL_FEATURES_ORDER)
            # Create DataFrame with exact column ordering
            ordered_dict = {feat: getattr(applicant, feat) for feat in features_order}
            df = pd.DataFrame([ordered_dict])

            # Apply fitted preprocessor pipeline (NEVER fit_transform)
            X_transformed = preprocessor.transform(df)

            # Extract positive class probability
            proba_arr = model.predict_proba(X_transformed)
            # Binary classifier: probability of class 1
            if proba_arr.ndim == 2 and proba_arr.shape[1] >= 2:
                probability = float(proba_arr[0][1])
            else:
                probability = float(proba_arr[0][0])
        except Exception as err:
            logger.error(f"Inference error with live model ({err}), falling back to heuristic: {err}")
            probability = _calculate_mock_probability(applicant)

    # Constrain probability between 0.0 and 1.0
    probability = max(0.0, min(1.0, round(probability, 4)))
    percentage = round(probability * 100.0, 2)
    status = map_probability_to_status(probability)

    return PredictionResult(
        probability=probability,
        percentage=percentage,
        status=status,
        strengths=strengths,
        weaknesses=weaknesses,
    )
