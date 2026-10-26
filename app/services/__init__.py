"""
Services module for SIKUKER application.
Contains business logic for ML inference, qualification gap analysis, and Gemini API integration.
"""

from app.services.gap_analyzer import analyze_qualification_gaps
from app.services.ml_service import (
    load_ml_artifacts,
    predict_applicant_suitability,
    is_mock_mode_active,
)
from app.services.gemini_service import (
    generate_career_feedback,
    generate_fallback_feedback,
)

__all__ = [
    "analyze_qualification_gaps",
    "load_ml_artifacts",
    "predict_applicant_suitability",
    "is_mock_mode_active",
    "generate_career_feedback",
    "generate_fallback_feedback",
]
