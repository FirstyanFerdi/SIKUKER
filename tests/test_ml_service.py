"""
Unit tests for Machine Learning Inference Service (app/services/ml_service.py).
Tests artifact loading, probability ranges, 3-tier status mapping, and heuristic fallback.
"""

import pytest
from app.schemas.applicant_schema import ApplicantInput, PredictionResult, ALL_FEATURES_ORDER
from app.services.ml_service import (
    load_ml_artifacts,
    predict_applicant_suitability,
    map_probability_to_status,
    is_mock_mode_active,
)
from app.core.config import TERIMA_THRESHOLD, PERTIMBANGAN_THRESHOLD


@pytest.fixture
def strong_applicant() -> ApplicantInput:
    """Fixture representing a high-caliber candidate expected to be Accepted (Terima)."""
    return ApplicantInput(
        gpa=3.85,
        test_score=92.0,
        experience_years=14.0,
        training_hours=85.0,
        city_development_index=0.92,
        education_level="Masters",
        major_discipline="STEM",
        relevent_experience="Has relevent experience",
        enrolled_university="no_enrollment",
    )


@pytest.fixture
def moderate_applicant() -> ApplicantInput:
    """Fixture representing a borderline candidate expected to be In Consideration (Pertimbangan)."""
    return ApplicantInput(
        gpa=3.30,
        test_score=68.0,
        experience_years=4.0,
        training_hours=35.0,
        city_development_index=0.82,
        education_level="Graduate",
        major_discipline="STEM",
        relevent_experience="Has relevent experience",
        enrolled_university="no_enrollment",
    )


@pytest.fixture
def weak_applicant() -> ApplicantInput:
    """Fixture representing an under-qualified candidate expected to be Rejected (Tolak)."""
    return ApplicantInput(
        gpa=2.20,
        test_score=42.0,
        experience_years=0.5,
        training_hours=10.0,
        city_development_index=0.55,
        education_level="High School",
        major_discipline="No Major",
        relevent_experience="No relevent experience",
        enrolled_university="Full time course",
    )


def test_load_ml_artifacts():
    """Verify that model artifacts load correctly or fallback cleanly without exceptions."""
    model, preprocessor, metadata = load_ml_artifacts()
    assert model is not None, "Model must not be None"
    assert preprocessor is not None, "Preprocessor must not be None"
    assert isinstance(metadata, dict), "Metadata must be a dictionary"
    assert "all_features_order" in metadata, "Metadata must contain 'all_features_order'"
    assert metadata["all_features_order"] == ALL_FEATURES_ORDER


def test_map_probability_to_status():
    """Verify exact 3-tier threshold boundary mappings."""
    # Boundary: TERIMA (>= 0.75)
    assert map_probability_to_status(1.0) == "Terima"
    assert map_probability_to_status(0.75) == "Terima"
    assert map_probability_to_status(TERIMA_THRESHOLD) == "Terima"

    # Boundary: PERTIMBANGAN (0.45 <= P < 0.75)
    assert map_probability_to_status(0.7499) == "Pertimbangan"
    assert map_probability_to_status(0.60) == "Pertimbangan"
    assert map_probability_to_status(0.45) == "Pertimbangan"
    assert map_probability_to_status(PERTIMBANGAN_THRESHOLD) == "Pertimbangan"

    # Boundary: TOLAK (< 0.45)
    assert map_probability_to_status(0.4499) == "Tolak"
    assert map_probability_to_status(0.10) == "Tolak"
    assert map_probability_to_status(0.0) == "Tolak"


def test_predict_strong_candidate(strong_applicant):
    """Test that a strong candidate receives 'Terima' status with probability >= 0.75."""
    result: PredictionResult = predict_applicant_suitability(strong_applicant)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.probability <= 1.0, "Probability must be between 0.0 and 1.0"
    assert 0.0 <= result.percentage <= 100.0, "Percentage must be between 0.0 and 100.0"
    assert result.probability >= TERIMA_THRESHOLD, f"Expected probability >= 0.75, got {result.probability}"
    assert result.status == "Terima", f"Expected status 'Terima', got '{result.status}'"
    assert len(result.strengths) > 0, "Strong candidate should have identified strengths"


def test_predict_moderate_candidate(moderate_applicant):
    """Test that a moderate candidate receives 'Pertimbangan' status (0.45 <= P < 0.75)."""
    result: PredictionResult = predict_applicant_suitability(moderate_applicant)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.status == "Pertimbangan", f"Expected status 'Pertimbangan', got '{result.status}'"
    assert PERTIMBANGAN_THRESHOLD <= result.probability < TERIMA_THRESHOLD


def test_predict_weak_candidate(weak_applicant):
    """Test that an under-qualified candidate receives 'Tolak' status with probability < 0.45."""
    result: PredictionResult = predict_applicant_suitability(weak_applicant)

    assert isinstance(result, PredictionResult)
    assert 0.0 <= result.probability <= 1.0
    assert result.status == "Tolak", f"Expected status 'Tolak', got '{result.status}'"
    assert result.probability < PERTIMBANGAN_THRESHOLD
    assert len(result.weaknesses) > 0, "Weak candidate should have identified qualification gaps"


def test_applicant_dataframe_structure(strong_applicant):
    """Test that applicant input converts to a DataFrame with exact column names and order."""
    df = strong_applicant.to_dataframe()
    assert len(df) == 1, "DataFrame must have exactly 1 row"
    assert list(df.columns) == ALL_FEATURES_ORDER, "Columns must exactly match ALL_FEATURES_ORDER"
