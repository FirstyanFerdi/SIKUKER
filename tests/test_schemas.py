"""
Unit tests for Pydantic data schemas (app/schemas/applicant_schema.py).
Tests boundary validations, constraints, serialization, and invalid inputs.
"""

import pytest
from pydantic import ValidationError
from app.schemas.applicant_schema import (
    ApplicantInput,
    PredictionResult,
    ALL_FEATURES_ORDER,
)


@pytest.fixture
def valid_dict() -> dict:
    return {
        "gpa": 3.50,
        "test_score": 80.0,
        "experience_years": 5.0,
        "training_hours": 45.0,
        "city_development_index": 0.90,
        "education_level": "Graduate",
        "major_discipline": "STEM",
        "relevent_experience": "Has relevent experience",
        "enrolled_university": "no_enrollment",
    }


def test_valid_applicant_input(valid_dict):
    """Test valid applicant input creation and ordered dictionary conversion."""
    app = ApplicantInput(**valid_dict)
    assert app.gpa == 3.50
    ordered = app.to_dict_ordered()
    assert list(ordered.keys()) == ALL_FEATURES_ORDER


@pytest.mark.parametrize(
    "field,invalid_val",
    [
        ("gpa", 1.99),  # Below min 2.0
        ("gpa", 4.01),  # Above max 4.0
        ("test_score", -1.0),  # Below min 0.0
        ("test_score", 100.1),  # Above max 100.0
        ("experience_years", -0.5),  # Below min 0.0
        ("experience_years", 30.5),  # Above max 30.0
        ("training_hours", -1.0),  # Below min 0.0
        ("training_hours", 501.0),  # Above max 500.0
        ("city_development_index", 0.39),  # Below min 0.40
        ("city_development_index", 1.01),  # Above max 1.00
        ("education_level", "Diploma"),  # Not in valid literal
        ("major_discipline", "Law"),  # Not in valid literal
        ("relevent_experience", "Some experience"),  # Not in valid literal
        ("enrolled_university", "Online course"),  # Not in valid literal
    ],
)
def test_invalid_applicant_input_bounds(valid_dict, field, invalid_val):
    """Test that all out-of-boundary and invalid categorical inputs raise ValidationError."""
    broken = {**valid_dict, field: invalid_val}
    with pytest.raises(ValidationError):
        ApplicantInput(**broken)


def test_prediction_result_validation():
    """Test PredictionResult field validations and constraints."""
    res = PredictionResult(
        probability=0.85,
        percentage=85.0,
        status="Terima",
        strengths=["Kekuatan 1"],
        weaknesses=[],
    )
    assert res.status == "Terima"
    assert res.probability == 0.85

    # Test invalid probability (> 1.0)
    with pytest.raises(ValidationError):
        PredictionResult(probability=1.5, percentage=150.0, status="Terima")

    # Test invalid status
    with pytest.raises(ValidationError):
        PredictionResult(probability=0.5, percentage=50.0, status="Mungkin")
