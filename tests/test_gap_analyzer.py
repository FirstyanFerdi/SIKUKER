"""
Unit tests for Qualification Gap Analyzer Service (app/services/gap_analyzer.py).
Tests numeric benchmark comparisons and categorical qualification rules.
"""

import pytest
from app.schemas.applicant_schema import ApplicantInput
from app.services.gap_analyzer import analyze_qualification_gaps


@pytest.fixture
def sample_metadata() -> dict:
    return {
        "ideal_numeric_benchmarks": {
            "gpa": 3.45,
            "test_score": 78.5,
            "experience_years": 12.0,
            "training_hours": 50.0,
            "city_development_index": 0.91,
        }
    }


def test_gap_analyzer_identifies_numeric_weaknesses(sample_metadata):
    """Verify that values below ideal benchmarks are properly registered in weaknesses."""
    applicant = ApplicantInput(
        gpa=2.50,  # Below benchmark 3.45
        test_score=55.0,  # Below benchmark 78.5
        experience_years=2.0,  # Below benchmark 12.0
        training_hours=20.0,  # Below benchmark 50.0
        city_development_index=0.60,  # Below benchmark 0.91
        education_level="Graduate",
        major_discipline="STEM",
        relevent_experience="Has relevent experience",
        enrolled_university="no_enrollment",
    )

    strengths, weaknesses = analyze_qualification_gaps(applicant, sample_metadata)

    # Convert weaknesses to lowercase for robust assertion
    weaknesses_str = " ".join(weaknesses).lower()

    assert "ipk" in weaknesses_str
    assert "skor tes" in weaknesses_str
    assert "pengalaman kerja" in weaknesses_str
    assert "jam pelatihan" in weaknesses_str
    assert "kota domisili" in weaknesses_str or "indeks perkembangan" in weaknesses_str


def test_gap_analyzer_identifies_numeric_strengths(sample_metadata):
    """Verify that values at or above ideal benchmarks are properly registered in strengths."""
    applicant = ApplicantInput(
        gpa=3.80,  # Above 3.45
        test_score=88.0,  # Above 78.5
        experience_years=15.0,  # Above 12.0
        training_hours=90.0,  # Above 50.0
        city_development_index=0.95,  # Above 0.91
        education_level="Masters",
        major_discipline="STEM",
        relevent_experience="Has relevent experience",
        enrolled_university="no_enrollment",
    )

    strengths, weaknesses = analyze_qualification_gaps(applicant, sample_metadata)
    strengths_str = " ".join(strengths).lower()

    assert "ipk" in strengths_str
    assert "skor tes" in strengths_str
    assert "pengalaman kerja" in strengths_str
    assert "jam pelatihan" in strengths_str


def test_gap_analyzer_categorical_rules(sample_metadata):
    """Verify categorical evaluation logic for non-relevant experience and education levels."""
    # Case 1: Non-relevant experience and High School education
    weak_cat = ApplicantInput(
        gpa=3.50,
        test_score=80.0,
        experience_years=5.0,
        training_hours=60.0,
        city_development_index=0.92,
        education_level="High School",
        major_discipline="No Major",
        relevent_experience="No relevent experience",
        enrolled_university="Full time course",
    )

    strengths, weaknesses = analyze_qualification_gaps(weak_cat, sample_metadata)
    weaknesses_str = " ".join(weaknesses).lower()

    assert "belum memiliki pengalaman kerja yang relevan" in weaknesses_str
    assert "di bawah standar pendidikan sarjana" in weaknesses_str
    assert "studi penuh waktu" in weaknesses_str or "full time course" in weaknesses_str


def test_gap_analyzer_handles_empty_metadata():
    """Verify that analyzer falls back gracefully to internal default benchmarks if metadata is empty."""
    applicant = ApplicantInput(
        gpa=2.10,
        test_score=40.0,
        experience_years=1.0,
        training_hours=10.0,
        city_development_index=0.50,
        education_level="Primary School",
        major_discipline="Other",
        relevent_experience="No relevent experience",
        enrolled_university="Part time course",
    )

    strengths, weaknesses = analyze_qualification_gaps(applicant, {})
    assert len(weaknesses) > 0
    assert any("ipk" in w.lower() for w in weaknesses)
