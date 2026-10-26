"""
Unit tests for Gemini Career Feedback Service (app/services/gemini_service.py).
Tests prompt construction, fallback template generation, and resilience against missing API keys.
"""

import os
from unittest.mock import patch
import pytest
from app.schemas.applicant_schema import ApplicantInput, PredictionResult
from app.utils.prompt_templates import SYSTEM_INSTRUCTION, build_career_feedback_prompt
from app.services.gemini_service import (
    generate_career_feedback,
    generate_fallback_feedback,
)


@pytest.fixture
def sample_applicant() -> ApplicantInput:
    return ApplicantInput(
        gpa=3.20,
        test_score=65.0,
        experience_years=3.0,
        training_hours=25.0,
        city_development_index=0.80,
        education_level="Graduate",
        major_discipline="STEM",
        relevent_experience="Has relevent experience",
        enrolled_university="no_enrollment",
    )


@pytest.fixture
def sample_prediction() -> PredictionResult:
    return PredictionResult(
        probability=0.55,
        percentage=55.0,
        status="Pertimbangan",
        strengths=["Latar belakang STEM linear dengan kebutuhan data."],
        weaknesses=[
            "IPK (3.20) di bawah benchmark (3.45), selisih -0.25 poin.",
            "Jam pelatihan (25 jam) di bawah benchmark (50 jam), selisih -25 jam.",
        ],
    )


def test_system_instruction():
    """Verify that SYSTEM_INSTRUCTION defines the HR Consultant role and professional Indonesian tone."""
    assert "Konsultan Karir HR Senior" in SYSTEM_INSTRUCTION
    assert "Bahasa Indonesia" in SYSTEM_INSTRUCTION


def test_build_career_feedback_prompt(sample_applicant, sample_prediction):
    """Verify that prompt builder injects candidate data, status, and 3 mandatory sections."""
    prompt = build_career_feedback_prompt(
        applicant_name="Budi Santoso",
        target_role="Data Scientist",
        applicant=sample_applicant,
        prediction=sample_prediction,
    )

    assert "Budi Santoso" in prompt
    assert "Data Scientist" in prompt
    assert "Pertimbangan" in prompt
    assert "55.00%" in prompt
    assert "### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir" in prompt
    assert "### 2. Analisis Kesenjangan Kualifikasi (Gap Analysis)" in prompt
    assert "### 3. Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (Upskilling)" in prompt


def test_generate_fallback_feedback_sections(sample_applicant, sample_prediction):
    """Verify that deterministic fallback feedback contains all 3 required sections."""
    feedback = generate_fallback_feedback(
        applicant_name="Budi Santoso",
        target_role="Data Scientist",
        applicant=sample_applicant,
        prediction=sample_prediction,
    )

    assert "### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir" in feedback
    assert "### 2. Analisis Kesenjangan Kualifikasi (Gap Analysis)" in feedback
    assert "### 3. Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (Upskilling)" in feedback
    assert "Budi Santoso" in feedback
    assert "Pertimbangan" in feedback


def test_generate_career_feedback_fallback_on_missing_key(sample_applicant, sample_prediction):
    """Verify that when GEMINI_API_KEY is empty, fallback feedback is returned without throwing error."""
    with patch.dict(os.environ, {"GEMINI_API_KEY": ""}):
        feedback = generate_career_feedback(
            applicant_name="Budi Santoso",
            target_role="Data Scientist",
            applicant=sample_applicant,
            prediction=sample_prediction,
        )

        assert feedback is not None
        assert len(feedback) > 100
        assert "### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir" in feedback
        assert "Catatan Sistem" in feedback
