"""
Pydantic schemas for applicant input validation and prediction results.
Strictly adheres to feature definitions in models/feature_metadata.json.
"""

from typing import Literal, List, Dict, Any
from pydantic import BaseModel, Field


# Categorical Option Types based on models/feature_metadata.json
EducationLevel = Literal[
    "Graduate",
    "Masters",
    "Phd",
    "High School",
    "Primary School",
]

MajorDiscipline = Literal[
    "STEM",
    "Business Degree",
    "Arts",
    "Humanities",
    "No Major",
    "Other",
]

RelevantExperience = Literal[
    "Has relevent experience",
    "No relevent experience",
]

EnrolledUniversity = Literal[
    "no_enrollment",
    "Full time course",
    "Part time course",
]

PredictionStatus = Literal[
    "Terima",
    "Pertimbangan",
    "Tolak",
]

# Explicit column ordering as expected by preprocessor pipeline and classifier
ALL_FEATURES_ORDER: List[str] = [
    "gpa",
    "test_score",
    "experience_years",
    "training_hours",
    "city_development_index",
    "education_level",
    "major_discipline",
    "relevent_experience",
    "enrolled_university",
]


class ApplicantInput(BaseModel):
    """
    Input schema validating the 9 qualification features required for inference.
    Feature order and validation constraints strictly follow models/feature_metadata.json.
    """

    # --- Numeric Features ---
    gpa: float = Field(
        ...,
        ge=2.0,
        le=4.0,
        description="Indeks Prestasi Kumulatif (IPK) kandidat (2.00 - 4.00)",
        examples=[3.5],
    )
    test_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Skor ujian teknis kandidat (0.0 - 100.0)",
        examples=[80.0],
    )
    experience_years: float = Field(
        ...,
        ge=0.0,
        le=30.0,
        description="Pengalaman kerja profesional dalam tahun (0.0 - 30.0)",
        examples=[5.0],
    )
    training_hours: float = Field(
        ...,
        ge=0.0,
        le=500.0,
        description="Total durasi pelatihan/kursus dalam jam (0.0 - 500.0)",
        examples=[60.0],
    )
    city_development_index: float = Field(
        ...,
        ge=0.4,
        le=1.0,
        description="Indeks perkembangan kota domisili kandidat (0.40 - 1.00)",
        examples=[0.92],
    )

    # --- Categorical Features ---
    education_level: EducationLevel = Field(
        ...,
        description="Tingkat pendidikan terakhir kandidat",
        examples=["Graduate"],
    )
    major_discipline: MajorDiscipline = Field(
        ...,
        description="Disiplin ilmu / jurusan kandidat",
        examples=["STEM"],
    )
    relevent_experience: RelevantExperience = Field(
        ...,
        description="Relevansi pengalaman kerja dengan bidang pekerjaan",
        examples=["Has relevent experience"],
    )
    enrolled_university: EnrolledUniversity = Field(
        ...,
        description="Status pendaftaran kuliah kandidat saat ini",
        examples=["no_enrollment"],
    )

    def to_dict_ordered(self) -> Dict[str, Any]:
        """
        Convert to dictionary with keys strictly adhering to all_features_order.
        """
        data = self.model_dump()
        return {feature: data[feature] for feature in ALL_FEATURES_ORDER}

    def to_dataframe(self):
        """
        Convert the validated input into a single-row pandas DataFrame
        with exact column ordering as required by the preprocessing pipeline.
        """
        import pandas as pd
        return pd.DataFrame([self.to_dict_ordered()])

    model_config = {
        "json_schema_extra": {
            "example": {
                "gpa": 3.65,
                "test_score": 85.0,
                "experience_years": 8.0,
                "training_hours": 75.0,
                "city_development_index": 0.92,
                "education_level": "Graduate",
                "major_discipline": "STEM",
                "relevent_experience": "Has relevent experience",
                "enrolled_university": "no_enrollment",
            }
        }
    }


class PredictionResult(BaseModel):
    """
    Output schema containing classification probability, percentage,
    3-tier status, along with strengths and weaknesses identified.
    """

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probabilitas kelayakan kandidat (0.0 - 1.0)",
        examples=[0.82],
    )
    percentage: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Persentase kelayakan kandidat (0.0 - 100.0%)",
        examples=[82.0],
    )
    status: PredictionStatus = Field(
        ...,
        description="Status keputusan 3-tier: 'Terima', 'Pertimbangan', atau 'Tolak'",
        examples=["Terima"],
    )
    strengths: List[str] = Field(
        default_factory=list,
        description="Daftar keunggulan kualifikasi kandidat",
        examples=[["IPK 3.65 berada di atas benchmark kelulusan (3.45)."]],
    )
    weaknesses: List[str] = Field(
        default_factory=list,
        description="Daftar kesenjangan kualifikasi kandidat",
        examples=[[]],
    )
