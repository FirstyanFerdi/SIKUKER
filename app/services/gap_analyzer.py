"""
Qualification Gap Analyzer Service for SIKUKER.
Compares applicant qualifications against numeric benchmarks and categorical rules
to identify strengths and areas of development (weaknesses).
"""

from typing import Tuple, List, Dict, Any
from app.schemas.applicant_schema import ApplicantInput


def analyze_qualification_gaps(
    applicant: ApplicantInput,
    metadata: Dict[str, Any]
) -> Tuple[List[str], List[str]]:
    """
    Analyze qualification strengths and gaps/weaknesses of an applicant.

    Args:
        applicant (ApplicantInput): Validated applicant input data.
        metadata (Dict[str, Any]): Feature metadata containing 'ideal_numeric_benchmarks'.

    Returns:
        Tuple[List[str], List[str]]:
            - strengths: List of bullet points describing qualification strengths in Indonesian.
            - weaknesses: List of bullet points describing qualification gaps/weaknesses in Indonesian.
    """
    strengths: List[str] = []
    weaknesses: List[str] = []

    # Extract numeric benchmarks with safe fallbacks based on models/feature_metadata.json
    benchmarks = metadata.get("ideal_numeric_benchmarks", {})
    gpa_bench = float(benchmarks.get("gpa", 3.45))
    test_score_bench = float(benchmarks.get("test_score", 78.5))
    exp_years_bench = float(benchmarks.get("experience_years", 12.0))
    train_hours_bench = float(benchmarks.get("training_hours", 50.0))
    cdi_bench = float(benchmarks.get("city_development_index", 0.91))

    # --- 1. Numeric Feature Evaluation ---

    # GPA / IPK
    if applicant.gpa >= gpa_bench:
        strengths.append(
            f"IPK ({applicant.gpa:.2f}) unggul dan memenuhi benchmark kelulusan ({gpa_bench:.2f})."
        )
    else:
        gap = gpa_bench - applicant.gpa
        weaknesses.append(
            f"IPK ({applicant.gpa:.2f}) masih berada di bawah target benchmark ({gpa_bench:.2f}), selisih -{gap:.2f} poin."
        )

    # Test Score / Skor Ujian Teknis
    if applicant.test_score >= test_score_bench:
        strengths.append(
            f"Skor tes teknis ({applicant.test_score:.1f}) sangat kompetitif, melampaui benchmark ({test_score_bench:.1f})."
        )
    else:
        gap = test_score_bench - applicant.test_score
        weaknesses.append(
            f"Skor tes teknis ({applicant.test_score:.1f}) di bawah standar kelulusan ({test_score_bench:.1f}), selisih -{gap:.1f} poin."
        )

    # Experience Years / Pengalaman Kerja
    if applicant.experience_years >= exp_years_bench:
        strengths.append(
            f"Pengalaman kerja ({applicant.experience_years:.1f} tahun) sangat matang, melampaui benchmark ({exp_years_bench:.1f} tahun)."
        )
    else:
        gap = exp_years_bench - applicant.experience_years
        weaknesses.append(
            f"Durasi pengalaman kerja ({applicant.experience_years:.1f} tahun) masih di bawah benchmark ({exp_years_bench:.1f} tahun), selisih -{gap:.1f} tahun."
        )

    # Training Hours / Jam Pelatihan
    if applicant.training_hours >= train_hours_bench:
        strengths.append(
            f"Total jam pelatihan ({applicant.training_hours:.0f} jam) menunjukkan komitmen upskilling yang tinggi (benchmark: {train_hours_bench:.0f} jam)."
        )
    else:
        gap = train_hours_bench - applicant.training_hours
        weaknesses.append(
            f"Total jam pelatihan ({applicant.training_hours:.0f} jam) masih di bawah target benchmark ({train_hours_bench:.0f} jam), selisih -{gap:.0f} jam."
        )

    # City Development Index / Ekosistem Kota
    if applicant.city_development_index >= cdi_bench:
        strengths.append(
            f"Indeks perkembangan kota domisili ({applicant.city_development_index:.2f}) berada di ekosistem industri yang sangat maju (benchmark: {cdi_bench:.2f})."
        )
    else:
        gap = cdi_bench - applicant.city_development_index
        weaknesses.append(
            f"Indeks perkembangan kota domisili ({applicant.city_development_index:.2f}) di bawah benchmark ekosistem industri ({cdi_bench:.2f}), selisih -{gap:.2f}."
        )

    # --- 2. Categorical Feature Evaluation ---

    # Relevansi Pengalaman Kerja
    if applicant.relevent_experience == "Has relevent experience":
        strengths.append("Memiliki rekam jejak pengalaman kerja yang relevan langsung dengan bidang kualifikasi.")
    else:
        weaknesses.append("Belum memiliki pengalaman kerja yang relevan langsung di bidang kualifikasi terkait.")

    # Tingkat Pendidikan Terakhir
    if applicant.education_level in ["Masters", "Phd"]:
        strengths.append(
            f"Tingkat pendidikan pascasarjana ({applicant.education_level}) memberikan fondasi riset dan analitis yang sangat kuat."
        )
    elif applicant.education_level == "Graduate":
        strengths.append("Tingkat pendidikan Sarjana (Graduate/S1) memenuhi standar kualifikasi formal yang disyaratkan.")
    else:
        # High School / Primary School
        weaknesses.append(
            f"Tingkat pendidikan terakhir ({applicant.education_level}) masih di bawah standar pendidikan sarjana (Graduate)."
        )

    # Disiplin Ilmu / Jurusan
    if applicant.major_discipline in ["STEM", "Business Degree"]:
        strengths.append(
            f"Latar belakang jurusan ({applicant.major_discipline}) sangat selaras dan mendukung peran teknis/analitik."
        )
    elif applicant.major_discipline in ["No Major", "Other"]:
        weaknesses.append(
            f"Latar belakang disiplin ilmu ({applicant.major_discipline}) belum terfokus atau kurang linear dengan bidang teknis yang dibutuhkan."
        )

    # Status Kuliah / Komitmen Waktu
    if applicant.enrolled_university == "Full time course":
        weaknesses.append(
            "Sedang terdaftar studi penuh waktu (Full time course), berpotensi membatasi fleksibilitas jam kerja reguler."
        )
    elif applicant.enrolled_university == "no_enrollment":
        strengths.append(
            "Tidak terikat jadwal perkuliahan aktif (no_enrollment), siap berkomitmen penuh waktu untuk peran profesional."
        )

    return strengths, weaknesses
