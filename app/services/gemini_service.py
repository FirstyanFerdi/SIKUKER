"""
Gemini AI Career Feedback Service for SIKUKER.
Uses the official Google GenAI SDK (google-genai) to generate structured,
empathetic, and actionable career guidance in Indonesian.
Includes a deterministic fallback template if the API key is missing or offline.
"""

import os
import logging
from typing import Optional

from google import genai
from google.genai import types

from app.core.config import GEMINI_API_KEY, GEMINI_MODEL
from app.schemas.applicant_schema import ApplicantInput, PredictionResult
from app.utils.prompt_templates import (
    SYSTEM_INSTRUCTION,
    build_career_feedback_prompt,
)

logger = logging.getLogger(__name__)


def generate_fallback_feedback(
    applicant_name: str,
    target_role: str,
    applicant: ApplicantInput,
    prediction: PredictionResult,
    error_message: Optional[str] = None,
) -> str:
    """
    Generate a high-quality, structured deterministic Markdown feedback report
    when the Gemini API is inaccessible or GEMINI_API_KEY is not configured.

    Args:
        applicant_name (str): Candidate's name.
        target_role (str): Desired position.
        applicant (ApplicantInput): Validated applicant input.
        prediction (PredictionResult): ML prediction results and qualification gaps.
        error_message (Optional[str]): Reason for fallback, if applicable.

    Returns:
        str: Comprehensive 3-section Markdown feedback.
    """
    clean_name = applicant_name.strip() if applicant_name.strip() else "Kandidat"
    clean_role = target_role.strip() if target_role.strip() else "Data Scientist / Machine Learning Engineer"

    # Status-specific opening paragraph for Section 1
    if prediction.status == "Terima":
        opening_summary = (
            f"Selamat! Berdasarkan hasil evaluasi kualifikasi menggunakan model Machine Learning SIKUKER, "
            f"profil Anda dinyatakan **Sangat Sesuai (Terima)** untuk posisi **{clean_role}** "
            f"dengan skor kelayakan mencapai **{prediction.percentage:.1f}%**. "
            f"Kombinasi latar belakang pendidikan ({applicant.education_level}), rekam jejak pengalaman, "
            f"serta hasil asesmen teknis Anda telah melampaui benchmark kandidat unggulan kami. "
            f"Tim rekrutmen kami merekomendasikan Anda untuk segera melanjutkan ke tahapan wawancara akhir."
        )
    elif prediction.status == "Pertimbangan":
        opening_summary = (
            f"Terima kasih atas dedikasi dan minat Anda terhadap posisi **{clean_role}**. "
            f"Berdasarkan evaluasi kualifikasi Machine Learning SIKUKER, profil Anda saat ini berada dalam kategori "
            f"**Dalam Pertimbangan (Review Lanjutan)** dengan skor kesesuaian **{prediction.percentage:.1f}%**. "
            f"Anda memiliki fondasi kompetensi yang menjanjikan, namun terdapat beberapa aspek spesifik yang membutuhkan "
            f"verifikasi portofolio atau tes komplementer sebelum keputusan final diambil."
        )
    else:  # Tolak
        opening_summary = (
            f"Terima kasih telah meluangkan waktu dan menunjukkan antusiasme untuk posisi **{clean_role}**. "
            f"Setelah melalui evaluasi objektif berbasis Machine Learning SIKUKER, saat ini profil Anda "
            f"berada pada status **Belum Sesuai (Tolak)** dengan skor kelayakan **{prediction.percentage:.1f}%**. "
            f"Keputusan ini semata-mata mencerminkan keselarasan kualifikasi saat ini terhadap standar minimum posisi yang sangat ketat, "
            f"bukan penilaian atas potensi jangka panjang Anda. Kami sangat mendorong Anda untuk memanfaatkan rekomendasi berikut "
            f"sebagai pijakan akselerasi karir ke depan."
        )

    # Construct Section 2: Gap Analysis breakdown
    if prediction.weaknesses:
        gap_points = "\n".join([f"- **Kesenjangan Kualifikasi:** {w}" for w in prediction.weaknesses])
        gap_explanation = (
            f"Berdasarkan perbandingan profil Anda terhadap benchmark kandidat ideal pada posisi {clean_role}, "
            f"berikut adalah poin-poin kesenjangan utama yang menjadi perhatian tim evaluator:\n\n"
            f"{gap_points}\n\n"
            f"*Catatan Evaluator:* Pada industri teknologi yang kompetitif, pemenuhan benchmark teknis dan "
            f"durasi pelatihan intensif sangat berpengaruh terhadap kesiapan kerja langsung (job readiness) di lingkungan produksi."
        )
    else:
        gap_explanation = (
            f"Tidak ditemukan kesenjangan kualifikasi yang signifikan pada profil Anda terhadap benchmark standar posisi {clean_role}. "
            f"Seluruh indikator numerik maupun kategorikal Anda berada pada rentang ideal kandidat yang diterima."
        )

    # Construct Section 3: Targeted Upskilling Recommendations based on identified weaknesses
    recommendations = []
    
    # 1. Technical / Test score recommendation
    if applicant.test_score < 78.5:
        recommendations.append(
            "- **Pendalaman Fondasi Teknis & Asesmen Koding:** Latih kemampuan problem-solving dan algoritma terapan "
            "pada platform seperti HackerRank (SQL & Python Intermediate) atau LeetCode. Targetkan skor evaluasi minimal 80+."
        )
    else:
        recommendations.append(
            "- **Penguatan Arsitektur Sistem Machine Learning:** Pelajari *Machine Learning System Design* dan *MLOps* "
            "(Docker, MLflow, CI/CD for ML) untuk membuktikan kesiapan menangani pipeline data skala besar."
        )

    # 2. Training / Courses recommendation
    if applicant.training_hours < 50.0:
        recommendations.append(
            "- **Peningkatan Jam Pelatihan Intensif (Upskilling):** Ikuti kursus terstruktur berbasis sertifikasi, "
            "seperti *DeepLearning.AI Machine Learning Specialization* atau *Google Professional Data Engineer* "
            "untuk melengkapi jam pelatihan hingga mencapai minimal 50-100 jam terarah."
        )
    else:
        recommendations.append(
            "- **Sertifikasi Spesialisasi Lanjutan:** Lengkapi jam pelatihan yang telah kuat dengan sertifikasi berstandar industri "
            "seperti *AWS Certified Machine Learning - Specialty* atau *TensorFlow Developer Certificate*."
        )

    # 3. Experience & Portfolio recommendation
    if applicant.relevent_experience == "No relevent experience" or applicant.experience_years < 3.0:
        recommendations.append(
            "- **Bangun Portofolio End-to-End di GitHub:** Ciptakan 2-3 proyek terapan nyata (bukan sekadar tutorial standar), "
            "misalnya membangun API klasifikasi berbasis FastAPI/Streamlit, lengkap dengan dokumentasi arsitektur dan container Docker."
        )
    else:
        recommendations.append(
            "- **Kontribusi Komunitas & Open Source:** Tunjukkan kepemimpinan teknis dengan berkontribusi pada proyek open-source "
            "atau menulis artikel teknis (Medium / Substack) mengenai optimasi model dan pemecahan masalah data di dunia nyata."
        )

    # 4. Academic / Learning discipline
    if applicant.gpa < 3.45:
        recommendations.append(
            "- **Kompensasi Akademik melalui Bukti Praktis:** Karena IPK berada di bawah benchmark, jadikan portofolio nyata, "
            "peringkat kompetisi Kaggle, dan kontribusi proyek riil sebagai pembuktian utama kompetensi analitis Anda kepada recruiter."
        )

    rec_text = "\n".join(recommendations)

    fallback_banner = (
        "> **Catatan Sistem:** Umpan balik ini disusun menggunakan *Local Deterministic Template* "
        f"({'Kunci GEMINI_API_KEY belum dikonfigurasi di file .env' if not GEMINI_API_KEY or 'your_gemini' in GEMINI_API_KEY else f'Layanan API sedang tidak dapat dijangkau: {error_message}'}). "
        "Seluruh evaluasi tetap dihitung secara presisi berdasarkan hasil Machine Learning dan aturan Gap Analyzer.\n\n"
    )

    return f"""{fallback_banner}### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir
Kepada Yth. **{clean_name}**,

{opening_summary}

Salam hangat,  
**Tim Rekrutmen & Pengembangan Bakat SIKUKER**

---

### 2. Analisis Kesenjangan Kualifikasi (Gap Analysis)
{gap_explanation}

---

### 3. Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (Upskilling)
Berikut adalah rekomendasi prioritas yang dirancang khusus untuk memperkuat kualifikasi Anda dalam 3 hingga 6 bulan ke depan:

{rec_text}
"""


def generate_career_feedback(
    applicant_name: str,
    target_role: str,
    applicant: ApplicantInput,
    prediction: PredictionResult,
) -> str:
    """
    Generate comprehensive career feedback and upskilling recommendations.
    Uses Google Gemini API via official google-genai SDK, with automatic fallback
    handling if API key is unconfigured or request fails.

    Args:
        applicant_name (str): Candidate's name.
        target_role (str): Target job role.
        applicant (ApplicantInput): Validated applicant input.
        prediction (PredictionResult): ML prediction outcome.

    Returns:
        str: Structured Markdown career feedback.
    """
    api_key = os.getenv("GEMINI_API_KEY", GEMINI_API_KEY).strip()

    # Case A: Missing or placeholder API key -> Graceful Fallback
    if not api_key or "your_gemini_api_key" in api_key.lower():
        logger.info("GEMINI_API_KEY is not set or using placeholder. Returning deterministic fallback feedback.")
        return generate_fallback_feedback(
            applicant_name=applicant_name,
            target_role=target_role,
            applicant=applicant,
            prediction=prediction,
            error_message="API Key belum dikonfigurasi.",
        )

    # Case B: Call Google GenAI SDK with Timeout & Model Resiliency
    prompt = build_career_feedback_prompt(
        applicant_name=applicant_name,
        target_role=target_role,
        applicant=applicant,
        prediction=prediction,
    )

    # Candidate models in priority order to handle deprecated/unavailable model IDs gracefully
    primary_model = os.getenv("GEMINI_MODEL", GEMINI_MODEL).strip() or "gemini-3.1-flash-lite"
    
    # Auto-resolve deprecated/legacy Google model names directly to their modern active successor
    # to avoid unnecessary network latency and 404 errors.
    DEPRECATED_MODEL_MAP = {
        "gemini-2.5-flash": "gemini-3.1-flash-lite",
        "gemini-2.5-flash-lite": "gemini-3.1-flash-lite",
        "gemini-2.0-flash": "gemini-3.1-flash-lite",
        "gemini-1.5-flash": "gemini-3.1-flash-lite",
    }
    resolved_primary = DEPRECATED_MODEL_MAP.get(primary_model, primary_model)

    candidate_models = [
        resolved_primary,
        "gemini-3.1-flash-lite",
        "gemini-flash-lite-latest",
        "gemini-3.5-flash-lite",
        "gemini-3.8-flash",
    ]
    # De-duplicate while preserving order
    seen = set()
    unique_models = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_error: Optional[Exception] = None

    for model_name in unique_models:
        try:
            # Note: In google-genai SDK, HttpOptions.timeout is in MILLISECONDS.
            # Setting 25,000 ms (25 seconds) gives ample time for generation without premature timeout.
            client = genai.Client(
                api_key=api_key,
                http_options=types.HttpOptions(timeout=25000),
            )
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                ),
            )

            if response and response.text:
                logger.info(f"Successfully generated feedback with Gemini model: {model_name}")
                return response.text.strip()
            else:
                raise ValueError("Respons Gemini kosong.")

        except Exception as exc:
            last_error = exc
            logger.warning(
                f"Attempt with model '{model_name}' failed ({exc}). Trying next candidate if available."
            )
            continue

    logger.error(f"All Gemini model attempts failed ({last_error}). Triggering deterministic fallback feedback.")
    return generate_fallback_feedback(
        applicant_name=applicant_name,
        target_role=target_role,
        applicant=applicant,
        prediction=prediction,
        error_message=str(last_error),
    )

