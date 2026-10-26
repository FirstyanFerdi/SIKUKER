"""
SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja & Umpan Balik Karir Berbasis AI.
Streamlit Main Application UI.
Strictly acts as a presentation layer, consuming services from app/services/ and schemas from app/schemas/.
"""

import sys
from pathlib import Path
from datetime import datetime
import streamlit as st

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import (
    MODEL_PATH,
    FEATURE_METADATA_PATH,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    TERIMA_THRESHOLD,
    PERTIMBANGAN_THRESHOLD,
)
from app.schemas.applicant_schema import (
    ApplicantInput,
    PredictionResult,
    ALL_FEATURES_ORDER,
)
from app.services.ml_service import (
    predict_applicant_suitability,
    is_mock_mode_active,
    load_ml_artifacts,
)
from app.services.gemini_service import generate_career_feedback

# --- Page Configuration ---
st.set_page_config(
    page_title="SIKUKER — AI Job Qualification & Career Feedback",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom Premium CSS Styling ---
st.markdown(
    """
    <style>
        /* Base typography & headers */
        .main-header {
            font-size: 2.2rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 0.2rem;
        }
        .sub-header {
            font-size: 1.05rem;
            color: #64748b;
            margin-bottom: 1.5rem;
        }
        
        /* Metric Badges & Status Cards */
        .status-badge-terima {
            background: linear-gradient(135deg, #10b981 0%, #059669 100%);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.3rem;
            display: inline-block;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2);
        }
        .status-badge-pertimbangan {
            background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.3rem;
            display: inline-block;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(245, 158, 11, 0.2);
        }
        .status-badge-tolak {
            background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
            color: white;
            padding: 0.6rem 1.2rem;
            border-radius: 8px;
            font-weight: 700;
            font-size: 1.3rem;
            display: inline-block;
            text-align: center;
            box-shadow: 0 4px 6px -1px rgba(239, 68, 68, 0.2);
        }
        
        /* Strengths & Weaknesses Boxes */
        .strength-box {
            background-color: #f0fdf4;
            border-left: 4px solid #22c55e;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 0.8rem;
        }
        .weakness-box {
            background-color: #fef2f2;
            border-left: 4px solid #ef4444;
            padding: 1rem 1.2rem;
            border-radius: 6px;
            margin-bottom: 0.8rem;
        }
        
        /* Career Feedback Container */
        .feedback-container {
            background-color: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 1.5rem;
            margin-top: 1rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Preset Candidate Profiles (FR-01) ---
PRESETS = {
    "kuat": {
        "applicant_name": "Ahmad Rizky Pratama, M.Kom.",
        "target_role": "Senior Data Scientist",
        "gpa": 3.85,
        "test_score": 92.0,
        "experience_years": 14.0,
        "training_hours": 85.0,
        "city_development_index": 0.92,
        "education_level": "Masters",
        "major_discipline": "STEM",
        "relevent_experience": "Has relevent experience",
        "enrolled_university": "no_enrollment",
    },
    "pertimbangan": {
        "applicant_name": "Dina Melina, S.T.",
        "target_role": "Data Scientist",
        "gpa": 3.30,
        "test_score": 68.0,
        "experience_years": 4.0,
        "training_hours": 35.0,
        "city_development_index": 0.82,
        "education_level": "Graduate",
        "major_discipline": "STEM",
        "relevent_experience": "Has relevent experience",
        "enrolled_university": "no_enrollment",
    },
    "kurang": {
        "applicant_name": "Fajar Nugraha",
        "target_role": "Junior Data Scientist",
        "gpa": 2.30,
        "test_score": 42.0,
        "experience_years": 0.5,
        "training_hours": 10.0,
        "city_development_index": 0.55,
        "education_level": "High School",
        "major_discipline": "No Major",
        "relevent_experience": "No relevent experience",
        "enrolled_university": "Full time course",
    },
}

# --- Initialize Session State with Default Values ---
default_init = PRESETS["pertimbangan"]
for field, val in default_init.items():
    if field not in st.session_state:
        st.session_state[field] = val

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None
if "career_feedback" not in st.session_state:
    st.session_state.career_feedback = None


def apply_preset(preset_key: str):
    """Callback to load preset values into session_state."""
    preset = PRESETS[preset_key]
    for key, value in preset.items():
        st.session_state[key] = value
    st.session_state.prediction_result = None
    st.session_state.career_feedback = None


# --- SIDEBAR: System Info, Status Indicators & Presets ---
with st.sidebar:
    st.markdown("### 💼 SIKUKER System")
    st.caption("Klasifikasi Kualifikasi & Umpan Balik Karir Berbasis AI")

    st.markdown("---")
    st.markdown("#### ⚙️ Status Infrastruktur")

    # Check Model Artifacts Status
    model_ready = MODEL_PATH.exists() and FEATURE_METADATA_PATH.exists()
    if model_ready and not is_mock_mode_active():
        st.success("✅ **Model ML:** Ready (Colab Random Forest)", icon="🟢")
    else:
        st.warning("⚠️ **Model ML:** Mock/Fallback Mode Aktif", icon="🟡")

    # Check Gemini API Status
    has_api_key = bool(GEMINI_API_KEY and "your_gemini" not in GEMINI_API_KEY.lower())
    if has_api_key:
        st.success(f"✅ **Gemini AI:** Online (`{GEMINI_MODEL}`)", icon="🟢")
    else:
        st.info("ℹ️ **Gemini AI:** Offline / Fallback Template", icon="🔵")

    st.markdown("---")
    st.markdown("#### ⚡ Preset Uji Simulasi Cepat")
    st.caption("Pilih skenario pelamar untuk mengisi seluruh form secara instan:")

    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("🟢 Kuat (Terima)", use_container_width=True, on_click=apply_preset, args=("kuat",)):
            pass
    with col_btn2:
        if st.button("🟡 Pertimbangan", use_container_width=True, on_click=apply_preset, args=("pertimbangan",)):
            pass

    if st.button("🔴 Kurang (Tolak)", use_container_width=True, on_click=apply_preset, args=("kurang",)):
        pass

    st.markdown("---")
    st.markdown("#### ℹ️ Aturan Ambang Batas (3-Tier)")
    st.markdown(
        f"""
        - **Terima:** Probabilitas $\\ge {int(TERIMA_THRESHOLD*100)}\\%$
        - **Pertimbangan:** ${int(PERTIMBANGAN_THRESHOLD*100)}\\% \\le P < {int(TERIMA_THRESHOLD*100)}\\%$
        - **Tolak:** Probabilitas $< {int(PERTIMBANGAN_THRESHOLD*100)}\\%$
        """
    )
    st.caption("Metodologi: Classic SDLC Waterfall")


# --- MAIN HEADER ---
st.markdown('<div class="main-header">🎯 AI Job Qualification & Career Feedback System</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">'
    'Evaluasi objektif kesesuaian kualifikasi kandidat menggunakan Machine Learning '
    'dan rekomendasi karir personal berbasis Google Gemini API.'
    '</div>',
    unsafe_allow_html=True,
)

# --- FORM INPUT KUALIFIKASI PELAMAR ---
with st.container():
    st.markdown("### 📝 Formulir Data & Kualifikasi Pelamar")
    
    # Row 1: Candidate Meta
    c_meta1, c_meta2 = st.columns(2)
    with c_meta1:
        applicant_name = st.text_input(
            "Nama Lengkap Pelamar",
            key="applicant_name",
            placeholder="Contoh: Budi Santoso, S.Kom.",
        )
    with c_meta2:
        target_role = st.text_input(
            "Posisi Kualifikasi yang Dituju",
            key="target_role",
            placeholder="Contoh: Data Scientist / ML Engineer",
        )

    st.markdown("##### 📊 Indikator Kualifikasi Utama (9 Fitur Model)")
    col1, col2, col3 = st.columns(3)

    # Column 1: Academic & Cognitive Metrics
    with col1:
        st.markdown("**Akademik & Asesmen**")
        gpa = st.number_input(
            "Indeks Prestasi Kumulatif (IPK)",
            min_value=2.00,
            max_value=4.00,
            step=0.01,
            format="%.2f",
            key="gpa",
            help="Rentang: 2.00 - 4.00 (Benchmark ideal: 3.45)",
        )
        test_score = st.number_input(
            "Skor Tes Teknis / Koding",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            key="test_score",
            help="Rentang: 0.0 - 100.0 (Benchmark ideal: 78.5)",
        )
        education_level = st.selectbox(
            "Tingkat Pendidikan Terakhir",
            options=["Graduate", "Masters", "Phd", "High School", "Primary School"],
            key="education_level",
        )

    # Column 2: Professional Experience & Industry Ecosystem
    with col2:
        st.markdown("**Pengalaman & Pelatihan**")
        experience_years = st.number_input(
            "Pengalaman Kerja (Tahun)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            format="%.1f",
            key="experience_years",
            help="Rentang: 0.0 - 30.0 tahun (Benchmark ideal: 12.0 tahun)",
        )
        training_hours = st.number_input(
            "Jam Pelatihan / Kursus Terverifikasi",
            min_value=0.0,
            max_value=500.0,
            step=5.0,
            format="%.0f",
            key="training_hours",
            help="Total durasi pelatihan dalam jam (Benchmark ideal: 50 jam)",
        )
        city_dev_index = st.number_input(
            "Indeks Perkembangan Kota (CDI)",
            min_value=0.40,
            max_value=1.00,
            step=0.01,
            format="%.2f",
            key="city_development_index",
            help="Indeks ekosistem industri kota domisili (0.40 - 1.00)",
        )

    # Column 3: Domain Relevance & Availability
    with col3:
        st.markdown("**Relevansi & Kesiapan Kerja**")
        major_discipline = st.selectbox(
            "Disiplin Ilmu / Jurusan",
            options=["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"],
            key="major_discipline",
        )
        relevent_experience = st.selectbox(
            "Relevansi Pengalaman Kerja",
            options=["Has relevent experience", "No relevent experience"],
            key="relevent_experience",
        )
        enrolled_university = st.selectbox(
            "Status Studi Kuliah Saat Ini",
            options=["no_enrollment", "Full time course", "Part time course"],
            key="enrolled_university",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    analyze_btn = st.button("🔍 Analisis Kualifikasi & Buat Umpan Balik Karir", type="primary", use_container_width=True)

# --- PROCESS PREDICTION & GEMINI FEEDBACK ON BUTTON CLICK ---
if analyze_btn:
    try:
        # Step 1: Validate input using Pydantic schema
        applicant_input = ApplicantInput(
            gpa=gpa,
            test_score=test_score,
            experience_years=experience_years,
            training_hours=training_hours,
            city_development_index=city_dev_index,
            education_level=education_level,
            major_discipline=major_discipline,
            relevent_experience=relevent_experience,
            enrolled_university=enrolled_university,
        )

        with st.spinner("Menghubungkan pipeline ML dan menjalankan inferensi kualifikasi..."):
            # Step 2: Run ML inference via service
            prediction = predict_applicant_suitability(applicant_input)
            st.session_state.prediction_result = prediction

        with st.spinner("Menyusun surat umpan balik karir terpersonalisasi via Gemini AI..."):
            # Step 3: Run Gemini AI feedback via service
            feedback = generate_career_feedback(
                applicant_name=applicant_name,
                target_role=target_role,
                applicant=applicant_input,
                prediction=prediction,
            )
            st.session_state.career_feedback = feedback

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses data: {e}")


# --- PANEL HASIL EVALUASI & FEEDBACK ---
if st.session_state.prediction_result and st.session_state.career_feedback:
    prediction: PredictionResult = st.session_state.prediction_result
    feedback: str = st.session_state.career_feedback

    st.markdown("---")
    st.markdown("## 📈 Hasil Evaluasi & Rekomendasi Seleksi")

    # --- Metrics & Status Bar ---
    col_stat1, col_stat2, col_stat3 = st.columns([1.5, 1.5, 2])
    
    with col_stat1:
        st.markdown("**Status Keputusan Akhir:**")
        if prediction.status == "Terima":
            st.markdown('<div class="status-badge-terima">✅ TERIMA (LOLOS)</div>', unsafe_allow_html=True)
        elif prediction.status == "Pertimbangan":
            st.markdown('<div class="status-badge-pertimbangan">⚠️ PERTIMBANGAN</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-badge-tolak">❌ TOLAK (BELUM SESUAI)</div>', unsafe_allow_html=True)

    with col_stat2:
        st.metric(
            label="Skor Probabilitas Kelayakan",
            value=f"{prediction.percentage:.1f}%",
            delta=f"{'+' if prediction.percentage >= 75 else ''}{prediction.percentage - 75:.1f}% vs Passing Standard",
        )

    with col_stat3:
        st.markdown("**Indikator Tingkat Kecocokan:**")
        st.progress(float(prediction.probability))
        st.caption(f"Confidence score: {prediction.probability:.4f} (Model: Random Forest Classifier)")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- Strengths vs Weaknesses Comparison ---
    col_str, col_weak = st.columns(2)

    with col_str:
        st.markdown("#### 🌟 Kekuatan Profil (Strengths)")
        if prediction.strengths:
            for s in prediction.strengths:
                st.markdown(f'<div class="strength-box">✔ {s}</div>', unsafe_allow_html=True)
        else:
            st.info("Belum ada indikator kualifikasi yang melampaui benchmark kandidat lolos.")

    with col_weak:
        st.markdown("#### ⚠️ Area Pengembangan (Qualification Gaps)")
        if prediction.weaknesses:
            for w in prediction.weaknesses:
                st.markdown(f'<div class="weakness-box">✖ {w}</div>', unsafe_allow_html=True)
        else:
            st.success("Hebat! Tidak ditemukan kesenjangan kualifikasi yang berada di bawah benchmark standar.")

    st.markdown("<br>", unsafe_allow_html=True)

    # --- GEMINI CAREER FEEDBACK & ROADMAP SECTION ---
    st.markdown("### 🤖 Draf Umpan Balik Karir & Rekomendasi Upskilling")
    st.caption("Dihasilkan secara terpersonalisasi melalui integrasi Google Gemini API.")

    with st.container():
        st.markdown(feedback)

    # --- DOWNLOAD EVALUATION REPORT BUTTON (FR-06) ---
    clean_filename_name = applicant_name.replace(" ", "_").replace(",", "").replace(".", "") if applicant_name else "Kandidat"
    report_filename = f"Laporan_SIKUKER_{clean_filename_name}_{prediction.status}_{datetime.now().strftime('%Y%m%d')}.md"

    report_content = f"""# LAPORAN EVALUASI KESESUAIAN KUALIFIKASI KERJA (SIKUKER)
**Tanggal Evaluasi:** {datetime.now().strftime('%d %B %Y, %H:%M WIB')}  
**Nama Pelamar:** {applicant_name if applicant_name else 'Kandidat'}  
**Posisi yang Dituju:** {target_role if target_role else 'Data Scientist'}  

---

## 1. Rangkuman Keputusan Seleksi
- **Status Keputusan:** **{prediction.status}**
- **Probabilitas Kesesuaian:** {prediction.probability:.4f} ({prediction.percentage:.2f}%)
- **Ambang Batas Model:** Terima (>= 75%), Pertimbangan (45% - 74%), Tolak (< 45%)

## 2. Rincian Poin Kekuatan Profil
{chr(10).join(['- ' + s for s in prediction.strengths]) if prediction.strengths else '- Tidak ada keunggulan di atas benchmark.'}

## 3. Rincian Kesenjangan Kualifikasi (Gap Analysis)
{chr(10).join(['- ' + w for w in prediction.weaknesses]) if prediction.weaknesses else '- Tidak ada kesenjangan di bawah benchmark.'}

---

## 4. Draf Umpan Balik Karir & Rekomendasi Belajar (Gemini AI)
{feedback}

---
*Dibuat secara otomatis oleh SIKUKER (Sistem Klasifikasi Kesesuaian Kualifikasi Kerja Berdasarkan ML & Gemini API).*
"""

    st.markdown("<br>", unsafe_allow_html=True)
    st.download_button(
        label="📥 Unduh Laporan Lengkap Evaluasi (.md)",
        data=report_content,
        file_name=report_filename,
        mime="text/markdown",
        use_container_width=True,
    )
