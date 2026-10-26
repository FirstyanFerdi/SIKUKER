"""
SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja & Umpan Balik Karir Berbasis AI.
Streamlit Main Application UI.
Strictly acts as a presentation layer, consuming services from app/services/ and schemas from app/schemas/.
Enhanced with modern, aesthetic, premium Glassmorphism & SaaS Design System.
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

# --- Custom Premium CSS Styling (Modern SaaS & Glassmorphism Design System) ---
st.markdown(
    """
    <style>
        /* Google Fonts Import */
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

        /* Root Typography */
        html, body, [class*="css"] {
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
            letter-spacing: -0.01em;
        }

        /* Container Border Styling - Modern SaaS Card */
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #ffffff;
            border: 1px solid #e2e8f0 !important;
            border-radius: 18px !important;
            padding: 1.2rem 1.4rem !important;
            box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.03) !important;
            transition: all 0.25s ease !important;
            margin-bottom: 1.2rem !important;
        }
        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.06) !important;
            border-color: #cbd5e1 !important;
        }

        /* Hero Header Section */
        .hero-container {
            background: linear-gradient(135deg, rgba(248, 250, 252, 0.95) 0%, rgba(238, 242, 255, 0.85) 100%);
            border: 1px solid rgba(226, 232, 240, 0.9);
            border-radius: 20px;
            padding: 2.2rem 2.5rem;
            margin-bottom: 1.8rem;
            box-shadow: 0 10px 30px -10px rgba(79, 70, 229, 0.07);
            position: relative;
            overflow: hidden;
        }
        .hero-container::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 4px;
            background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
        }
        .hero-badge {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.35rem 0.9rem;
            border-radius: 9999px;
            background: linear-gradient(135deg, rgba(59, 130, 246, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
            border: 1px solid rgba(99, 102, 241, 0.25);
            color: #4f46e5;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
            margin-bottom: 0.8rem;
        }
        .hero-title {
            font-size: 2.35rem;
            font-weight: 800;
            line-height: 1.2;
            background: linear-gradient(135deg, #0f172a 0%, #1e40af 50%, #4338ca 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.6rem;
        }
        .hero-subtitle {
            font-size: 1.05rem;
            color: #475569;
            line-height: 1.6;
            max-width: 950px;
            margin-bottom: 1.2rem;
        }
        .hero-chips {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
        }
        .chip {
            background: white;
            border: 1px solid #e2e8f0;
            border-radius: 8px;
            padding: 0.3rem 0.75rem;
            font-size: 0.82rem;
            font-weight: 600;
            color: #334155;
            display: inline-flex;
            align-items: center;
            gap: 0.4rem;
            box-shadow: 0 1px 2px rgba(0,0,0,0.03);
        }

        /* Card Section Header Tag */
        .card-header-tag {
            font-size: 1.05rem;
            font-weight: 700;
            color: #1e293b;
            margin-bottom: 1rem;
            display: flex;
            align-items: center;
            gap: 0.5rem;
            padding-bottom: 0.5rem;
            border-bottom: 2px solid #f1f5f9;
        }

        /* Status Decision Cards */
        .status-card-terima {
            background: linear-gradient(135deg, #059669 0%, #10b981 100%);
            color: white;
            padding: 1.35rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(16, 185, 129, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .status-card-pertimbangan {
            background: linear-gradient(135deg, #d97706 0%, #f59e0b 100%);
            color: white;
            padding: 1.35rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(245, 158, 11, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .status-card-tolak {
            background: linear-gradient(135deg, #dc2626 0%, #ef4444 100%);
            color: white;
            padding: 1.35rem;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 10px 25px -5px rgba(239, 68, 68, 0.35);
            border: 1px solid rgba(255, 255, 255, 0.25);
        }
        .status-card-title {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            opacity: 0.95;
            font-weight: 600;
            margin-bottom: 0.35rem;
        }
        .status-card-badge {
            font-size: 1.55rem;
            font-weight: 800;
            letter-spacing: -0.01em;
        }

        /* Strengths & Weaknesses Pills */
        .strength-pill {
            background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
            border: 1px solid #bbf7d0;
            border-left: 5px solid #16a34a;
            color: #14532d;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.5;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
        }
        .weakness-pill {
            background: linear-gradient(135deg, #fff1f2 0%, #ffe4e6 100%);
            border: 1px solid #fecdd3;
            border-left: 5px solid #e11d48;
            color: #881337;
            padding: 0.8rem 1rem;
            border-radius: 10px;
            margin-bottom: 0.6rem;
            font-size: 0.92rem;
            font-weight: 500;
            line-height: 1.5;
            box-shadow: 0 2px 4px rgba(0,0,0,0.02);
            display: flex;
            align-items: flex-start;
            gap: 0.5rem;
        }

        /* Gemini AI Career Feedback Header */
        .feedback-header-bar {
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 2px solid #f1f5f9;
            padding-bottom: 0.9rem;
            margin-bottom: 1.4rem;
        }

        /* Sidebar Styling */
        .sidebar-brand {
            background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
            color: white;
            padding: 1.2rem;
            border-radius: 14px;
            margin-bottom: 1.2rem;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        .sidebar-brand h3 {
            color: white !important;
            margin: 0;
            font-size: 1.25rem;
            font-weight: 700;
        }
        .sidebar-brand p {
            color: #94a3b8;
            font-size: 0.8rem;
            margin: 0.3rem 0 0 0;
        }
        .infra-badge {
            background: #ffffff;
            border: 1px solid #e2e8f0;
            border-radius: 10px;
            padding: 0.55rem 0.85rem;
            margin-bottom: 0.45rem;
            font-size: 0.82rem;
            font-weight: 600;
            display: flex;
            align-items: center;
            justify-content: space-between;
            box-shadow: 0 1px 2px rgba(0,0,0,0.02);
        }
        .threshold-box {
            background: #f8fafc;
            border-left: 3px solid #6366f1;
            padding: 0.75rem 0.9rem;
            border-radius: 6px;
            font-size: 0.82rem;
            line-height: 1.5;
            color: #334155;
            margin-top: 0.8rem;
        }

        /* Streamlit Primary Button Enhancement */
        div.stButton > button[kind="primary"] {
            background: linear-gradient(135deg, #2563eb 0%, #4f46e5 100%) !important;
            color: white !important;
            border: none !important;
            border-radius: 14px !important;
            padding: 0.85rem 2.2rem !important;
            font-size: 1.05rem !important;
            font-weight: 700 !important;
            box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.45) !important;
            transition: all 0.25s ease !important;
        }
        div.stButton > button[kind="primary"]:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 16px 32px -5px rgba(79, 70, 229, 0.6) !important;
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


# --- SIDEBAR: Brand, Infrastructure & Presets ---
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <h3>💼 SIKUKER Engine</h3>
            <p>Sistem Cerdas Klasifikasi Kualifikasi & Umpan Balik Karir Generative AI</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("##### ⚙️ Status Infrastruktur")

    # Check Model Artifacts Status
    model_ready = MODEL_PATH.exists() and FEATURE_METADATA_PATH.exists()
    if model_ready and not is_mock_mode_active():
        st.markdown(
            """
            <div class="infra-badge">
                <span>🤖 <b>ML Classifier</b></span>
                <span style="color: #059669; font-weight: 700;">● Ready (Colab RF)</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="infra-badge">
                <span>🤖 <b>ML Classifier</b></span>
                <span style="color: #d97706; font-weight: 700;">▲ Fallback Mock</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Check Gemini API Status
    has_api_key = bool(GEMINI_API_KEY and "your_gemini" not in GEMINI_API_KEY.lower())
    if has_api_key:
        st.markdown(
            f"""
            <div class="infra-badge">
                <span>✨ <b>Gemini Coach</b></span>
                <span style="color: #059669; font-weight: 700;">● Online ({GEMINI_MODEL})</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="infra-badge">
                <span>✨ <b>Gemini Coach</b></span>
                <span style="color: #3b82f6; font-weight: 700;">ℹ Local Fallback</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("##### ⚡ Preset Simulasi Cepat")
    st.caption("Pilih skenario profil untuk demonstrasi instan:")

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
    st.markdown("##### ℹ️ Aturan Ambang Batas (3-Tier)")
    st.markdown(
        f"""
        <div class="threshold-box">
            <b>Skema Penilaian (BR-01):</b><br>
            • <b>Terima:</b> Probabilitas ≥ {int(TERIMA_THRESHOLD*100)}%<br>
            • <b>Pertimbangan:</b> {int(PERTIMBANGAN_THRESHOLD*100)}% ≤ P &lt; {int(TERIMA_THRESHOLD*100)}%<br>
            • <b>Tolak:</b> Probabilitas &lt; {int(PERTIMBANGAN_THRESHOLD*100)}%<br>
            <hr style="margin: 0.5rem 0; border: none; border-top: 1px dashed #cbd5e1;">
            <small style="color: #64748b;">Metodologi: Phased SDLC Waterfall</small>
        </div>
        """,
        unsafe_allow_html=True,
    )


# --- HERO HEADER SECTION ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-badge">✨ SISTEM CERDAS 2026 • AI-POWERED HR ANALYTICS</div>
        <div class="hero-title">SIKUKER: AI Qualification & Career Intelligence</div>
        <div class="hero-subtitle">
            Platform evaluasi kualifikasi pelamar kerja berbasis <b>Machine Learning (Random Forest)</b> 
            dengan integrasi <b>Google Gemini API</b> untuk menyusun surat evaluasi karir transparan 
            dan rencana pengembangan diri (<i>upskilling roadmap</i>) yang personal dan aplikatif.
        </div>
        <div class="hero-chips">
            <span class="chip">🎯 3-Tier Objective Decision</span>
            <span class="chip">🔍 Qualification Gap Analyzer</span>
            <span class="chip">🤖 Google Gemini 3.1 Flash-Lite</span>
            <span class="chip">⚡ Zero-Crash Resilience</span>
            <span class="chip">📊 9 Standardized Features</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# --- MAIN INPUT FORM ---
# Card 1: Candidate Metadata
with st.container(border=True):
    st.markdown('<div class="card-header-tag">👤 Informasi Profil & Posisi yang Dilamar</div>', unsafe_allow_html=True)
    
    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        applicant_name = st.text_input(
            "Nama Lengkap Pelamar",
            key="applicant_name",
            placeholder="Contoh: Budi Pratama, S.Kom.",
            help="Masukkan nama lengkap kandidat untuk personalisasi surat umpan balik.",
        )
    with meta_col2:
        target_role = st.text_input(
            "Posisi Kualifikasi yang Dituju",
            key="target_role",
            placeholder="Contoh: Data Scientist / Machine Learning Engineer",
            help="Posisi pekerjaan target yang dijadikan acuan tolok ukur (benchmark).",
        )

# Card 2: 9 Feature Inputs in 3 Functional Columns
with st.container(border=True):
    st.markdown('<div class="card-header-tag">📊 Indikator Kualifikasi Utama (9 Fitur Model Terstandarisasi)</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    # Column 1: Academic & Cognitive Metrics
    with col1:
        st.markdown("<p style='font-weight: 700; color: #4338ca; margin-bottom: 0.5rem;'>🎓 Akademik & Asesmen</p>", unsafe_allow_html=True)
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
            "Skor Ujian Teknis / Koding",
            min_value=0.0,
            max_value=100.0,
            step=1.0,
            format="%.1f",
            key="test_score",
            help="Rentang: 0.0 - 100.0 (Benchmark ideal kelulusan: 78.5)",
        )
        education_level = st.selectbox(
            "Tingkat Pendidikan Terakhir",
            options=["Graduate", "Masters", "Phd", "High School", "Primary School"],
            key="education_level",
            help="Jenjang pendidikan formal terakhir kandidat.",
        )

    # Column 2: Professional Experience & Industry Ecosystem
    with col2:
        st.markdown("<p style='font-weight: 700; color: #4338ca; margin-bottom: 0.5rem;'>💼 Pengalaman & Pelatihan</p>", unsafe_allow_html=True)
        experience_years = st.number_input(
            "Total Pengalaman Kerja (Tahun)",
            min_value=0.0,
            max_value=30.0,
            step=0.5,
            format="%.1f",
            key="experience_years",
            help="Durasi total pengalaman profesional (Benchmark ideal: 12.0 tahun)",
        )
        training_hours = st.number_input(
            "Jam Kursus / Pelatihan Terverifikasi",
            min_value=0.0,
            max_value=500.0,
            step=5.0,
            format="%.0f",
            key="training_hours",
            help="Total jam pelatihan intensif yang telah diselesaikan (Benchmark ideal: 50 jam)",
        )
        city_dev_index = st.number_input(
            "City Development Index (CDI)",
            min_value=0.40,
            max_value=1.00,
            step=0.01,
            format="%.2f",
            key="city_development_index",
            help="Indeks kemajuan ekosistem teknologi kota domisili (Benchmark ideal: 0.91)",
        )

    # Column 3: Domain Relevance & Availability
    with col3:
        st.markdown("<p style='font-weight: 700; color: #4338ca; margin-bottom: 0.5rem;'>🎯 Relevansi & Kesiapan</p>", unsafe_allow_html=True)
        major_discipline = st.selectbox(
            "Disiplin Ilmu / Jurusan Kuliah",
            options=["STEM", "Business Degree", "Arts", "Humanities", "No Major", "Other"],
            key="major_discipline",
            help="Latar belakang studi (STEM/Business diprioritaskan untuk peran data).",
        )
        relevent_experience = st.selectbox(
            "Relevansi Pengalaman Kerja",
            options=["Has relevent experience", "No relevent experience"],
            key="relevent_experience",
            help="Kesesuaian proyek kerja sebelumnya dengan ranah teknologi.",
        )
        enrolled_university = st.selectbox(
            "Status Perkuliahan Saat Ini",
            options=["no_enrollment", "Full time course", "Part time course"],
            key="enrolled_university",
            help="Komitmen waktu studi (no_enrollment siap kerja penuh waktu).",
        )

# Big Prominent Action Button
analyze_btn = st.button(
    "🚀 Jalankan Evaluasi Kualifikasi & Umpan Balik AI",
    type="primary",
    use_container_width=True,
)


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

        with st.spinner("⚡ Menghubungkan pipeline Machine Learning dan menghitung probabilitas kelayakan..."):
            # Step 2: Run ML inference via service
            prediction = predict_applicant_suitability(applicant_input)
            st.session_state.prediction_result = prediction

        with st.spinner("🤖 Menyusun surat evaluasi karir & peta jalan upskilling via Google Gemini AI..."):
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 📈 Hasil Evaluasi Kualifikasi & Rekomendasi Seleksi")

    # --- Top Metrics & Decision Dashboard ---
    with st.container(border=True):
        m_col1, m_col2, m_col3 = st.columns([1.6, 1.4, 2])
        
        with m_col1:
            if prediction.status == "Terima":
                st.markdown(
                    """
                    <div class="status-card-terima">
                        <div class="status-card-title">Status Keputusan Akhir</div>
                        <div class="status-card-badge">✅ TERIMA (LOLOS)</div>
                        <div style="font-size: 0.82rem; margin-top: 0.35rem; opacity: 0.95;">Memenuhi Seluruh Standar Kualifikasi</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            elif prediction.status == "Pertimbangan":
                st.markdown(
                    """
                    <div class="status-card-pertimbangan">
                        <div class="status-card-title">Status Keputusan Akhir</div>
                        <div class="status-card-badge">⚠️ PERTIMBANGAN</div>
                        <div style="font-size: 0.82rem; margin-top: 0.35rem; opacity: 0.95;">Memerlukan Review Portofolio Lanjutan</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    """
                    <div class="status-card-tolak">
                        <div class="status-card-title">Status Keputusan Akhir</div>
                        <div class="status-card-badge">❌ TOLAK (BELUM SESUAI)</div>
                        <div style="font-size: 0.82rem; margin-top: 0.35rem; opacity: 0.95;">Disarankan Mengikuti Program Upskilling</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        with m_col2:
            st.metric(
                label="Skor Probabilitas Kelayakan",
                value=f"{prediction.percentage:.1f}%",
                delta=f"{'+' if prediction.percentage >= 75 else ''}{prediction.percentage - 75:.1f}% vs Passing Standard (75%)",
            )

        with m_col3:
            st.markdown(f"**Tingkat Kesiapan Kerja: `{prediction.status}`**")
            st.progress(float(prediction.probability))
            st.caption(f"Confidence Score: **{prediction.probability:.4f}** • Model: *Random Forest (200 Trees)*")

    # --- Strengths vs Weaknesses Comparison Cards ---
    col_str, col_weak = st.columns(2)

    with col_str:
        with st.container(border=True):
            st.markdown('<div class="card-header-tag" style="color: #15803d;">🌟 Kekuatan Profil (Strengths)</div>', unsafe_allow_html=True)
            if prediction.strengths:
                for s in prediction.strengths:
                    st.markdown(f'<div class="strength-pill"><span>✔</span> <span>{s}</span></div>', unsafe_allow_html=True)
            else:
                st.info("Belum ada indikator kualifikasi yang melampaui benchmark kandidat ideal.")

    with col_weak:
        with st.container(border=True):
            st.markdown('<div class="card-header-tag" style="color: #b91c1c;">⚠️ Area Pengembangan Kualifikasi (Gap Analysis)</div>', unsafe_allow_html=True)
            if prediction.weaknesses:
                for w in prediction.weaknesses:
                    st.markdown(f'<div class="weakness-pill"><span>✖</span> <span>{w}</span></div>', unsafe_allow_html=True)
            else:
                st.success("Luar biasa! Tidak ditemukan kesenjangan kualifikasi terhadap benchmark standar.")

    # --- GEMINI CAREER FEEDBACK & ROADMAP SECTION ---
    with st.container(border=True):
        st.markdown(
            """
            <div class="feedback-header-bar">
                <div>
                    <h3 style="margin: 0; color: #1e293b; font-weight: 700; font-size: 1.3rem;">🤖 Surat Evaluasi Karir & Peta Jalan Upskilling</h3>
                    <p style="margin: 0.3rem 0 0 0; color: #64748b; font-size: 0.88rem;">
                        Disusun secara terpersonalisasi oleh <b>Google Gemini AI</b> atas profil kandidat 
                        berdasarkan hasil prediksi Machine Learning dan aturan Qualification Gap Analyzer.
                    </p>
                </div>
                <div style="background: #eef2ff; color: #4f46e5; padding: 0.45rem 0.9rem; border-radius: 8px; font-weight: 700; font-size: 0.8rem; border: 1px solid #c7d2fe;">
                    Official HR Assessment
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
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
