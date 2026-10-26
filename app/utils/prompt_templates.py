"""
Prompt templates and prompt builders for Gemini AI career feedback generation.
Formats applicant profile and ML prediction outcomes into structured prompts.
"""

from app.schemas.applicant_schema import ApplicantInput, PredictionResult


SYSTEM_INSTRUCTION: str = (
    "Anda adalah seorang Konsultan Karir HR Senior & Technical Recruiter berpengalaman di industri teknologi.\n"
    "Tugas utama Anda adalah menyusun umpan balik karir yang objektif, mendalam, konstruktif, dan memberdayakan "
    "berdasarkan data kualifikasi kandidat serta hasil prediksi model Machine Learning.\n"
    "Gunakan Bahasa Indonesia formal yang santun, empatik, dan profesional. "
    "Fokuslah pada solusi nyata dan langkah aksionabel yang dapat segera diambil oleh pelamar "
    "untuk meningkatkan daya saing profesional mereka."
)


def build_career_feedback_prompt(
    applicant_name: str,
    target_role: str,
    applicant: ApplicantInput,
    prediction: PredictionResult,
) -> str:
    """
    Construct a structured prompt for Google Gemini API.

    Args:
        applicant_name (str): Name of the applicant.
        target_role (str): Desired job role (e.g., 'Data Scientist / Machine Learning Engineer').
        applicant (ApplicantInput): Validated applicant input profile.
        prediction (PredictionResult): ML prediction outcome and gap analysis lists.

    Returns:
        str: Fully formatted prompt ready for LLM generation.
    """
    clean_name = applicant_name.strip() if applicant_name.strip() else "Kandidat"
    clean_role = target_role.strip() if target_role.strip() else "Data Scientist / Machine Learning Engineer"

    strengths_text = (
        "\n".join([f"- {s}" for s in prediction.strengths])
        if prediction.strengths
        else "- Belum ada poin keunggulan signifikan yang melampaui benchmark kandidat lolos."
    )

    weaknesses_text = (
        "\n".join([f"- {w}" for w in prediction.weaknesses])
        if prediction.weaknesses
        else "- Tidak ditemukan kesenjangan kualifikasi mayor terhadap benchmark standar."
    )

    status_guidance = {
        "Terima": (
            "Status 'Terima' (Probabilitas tinggi): Berikan ucapan selamat yang profesional dan apresiatif. "
            "Tegaskan bahwa kualifikasi kandidat sangat kuat dan siap menghadapi tahapan seleksi teknis lanjutan atau offering. "
            "Fokus rekomendasi adalah penguatan kepemimpinan teknis dan proyek berskala besar."
        ),
        "Pertimbangan": (
            "Status 'Pertimbangan' (Probabilitas moderat): Gunakan nada berimbang dan objektif. "
            "Apresiasi potensi dasar kandidat, namun sampaikan dengan jujur area kualifikasi yang masih meragukan atau perlu "
            "dibuktikan lebih lanjut (misalnya jam pelatihan tambahan, sertifikasi, atau portofolio proyek terapan)."
        ),
        "Tolak": (
            "Status 'Tolak' (Probabilitas rendah saat ini): Gunakan nada yang sangat empatik, suportif, dan membesarkan hati. "
            "Tegaskan bahwa hasil ini bukan penolakan atas nilai pribadi kandidat, melainkan cerminan kesiapan kualifikasi saat ini "
            "terhadap standar ketat posisi tersebut. Berikan dorongan kuat dan peta jalan realistis untuk belajar kembali."
        ),
    }.get(prediction.status, "")

    prompt = f"""Berikut adalah data evaluasi kualifikasi pelamar kerja untuk posisi **{clean_role}**:

### 1. Data Diri & Profil Kualifikasi Kandidat:
- **Nama Kandidat:** {clean_name}
- **Posisi yang Dituju:** {clean_role}
- **IPK Terakhir:** {applicant.gpa:.2f} (Skala 4.00)
- **Skor Tes Teknis:** {applicant.test_score:.1f} / 100.0
- **Pengalaman Kerja Profesional:** {applicant.experience_years:.1f} tahun
- **Jam Pelatihan / Kursus Intensif:** {applicant.training_hours:.0f} jam
- **Indeks Perkembangan Kota Domisili:** {applicant.city_development_index:.2f}
- **Tingkat Pendidikan Terakhir:** {applicant.education_level}
- **Jurusan / Disiplin Ilmu:** {applicant.major_discipline}
- **Relevansi Pengalaman Kerja:** {applicant.relevent_experience}
- **Status Perkuliahan Saat Ini:** {applicant.enrolled_university}

### 2. Hasil Keputusan Machine Learning:
- **Status Rekomendasi:** **{prediction.status}**
- **Skor Probabilitas Kesesuaian:** {prediction.probability:.4f} ({prediction.percentage:.2f}%)
- **Panduan Nada Komunikasi:** {status_guidance}

### 3. Temuan Qualification Gap Analysis:
**Poin Kekuatan / Keunggulan:**
{strengths_text}

**Poin Kesenjangan / Kelemahan Kualifikasi:**
{weaknesses_text}

---

### INSTRUKSI TUGAS & STRUKTUR OUTPUT:
Susunlah draf umpan balik karir dalam format Markdown yang rapi, profesional, dan terstruktur persis ke dalam 3 bagian wajib berikut:

### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir
Tuliskan draf surat evaluasi resmi dari tim HR/Recruiter kepada {clean_name}. Sesuaikan nada bicara dengan status ({prediction.status}). Draf harus terasa personal, tidak kaku, dan menyampaikan hasil secara transparan namun tetap memotivasi.

### 2. Analisis Kesenjangan Kualifikasi (Gap Analysis)
Jelaskan secara mendalam dan konstruktif poin-poin kelemahan yang terdeteksi di atas. Jelaskan mengapa aspek-aspek tersebut (seperti skor tes, durasi pengalaman, jam pelatihan, atau latar belakang pendidikan) menjadi pertimbangan penting bagi recruiter untuk posisi {clean_role}.

### 3. Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (Upskilling)
Berikan 3 hingga 5 rekomendasi konkret yang dapat langsung dijalankan oleh {clean_name} dalam 3 hingga 6 bulan ke depan, meliputi:
- Rekomendasi topik teknis mendalam atau kursus spesifik (misalnya di Coursera, Kaggle, Udemy, DeepLearning.AI, atau sertifikasi industri resmi).
- Ide proyek portofolio praktis berbasis data nyata yang relevan untuk diunggah ke GitHub.
- Strategi peningkatan kualifikasi untuk menutup selisih gap yang ditemukan.

Pastikan tidak menyertakan placeholder seperti '[Nama Recruiter]' yang kosong, melainkan tanda tangani sebagai 'Tim Rekrutmen & Pengembangan Bakat SIKUKER'.
"""
    return prompt
