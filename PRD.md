# Product Requirements Document (PRD)

**Nama Produk:** SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja & Generator Umpan Balik Karir Berbasis AI  
**Metodologi Pengembangan:** SDLC Waterfall (Sequential Phase Gate)  
**Arsitektur Sistem:** Decoupled Pipeline (Google Colab untuk ML Training & Antigravity/Streamlit untuk Web App + Gemini API)  
**Status Dokumen:** Approved / Baseline Fase 1 SDLC  

---

## 1. Latar Belakang & Rumusan Masalah (Problem Statement)
Proses seleksi awal (*screening*) pelamar kerja sering kali menghasilkan keputusan biner (diterima atau ditolak) tanpa transparansi mengenai bagian kualifikasi mana yang belum memenuhi standar. Di sisi lain, tim HR tidak memiliki waktu yang cukup untuk menulis surat umpan balik (*feedback*) yang personal dan konstruktif bagi ratusan kandidat yang belum lolos.

**Rumusan Masalah Utama:**
> *"Bagaimana merancang model klasifikasi Machine Learning berbasis data CSV di Google Colab untuk memprediksi tingkat kelayakan calon pelamar kerja berdasarkan data kualifikasi, serta bagaimana mengintegrasikan Gemini API pada aplikasi web untuk mengolah hasil klasifikasi tersebut menjadi saran pengembangan keterampilan (upskilling) yang konstruktif dan terpersonalisasi?"*

---

## 2. Tujuan Produk (Product Objectives)
1. **Otomatisasi Klasifikasi Kualifikasi (3-Tier Decision):** Memprediksi probabilitas kesesuaian kandidat menggunakan algoritma *Random Forest* / *XGBoost* dan mengelompokkannya ke dalam 3 status keputusan: **Terima**, **Pertimbangan**, dan **Tolak**.
2. **Transparansi Evaluasi (*Qualification Gap Analysis*):** Mengidentifikasi atribut kualifikasi kandidat yang sudah unggul (*strengths*) dan yang masih berada di bawah standar kandidat lolos (*weaknesses/gaps*).
3. **Personalisasi Umpan Balik Karir (Gemini API):** Menghasilkan draf surat evaluasi karir yang sopan, empatik, dan profesional dalam Bahasa Indonesia, lengkap dengan rekomendasi materi belajar (*upskilling roadmap*) yang menjawab kelemahan spesifik kandidat.

---

## 3. Target Pengguna & Persona (User Personas)
1. **Recruiter / Tim HR (Primary User):**
   - **Kebutuhan:** Memasukkan data kualifikasi pelamar dengan cepat, melihat skor kecocokan objektif, dan mendapatkan draf surat umpan balik siap kirim tanpa harus mengetik dari nol.
2. **Calon Pelamar Kerja / Job Seeker (Secondary/Beneficiary User):**
   - **Kebutuhan:** Melakukan simulasi mandiri (*self-assessment*) atas kualifikasi mereka saat ini dan mengetahui kursus/keterampilan spesifik apa yang harus dipelajari agar lolos seleksi di masa mendatang.

---

## 4. Spesifikasi Data & Skema Fitur (Input Schema)
Sistem menggunakan referensi skema dari *Kaggle HR Analytics: Job Change of Data Scientists* yang disesuaikan dengan atribut seleksi kualifikasi kandidat:

### A. Fitur Numerik (`NUMERIC_FEATURES`)
| Nama Fitur | Label UI (Bahasa Indonesia) | Rentang Nilai Valid | Deskripsi |
| :--- | :--- | :--- | :--- |
| `gpa` | Indeks Prestasi Kumulatif (IPK) | `2.00` – `4.00` | Nilai akademik akhir kandidat pada skala 4.00. |
| `test_score` | Skor Tes Teknis | `0.0` – `100.0` | Nilai ujian kemampuan teknis/koding kandidat. |
| `experience_years` | Pengalaman Kerja (Tahun) | `0.0` – `30.0` | Total durasi pengalaman kerja profesional. |
| `training_hours` | Jam Pelatihan / Kursus | `0.0` – `500.0` | Total jam pelatihan intensif atau sertifikasi yang diikuti. |
| `city_development_index` | Indeks Perkembangan Kota | `0.40` – `1.00` | Indeks ekosistem industri/teknologi domisili kandidat. |

### B. Fitur Kategorikal (`CATEGORICAL_FEATURES`)
| Nama Fitur | Label UI (Bahasa Indonesia) | Nilai Kategori Valid |
| :--- | :--- | :--- |
| `education_level` | Tingkat Pendidikan Terakhir | `Graduate`, `Masters`, `Phd`, `High School`, `Primary School` |
| `major_discipline` | Jurusan / Disiplin Ilmu | `STEM`, `Business Degree`, `Arts`, `Humanities`, `No Major`, `Other` |
| `relevent_experience` | Relevansi Pengalaman Kerja | `Has relevent experience`, `No relevent experience` |
| `enrolled_university` | Status Studi Saat Ini | `no_enrollment`, `Full time course`, `Part time course` |

---

## 5. Aturan Bisnis & Logika Sistem (Business Rules)

### BR-01: Pemetaan Ambang Batas Status Prediksi (3-Tier Mapping)
Model Machine Learning menghasilkan probabilitas kelas positif ($P \in [0.0, 1.0]$). Aplikasi memetakan nilai $P$ menggunakan aturan berikut:
- **Status "Terima" (Hijau):** $P \ge 0.75$ (Kandidat memenuhi kualifikasi utama dan siap lanjut ke tahap wawancara akhir/offering).
- **Status "Pertimbangan" (Kuning/Oranye):** $0.45 \le P < 0.75$ (Kandidat memiliki potensi dasar namun membutuhkan peninjauan tambahan atau tes lanjutan).
- **Status "Tolak" (Merah):** $P < 0.45$ (Kualifikasi kandidat saat ini belum sesuai dengan kebutuhan minimum posisi).

### BR-02: Aturan Deteksi Kelemahan (*Gap Analyzer*)
Sistem membandingkan setiap input kandidat terhadap nilai median kandidat lolos (`ideal_numeric_benchmarks` di `models/feature_metadata.json`) serta aturan kualifikasi kategorikal:
- Jika fitur numerik kandidat berada di bawah nilai *benchmark*, fitur tersebut dimasukkan ke daftar **`weaknesses`** beserta selisih angkanya.
- Jika fitur numerik kandidat sama atau di atas nilai *benchmark*, fitur tersebut dimasukkan ke daftar **`strengths`**.
- Jika `relevent_experience == "No relevent experience"`, sistem otomatis mencatat kurangnya pengalaman relevan sebagai kelemahan utama.

### BR-03: Kontrak Struktur Output Gemini API
Setiap respons yang dihasilkan oleh Gemini API (`gemini-2.5-flash`) wajib menggunakan Bahasa Indonesia yang santun dan terdiri dari 3 bagian terstruktur:
1. **Ringkasan Evaluasi & Surat Umpan Balik Karir:** Draf komunikasi yang disesuaikan dengan status (`Terima` / `Pertimbangan` / `Tolak`).
2. **Analisis Kesenjangan Kualifikasi (*Gap Analysis*):** Penjelasan objektif dan suportif mengenai poin kelemahan yang ditemukan oleh *Gap Analyzer*.
3. **Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (*Upskilling*):** Daftar 3–5 rekomendasi kursus, topik teknis, proyek portofolio, atau sertifikasi yang spesifik untuk menutup kelemahan kandidat.

---

## 6. Kebutuhan Fungsional & Kriteria Penerimaan (Acceptance Criteria)

| ID | Fitur | User Story | Acceptance Criteria (Kriteria Lulus Uji) |
| :--- | :--- | :--- | :--- |
| **FR-01** | Preset Simulasi Cepat | Sebagai pengguna, saya ingin menekan tombol contoh profil kandidat agar dapat mendemokan 3 status prediksi dengan cepat. | Tersedia 3 tombol di sidebar (*Kandidat Kuat*, *Kandidat Pertimbangan*, *Kandidat Kurang*) yang otomatis mengisi seluruh field form. |
| **FR-02** | Validasi Form Input | Sebagai sistem, saya harus memvalidasi tipe dan rentang nilai input sebelum memproses prediksi. | Input di luar rentang (misal IPK > 4.0) dicegah oleh komponen UI dan divalidasi oleh schema `Pydantic`. |
| **FR-03** | Inferensi ML Lokal | Sebagai pengguna, saya ingin melihat skor probabilitas dan status kelayakan seketika setelah menekan tombol analisis. | Sistem memuat `.joblib` dari `models/`, menampilkan persentase probabilitas (0–100%) dan badge status (`Terima`/`Pertimbangan`/`Tolak`) dalam waktu kurang dari 1 detik. |
| **FR-04** | Tampilan Kekuatan & Kelemahan | Sebagai pengguna, saya ingin melihat rincian kualifikasi mana yang sudah baik dan mana yang masih kurang. | UI menampilkan dua kolom terpisah berisi poin *Strengths* dan *Weaknesses* hasil olahan `gap_analyzer.py`. |
| **FR-05** | Generasi Saran Karir Gemini | Sebagai pengguna, saya ingin mendapatkan draf umpan balik karir dan rekomendasi belajar otomatis dari Gemini API. | Sistem memanggil `gemini-2.5-flash` via SDK `google-genai` dan merender hasil Markdown 3 bagian. Jika API gagal/tanpa key, sistem menampilkan *Fallback Template* tanpa crash. |
| **FR-06** | Unduh Laporan Evaluasi | Sebagai pengguna, saya ingin mengunduh hasil evaluasi dan saran karir dalam bentuk file. | Tersedia tombol `Download` yang mengekspor laporan lengkap kandidat ke format `.md` atau `.txt`. |

---

## 7. Batasan Ruang Lingkup (Out of Scope)
Agar proyek tetap fokus sesuai jadwal SDLC Waterfall:
- Aplikasi web **tidak** melakukan pelatihan ulang (*re-training*) model ML secara langsung dari UI web.
- Sistem **tidak** mencakup manajemen autentikasi multi-user (login/register) atau penyimpanan database relasional (PostgreSQL/MySQL) pada rilis v1.0.
- Pemrosesan dokumen CV berbentuk PDF/Word (OCR) berada di luar cakupan; input dilakukan melalui formulir terstruktur.