# Software Requirements Specification (SRS)
# SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja Berdasarkan Machine Learning dengan Pemanfaatan Gemini API untuk Penyempurnaan Umpan Balik Karir

- **Dokumen Identifikasi:** SRS-SIKUKER-01
- **Standar Format:** IEEE Std 830-1998 (Adapted)
- **Fase SDLC:** Fase 1 — Requirements Engineering
- **Status Dokumen:** Baseline / Approved
- **Versi Dokumen:** 2.0.0 (Comprehensive Expansion)
- **Terakhir Diperbarui:** 26 Oktober 2026

---

## 1. Pendahuluan (Introduction)

### 1.1 Tujuan Dokumen (Purpose)
Dokumen Spesifikasi Kebutuhan Perangkat Lunak (*Software Requirements Specification* - SRS) ini mendefinisikan secara lengkap, formal, dan terperinci seluruh kebutuhan fungsional, non-fungsional, batasan sistem, dan aturan bisnis untuk perangkat lunak **SIKUKER** (*Sistem Klasifikasi Kesesuaian Kualifikasi Kerja & Umpan Balik Karir Berbasis AI*). Dokumen ini berfungsi sebagai acuan tunggal (*single source of truth*) bagi pengembang perangkat lunak, data scientist, penguji kualitas (*QA tester*), dan pemangku kepentingan akademik.

### 1.2 Ruang Lingkup Produk (Product Scope)
SIKUKER adalah sistem cerdas berbasis web yang memadukan model klasifikasi *Machine Learning* teruji (*Random Forest Classifier*) dan *Large Language Model* (*Google Gemini API - gemini-2.5-flash*) untuk memfasilitasi seleksi kualifikasi pelamar kerja bidang teknologi secara transparan dan berdaya guna.

Cakupan sistem meliputi:
1. Validasi terstandar 9 atribut kualifikasi pelamar (5 numerik dan 4 kategorikal).
2. Inferensi probabilitas kelayakan kerja secara lokal tanpa latensi pelatihan ulang (*inference-only web runtime*).
3. Pengelompokan status keputusan kelayakan ke dalam 3 tier: **Terima**, **Pertimbangan**, dan **Tolak**.
4. Pembedahan komparatif atribut kualifikasi kandidat terhadap nilai tolok ukur (*ideal benchmarks*) kandidat lolos (*Qualification Gap Analysis*).
5. Pembangkitan draf surat umpan balik karir resmi dan peta jalan pengembangan diri (*upskilling roadmap*) dalam Bahasa Indonesia santun.
6. Penyediaan antarmuka web interaktif (Streamlit) dengan tombol preset demonstrasi instan serta kemampuan unduh berkas laporan hasil evaluasi.

### 1.3 Definisi, Akronim, dan Singkatan
| Istilah / Akronim | Penjelasan Lengkap |
| :--- | :--- |
| **SIKUKER** | Sistem Klasifikasi Kesesuaian Kualifikasi Kerja. |
| **ML** | *Machine Learning* (Pembelajaran Mesin). |
| **LLM** | *Large Language Model* (Model Bahasa Skala Besar seperti Google Gemini). |
| **CDI** | *City Development Index* (Indeks Kemajuan Ekosistem Industri/Teknologi Kota Domisili). |
| **GPA / IPK** | *Grade Point Average* / Indeks Prestasi Kumulatif pada skala 4.00. |
| **STEM** | *Science, Technology, Engineering, and Mathematics*. |
| **SoC** | *Separation of Concerns* (Pemisahan Tanggung Jawab Antar-Lapisan Perangkat Lunak). |
| **Decoupled Pipeline**| Strategi pemisahan lingkungan training (Google Colab) dari runtime web (Local App). |
| **RTM** | *Requirements Traceability Matrix* (Matriks Keterlacakan Kebutuhan). |

### 1.4 Referensi (References)
1. IEEE Std 830-1998, *IEEE Recommended Practice for Software Requirements Specifications*.
2. Kaggle HR Analytics: Job Change of Data Scientists Dataset (`arashnic/hr-analytics-job-change-of-data-scientists`).
3. Google GenAI SDK Official Python Documentation (`google-genai`).
4. PRD SIKUKER v1.0 & AGENT.md System Guidelines.

---

## 2. Deskripsi Umum Sistem (Overall Description)

### 2.1 Perspektif Produk & Konteks Operasional
SIKUKER beroperasi dalam paradigma **Decoupled Pipeline**:
- **Lingkungan Pelatihan (*Training Environment - Google Colab*):** Bertanggung jawab atas eksplorasi data (*EDA*), prapemrosesan data, pemisahan data latih/uji, pemilihan model, rekayasa fitur, dan serialisasi artefak model ke format binary joblib.
- **Lingkungan Aplikasi (*Application Runtime - Antigravity/Streamlit*):** Hanya bertindak sebagai *consumer* dari artefak serialisasi yang telah diekspor. Runtime web dilarang keras melatih ulang (*fitting*) model demi menjaga efisiensi memori dan latensi respon sub-detik.

### 2.2 Persona Pengguna (User Personas)

#### Persona 1: Recruiter / HR Specialist (Primary User)
- **Profil:** Praktisi rekrutmen talenta teknologi di perusahaan berkembang atau konsultan headhunter.
- **Tujuan:** Melakukan penilaian awal (*screening*) terhadap puluhan kandidat dengan kriteria objektif, mengetahui kelemahan spesifik pelamar tanpa kalkulasi manual, dan mendapatkan draf surat umpan balik yang siap kirim.
- **Pain Points:** Menghabiskan 15–30 menit per kandidat hanya untuk menulis draf penolakan yang sopan, sering kali berujung pada pengiriman *cold rejection template* tanpa panduan perbaikan.

#### Persona 2: Calon Pelamar Kerja / Data Science Aspirant (Secondary User)
- **Profil:** Fresh graduate atau professional switcher yang ingin melamar posisi *Data Scientist* atau *Machine Learning Engineer*.
- **Tujuan:** Melakukan simulasi kelayakan kualifikasi secara mandiri (*self-assessment*) dan memperoleh rekomendasi topik belajar/sertifikasi yang relevan untuk memperkuat CV mereka.

### 2.3 Lingkungan Operasi & Ketergantungan
- **Runtime Environment:** Python 3.10+ (Teruji resmi di Python 3.13.14 x64).
- **Web Server:** Streamlit Runtime Engine (Port 8501).
- **Hardware Minimum:** CPU 2 Cores, RAM 4 GB, Ruang Disk Kosong 500 MB.
- **Konektivitas:** Akses internet untuk menghubungi endpoint Gemini API (`https://generativelanguage.googleapis.com`). Jika koneksi terputus, sistem beroperasi secara otonom menggunakan *Deterministic Offline Fallback Engine*.

---

## 3. Spesifikasi Kebutuhan Fungsional (Functional Requirements)

### FR-01: Preset Simulasi Kandidat Cepat
- **Deskripsi:** Sistem menyediakan tombol pintas pada sidebar untuk mengisi seluruh formulir profil pelamar secara otomatis.
- **Aktor:** Pengguna (Recruiter / Pelamar).
- **Preset yang Tersedia:**
  1. *🟢 Kuat (Terima):* Profil unggulan (IPK 3.85, Tes 92, Pengalaman 14 th, S2 STEM, CDI 0.92, status kerja relevan).
  2. *🟡 Pertimbangan:* Profil ambang batas (IPK 3.30, Tes 68, Pengalaman 4 th, S1 STEM, CDI 0.82).
  3. *🔴 Kurang (Tolak):* Profil di bawah standar (IPK 2.30, Tes 42, Pengalaman 0.5 th, SMA, kuliah aktif penuh waktu).
- **Kriteria Penerimaan:** Menekan salah satu tombol preset harus memperbarui seluruh nilai field formulir seketika melalui `st.session_state` tanpa error.

### FR-02: Validasi Formulir Kualifikasi Masukan
- **Deskripsi:** Sistem memvalidasi seluruh input pengguna menggunakan skema `ApplicantInput` (Pydantic v2) sebelum inferensi dijalankan.
- **Batasan Validasi:**
  - `gpa`: Bilangan riil, $2.00 \le \text{IPK} \le 4.00$.
  - `test_score`: Bilangan riil, $0.0 \le \text{skor} \le 100.0$.
  - `experience_years`: Bilangan riil, $0.0 \le \text{tahun} \le 30.0$.
  - `training_hours`: Bilangan riil, $0.0 \le \text{jam} \le 500.0$.
  - `city_development_index`: Bilangan riil, $0.40 \le \text{CDI} \le 1.00$.
  - `education_level`: Pilihan kategori eksklusif (`Graduate`, `Masters`, `Phd`, `High School`, `Primary School`).
  - `major_discipline`: Pilihan kategori (`STEM`, `Business Degree`, `Arts`, `Humanities`, `No Major`, `Other`).
  - `relevent_experience`: Pilihan kategori (`Has relevent experience`, `No relevent experience`).
  - `enrolled_university`: Pilihan kategori (`no_enrollment`, `Full time course`, `Part time course`).
- **Kriteria Penerimaan:** Setiap input di luar batasan ditolak oleh form UI dan memicu `ValidationError` jika diakses via API/skrip.

### FR-03: Inferensi Klasifikasi Machine Learning Lokal
- **Deskripsi:** Sistem memuat model terlatih `classifier_model.joblib` dan pipeline prapemrosesan `preprocessor_pipeline.joblib`, mentransformasikan data masukan, dan menghitung probabilitas kesesuaian positif $P \in [0.0, 1.0]$.
- **Aturan Pemetaan Status Keputusan:**
  - $P \ge 0.75 \rightarrow$ Status **"Terima"**
  - $0.45 \le P < 0.75 \rightarrow$ Status **"Pertimbangan"**
  - $P < 0.45 \rightarrow$ Status **"Tolak"**
- **Kriteria Penerimaan:** Inferensi berjalan lokal dalam waktu $< 100$ ms dan menghasilkan objek terstruktur `PredictionResult`.

### FR-04: Qualification Gap Analysis Komparatif
- **Deskripsi:** Sistem membandingkan nilai profil pelamar terhadap tolok ukur kandidat ideal (`ideal_numeric_benchmarks`) dari `models/feature_metadata.json`:
  - `gpa`: $3.45$
  - `test_score`: $78.5$
  - `experience_years`: $12.0$
  - `training_hours`: $50.0$
  - `city_development_index`: $0.91$
- **Kriteria Penerimaan:**
  - Atribut di bawah benchmark otomatis dicatat ke list `weaknesses` lengkap dengan kalkulasi selisih nilai minus.
  - Atribut sama atau di atas benchmark dicatat ke list `strengths`.
  - Atribut kategorikal dinilai berdasarkan relevansi industri teknologi.

### FR-05: Pembangkitan Umpan Balik Karir Gemini AI
- **Deskripsi:** Sistem mengirimkan prompt terstruktur berisi profil pelamar, status prediksi, skor probabilitas, serta poin kekuatan dan kelemahan ke Google Gemini API menggunakan model `gemini-2.5-flash` (dengan fallback otomatis ke model flash aktif lainnya).
- **Format Output Wajib:** Respons berformat Markdown terdiri dari 3 bagian:
  1. `### 1. Ringkasan Evaluasi & Surat Umpan Balik Karir`
  2. `### 2. Analisis Kesenjangan Kualifikasi (Gap Analysis)`
  3. `### 3. Peta Jalan Pengembangan Diri & Rekomendasi Materi Belajar (Upskilling)`
- **Kriteria Penerimaan:** Jika API key kosong, kuota habis, atau koneksi terputus, sistem secara otomatis mengeksekusi *Deterministic Fallback Template* tanpa menyebabkan web aplikasi crash.

### FR-06: Ekspor Berkas Laporan Evaluasi
- **Deskripsi:** Sistem menyediakan tombol unduh laporan resmi hasil evaluasi pelamar dalam format Markdown (`.md`).
- **Kriteria Penerimaan:** Berkas yang diunduh memuat seluruh rincian kualifikasi, skor probabilitas, badge status keputusan, hasil gap analysis, dan draf umpan balik karir dengan penamaan berkas otomatis: `Laporan_SIKUKER_{Nama}_{Status}_{YYYYMMDD}.md`.

### FR-07: Indikator Kesiapan Infrastruktur Real-Time
- **Deskripsi:** Sistem menampilkan status kesiapan komponen backend pada sidebar:
  - Indikator artefak model `.joblib` (`🟢 Ready` atau `🟡 Mock Mode`).
  - Indikator ketersediaan kredensial Gemini API (`🟢 Online` atau `🔵 Fallback Template`).
- **Kriteria Penerimaan:** Perubahan konfigurasi status langsung terdeteksi pada rendering halaman berikutnya.

---

## 4. Kebutuhan Non-Fungsional (Non-Functional Requirements)

### 4.1 Kinerja (Performance - NFR-01)
- Pemuatan artefak model serialisasi ke memori harus di-cache menggunakan `@functools.lru_cache(maxsize=1)`.
- Waktu inferensi klasifikasi Machine Learning lokal harus $\le 200$ milidetik.
- Batas waktu tunggu (*network timeout*) untuk pemanggilan Gemini API dibatasi maksimal 10 detik untuk mencegah aplikasi menggantung (*hang*).

### 4.2 Keandalan & Ketahanan (Reliability & Fault-Tolerance - NFR-02)
- Sistem harus memiliki ketersediaan 99.9% saat offline berkat arsitektur *Zero-Crash Fallback*:
  - Jika berkas `.joblib` tidak ditemukan, sistem otomatis mengaktifkan *Mock Model* berbasis formula kualifikasi deterministik.
  - Jika Gemini API gagal merespons, sistem menghasilkan umpan balik komprehensif melalui *Deterministic Fallback Engine*.

### 4.3 Keamanan Data & Kredensial (Security - NFR-03)
- Kunci API Google Gemini (`GEMINI_API_KEY`) tidak boleh di-hardcode di dalam kode sumber aplikasi.
- Seluruh variabel sensitif disimpan di dalam berkas `.env` lokal yang diabaikan oleh `.gitignore`.
- Tidak ada data pribadi pelamar yang disimpan secara permanen ke basis data publik (*stateless session-based processing*).

### 4.4 Keterpeliharaan & Kualitas Kode (Maintainability - NFR-04)
- Seluruh fungsi utama wajib memiliki anotasi tipe data (*type hints*) eksplisit.
- Mengikuti pedoman penulisan kode Python PEP 8 dan dokumentasi docstring lengkap.
- Cakupan pengujian unit (*test coverage*) pada lapisan service dan skema minimal 90%.

---

## 5. Matriks Keterlacakan Kebutuhan (Requirements Traceability Matrix)

| Kebutuhan SRS | Modul Implementasi | Berkas Pengujian (Pytest) | Kriteria Verifikasi |
| :--- | :--- | :--- | :--- |
| **FR-01** (Preset Cepat) | `app/main_ui.py` | Browser Testing E2E | Lolos Uji Interaktif |
| **FR-02** (Validasi Input) | `app/schemas/applicant_schema.py` | `tests/test_schemas.py` | 16 Kasus Uji Validasi Passed |
| **FR-03** (Inferensi ML) | `app/services/ml_service.py` | `tests/test_ml_service.py` | 6 Kasus Uji Inferensi Passed |
| **FR-04** (Gap Analysis) | `app/services/gap_analyzer.py` | `tests/test_gap_analyzer.py` | 4 Kasus Uji Benchmark Passed |
| **FR-05** (Umpan Balik AI) | `app/services/gemini_service.py` | `tests/test_gemini_service.py` | 4 Kasus Uji LLM Passed |
| **FR-06** (Unduh Laporan) | `app/main_ui.py` | Browser Testing E2E | File `.md` Terunduh Sempurna |
| **FR-07** (Indikator Status) | `app/main_ui.py` & `app/core/config.py` | Browser Testing E2E | Status Sidebar Tepat |
