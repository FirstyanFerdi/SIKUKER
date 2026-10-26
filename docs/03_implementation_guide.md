# Technical Implementation & Developer Guide
# SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja Berdasarkan Machine Learning dengan Pemanfaatan Gemini API untuk Penyempurnaan Umpan Balik Karir

- **Dokumen Identifikasi:** IMPL-SIKUKER-03
- **Fase SDLC:** Fase 3 — Implementation
- **Status Dokumen:** Baseline / Approved
- **Versi Dokumen:** 2.0.0 (Comprehensive Expansion)
- **Terakhir Diperbarui:** 26 Oktober 2026

---

## 1. Prasyarat Lingkungan Pengembangan (Development Prerequisites)

### 1.1 Persyaratan Sistem Operasi & Runtime
- **Sistem Operasi:** Windows 10/11 64-bit, Linux (Ubuntu 20.04+), atau macOS (Monterey+).
- **Python Version:** Python 3.10 hingga 3.13 (Teruji resmi pada Python 3.13.14 x64 di Windows).
- **Pengelola Paket:** `pip` versi 24.0+ atau `uv`.
- **Manajemen Lingkungan Virtual:** Sangat disarankan menggunakan `venv` atau `conda` untuk mengisolasi dependensi.

### 1.2 Matriks Dependensi Terkunci (`requirements.txt`)
Berikut adalah daftar dependensi dengan versi yang diselaraskan secara presisi dengan artefak ekspor Google Colab:
```txt
streamlit>=1.35.0
pandas==2.2.3
numpy>=1.24.0
scikit-learn==1.6.1
xgboost==3.4.1
joblib==1.6.0
pydantic>=2.5.0
python-dotenv>=1.0.0
google-genai>=0.1.0
pytest>=8.0.0
```

> **Catatan Teknis Krusial (Version Pinning):**
> Pustaka `scikit-learn` wajib dikunci pada versi `1.6.1`. Versi `scikit-learn >= 1.7.0` melakukan restrukturisasi modul internal pada kelas `_RemainderColsList` di dalam `sklearn.compose._column_transformer`. Menggunakan versi di atas 1.6.1 akan memicu `AttributeError: Can't get attribute '_RemainderColsList'` saat proses unpickling berkas `models/preprocessor_pipeline.joblib`.

---

## 2. Blueprint Struktur Repositori & Tanggung Jawab Modul

```text
SIKUKER/
├── .env                          # Kredensial lokal sensitif (API Key) - JANGAN DI-COMMIT
├── .env.example                  # Template variabel lingkungan
├── .gitignore                    # Konfigurasi pengabaian berkas git
├── AGENT.md                      # Pedoman instruksi dan batasan sistem OpenCode Agent
├── PRD.md                        # Dokumen persyaratan produk (Product Requirements)
├── PROJECT_PLAN.md               # Rencana eksekusi tahapan SDLC Waterfall
├── PROSES.md                     # Log riwayat pelaksanaan proses rekayasa sistem
├── README.md                     # Berkas dokumentasi utama repositori
├── requirements.txt              # Daftar pustaka dependensi terstandarisasi
│
├── app/                          # Paket aplikasi web lokal SIKUKER
│   ├── __init__.py               # Inisialisasi package & versi aplikasi
│   ├── main_ui.py                # Presentation Layer berbasis antarmuka Streamlit
│   │
│   ├── core/                     # Modul konfigurasi sistem
│   │   ├── __init__.py           # Re-ekspor path & konstanta ambang batas
│   │   └── config.py             # Pemuatan .env dinamis & resolusi path artefak
│   │
│   ├── schemas/                  # Modul validasi data (Pydantic v2)
│   │   ├── __init__.py           # Re-ekspor skema masukan dan keluaran
│   │   └── applicant_schema.py   # Kelas ApplicantInput & PredictionResult
│   │
│   ├── services/                 # Lapisan logika bisnis & inferensi cerdas
│   │   ├── __init__.py           # Re-ekspor fungsi layanan
│   │   ├── gap_analyzer.py       # Penganalisis kesenjangan benchmark kandidat
│   │   ├── ml_service.py         # Loader model, transformator fitur, & inferensi
│   │   └── gemini_service.py     # Integrasi Google GenAI SDK & fallback engine
│   │
│   └── utils/                    # Modul utilitas & rekayasa prompt
│       ├── __init__.py           # Re-ekspor utilitas prompt
│       └── prompt_templates.py   # System instructions & pembangun prompt AI
│
├── docs/                         # Dokumentasi formal SDLC Waterfall
│   ├── 01_requirements_spec.md   # Software Requirements Specification (SRS)
│   ├── 02_system_design.md       # Software Design Document (SDD)
│   ├── 03_implementation_guide.md# Technical Implementation Guide (Dokumen ini)
│   └── 04_testing_report.md      # Testing & Verification Report
│
├── models/                       # Artefak serialisasi model (Read-Only)
│   ├── classifier_model.joblib   # Model terlatih RandomForest (200 estimator)
│   ├── preprocessor_pipeline.joblib # Pipeline ColumnTransformer (Scaler & Encoder)
│   └── feature_metadata.json     # Metadata fitur, batas nilai, & benchmark ideal
│
└── tests/                        # Suite automated unit testing (Pytest)
    ├── __init__.py               # Inisialisasi package test
    ├── test_gap_analyzer.py      # Pengujian unit aturan gap analyzer
    ├── test_gemini_service.py    # Pengujian prompt builder & fallback AI
    ├── test_ml_service.py        # Pengujian inferensi model & status threshold
    └── test_schemas.py           # Pengujian validasi batas nilai skema Pydantic
```

---

## 3. Rincian Implementasi Kode Sumber

### 3.1 Konfigurasi Sistem (`app/core/config.py`)
Modul ini bertugas membaca berkas `.env` secara dinamis dan menetapkan path absolut artefak model:
- Menggunakan `load_dotenv(dotenv_path=ENV_FILE_PATH, override=True)` agar setiap perubahan kunci API langsung terbaca tanpa memerlukan restart interpreter Python.
- Menggunakan `Path(__file__).resolve().parent.parent.parent` untuk menjamin resolusi path berkas model selalu tepat terlepas dari direktori kerja terminal saat aplikasi dijalankan.
- Mendefinisikan konstanta ambang batas klasifikasi:
  ```python
  TERIMA_THRESHOLD: float = 0.75
  PERTIMBANGAN_THRESHOLD: float = 0.45
  ```

### 3.2 Skema Data Masukan & Keluaran (`app/schemas/applicant_schema.py`)
Menerapkan validasi data berbasis Pydantic v2:
- **`ApplicantInput`:** Memvalidasi ke-9 atribut kualifikasi. Jika ada masukan di luar rentang (seperti IPK $> 4.00$ atau skor tes $< 0.0$), Pydantic akan langsung melempar `ValidationError`.
- **`to_dataframe()`:** Mengonversi data masukan tervalidasi menjadi objek `pandas.DataFrame` 1-baris dengan urutan kolom persis mengikuti daftar `all_features_order` pada `feature_metadata.json`. Hal ini krusial agar matriks masukan pada pipeline Scikit-Learn tidak tertukar.
- **`PredictionResult`:** Struktur data standar yang merangkum nilai `probability` ($0.0 - 1.0$), `percentage` ($0.0 - 100.0\%$), status keputusan 3-tier (`Terima`, `Pertimbangan`, atau `Tolak`), serta daftar string `strengths` dan `weaknesses`.

### 3.3 Layanan Inferensi Machine Learning (`app/services/ml_service.py`)
- **Caching Artefak:** Menggunakan `@functools.lru_cache(maxsize=1)` pada fungsi `load_ml_artifacts()`:
  ```python
  @functools.lru_cache(maxsize=1)
  def load_ml_artifacts():
      model = joblib.load(MODEL_PATH)
      preprocessor = joblib.load(PREPROCESSOR_PATH)
      ...
  ```
  Ini mencegah operasi I/O pembacaan file 8.8 MB pada setiap klik tombol di antarmuka web, sehingga respons inferensi menjadi seketika ($< 50$ ms).
- **Prapemrosesan Tanpa Fitting Ulang:** Fitur hanya ditransformasikan menggunakan `preprocessor.transform(df)`. Penggunaan `.fit()` atau `.fit_transform()` pada runtime web dilarang keras untuk mencegah data leakage dan kerusakan parameter skala.
- **Mekanisme Mock Model:** Jika berkas artefak belum diekspor atau rusak, fungsi otomatis beralih ke `MockModel` dan `MockPreprocessor` yang menghitung skor kelayakan heuristik tertimbang secara deterministik, menjamin aplikasi tidak pernah mengalami *crash*.

### 3.4 Layanan Qualification Gap Analyzer (`app/services/gap_analyzer.py`)
- Membandingkan nilai fitur numerik pelamar terhadap benchmark ideal:
  ```python
  if applicant.gpa >= gpa_bench:
      strengths.append(f"IPK ({applicant.gpa:.2f}) unggul dan memenuhi benchmark kelulusan ({gpa_bench:.2f}).")
  else:
      gap = gpa_bench - applicant.gpa
      weaknesses.append(f"IPK ({applicant.gpa:.2f}) masih berada di bawah target benchmark ({gpa_bench:.2f}), selisih -{gap:.2f} poin.")
  ```
- Mengevaluasi fitur kategorikal:
  - `relevent_experience == "No relevent experience"` $\rightarrow$ dicatat sebagai kelemahan utama.
  - `education_level` di bawah `"Graduate"` (SMA/SMP) $\rightarrow$ dicatat sebagai kesenjangan standar formal.
  - `enrolled_university == "Full time course"` $\rightarrow$ dicatat sebagai potensi kendala komitmen jam kerja reguler.

### 3.5 Layanan Integrasi Gemini API & Ketahanan Model (`app/services/gemini_service.py`)
- **Inisialisasi Klien dengan Timeout:**
  ```python
  client = genai.Client(
      api_key=api_key,
      http_options=types.HttpOptions(timeout=10.0),
  )
  ```
  Batas waktu 10 detik menjamin bahwa antarmuka web Streamlit tidak akan menggantung jika terjadi kelambatan jaringan pada koneksi Google Cloud.
- **Multi-Model Auto-Fallback:**
  Sistem mencoba model utama (`gemini-2.5-flash`), dan jika mengalami kendala (misal model 404 pada API key tertentu), sistem secara transparan mencoba kandidat model flash lain (`gemini-3.8-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`) sebelum beralih ke mode offline.
- **Deterministic Offline Fallback:**
  Jika tidak ada koneksi internet atau kunci API tidak terpasang, sistem memanggil `generate_fallback_feedback()` yang merangkum hasil evaluasi Machine Learning ke dalam format Markdown 3 bagian yang rapi dan kontekstual.

### 3.6 Antarmuka Web Streamlit (`app/main_ui.py`)
- Menggunakan tata letak `wide` dengan styling CSS modern pada kartu status, kotak kelebihan (*strengths*), dan kotak kekurangan (*weaknesses*).
- Sidebar menyediakan tombol preset instan (`🟢 Kuat`, `🟡 Pertimbangan`, `🔴 Kurang`) yang memperbarui `st.session_state` secara langsung.
- Formulir 3 kolom dirancang intuitif dengan nilai default, batas minimum, batas maksimum, dan petunjuk penggunaan (*help tooltip*).
- Hasil inferensi disajikan secara visual dengan badge warna bersinar, bilah persentase probabilitas, dan tombol ekspor laporan resmi berekstensi `.md`.

---

## 4. Panduan Menjalankan Sistem Secara Lokal

### 4.1 Pembuatan Virtual Environment (Direkomendasikan)
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 Pemasangan Dependensi
```bash
pip install -r requirements.txt
```

### 4.3 Menyiapkan Berkas Kredensial `.env`
Salin template `.env.example` ke `.env`:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```
Sunting berkas `.env` dan masukkan kunci API Gemini Anda:
```env
GEMINI_API_KEY=AIzaSyD...your_actual_api_key_here
GEMINI_MODEL=gemini-3.1-flash-lite
```

### 4.4 Menjalankan Aplikasi Web Streamlit
```bash
python -m streamlit run app/main_ui.py
# Atau jika script streamlit berada di PATH
streamlit run app/main_ui.py
```
Aplikasi akan terbuka otomatis di peramban web pada alamat:
`http://localhost:8501`

### 4.5 Menghentikan Aplikasi Web (Stop Server)
Untuk menghentikan server Streamlit:
- **Opsi 1 (Terminal Terbuka):** Tekan kombinasi tombol `Ctrl + C` pada jendela terminal yang menjalankan Streamlit.
- **Opsi 2 (PowerShell / Command Prompt):**
  ```powershell
  # Hentikan proses yang berjalan di port 8501
  Stop-Process -Id (Get-NetTCPConnection -LocalPort 8501).OwningProcess -Force
  ```
- **Opsi 3 (Windows Task Manager):** Buka Task Manager (`Ctrl + Shift + Esc`), cari proses **Python** / **Streamlit**, dan klik **End Task**.

### 4.6 Menjalankan Pengujian Unit Otomatis (Pytest)
Jalankan perintah berikut untuk memvalidasi integritas seluruh modul:
```bash
python -m pytest -v
```
Seluruh 30 kasus uji pada suite pengujian wajib berstatus **PASSED**.

---

## 5. Troubleshooting & FAQ (Solusi Kendala Teknis)

### Q1: Muncul `AttributeError: Can't get attribute '_RemainderColsList'` saat memuat model
- **Penyebab:** Versi `scikit-learn` pada lingkungan lokal lebih tinggi dari `1.6.1` (misalnya versi 1.7+ atau 1.9+).
- **Solusi:** Jalankan perintah `pip install scikit-learn==1.6.1` untuk menurunkan versi ke versi yang sama persis dengan versi serialisasi artefak di Google Colab.

### Q2: Muncul peringatan `The script streamlit.exe is installed in ... which is not on PATH`
- **Penyebab:** Direktori Scripts Python belum terdaftar di variabel lingkungan `PATH` Windows.
- **Solusi:** Jalankan aplikasi menggunakan pemanggilan modul Python eksplisit: `python -m streamlit run app/main_ui.py` atau `python -m pytest -v`.

### Q3: Tampilan web menunjukkan `ℹ️ Gemini AI: Offline / Fallback Template`
- **Penyebab:** Nilai `GEMINI_API_KEY` di berkas `.env` masih kosong atau berisi teks placeholder `your_gemini_api_key_here`.
- **Solusi:** Masukkan API key Gemini yang valid dari [Google AI Studio](https://aistudio.google.com/). Sistem tetap dapat berjalan normal dan menghasilkan umpan balik melalui template deterministik lokal.

### Q4: Muncul `httpx.ConnectTimeout` saat memanggil Gemini API
- **Penyebab:** Koneksi internet lokal mengalami gangguan atau latensi tinggi ke server Google Cloud.
- **Solusi:** Modul `gemini_service.py` telah dilengkapi batas waktu 10 detik dan secara otomatis mengalihkan tampilan ke *Deterministic Fallback* tanpa membuat aplikasi web terhenti (*hang*).
