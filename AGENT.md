# AGENT.md — System Instructions & Project Guidelines for OpenCode Agent

## 1. Project Overview & Problem Statement
- **Project Title:** SIKUKER: Sistem Klasifikasi Kesesuaian Kualifikasi Kerja Berdasarkan Machine Learning dengan Pemanfaatan Gemini API untuk Penyempurnaan Umpan Balik Karir.
- **Core Objective:** 
  1. Predict job applicant qualification suitability into 3 statuses (**Terima** / **Pertimbangan** / **Tolak**) along with a probability score using a pre-trained Machine Learning model (Random Forest / XGBoost).
  2. Identify specific qualification gaps/weaknesses from the applicant's profile (e.g., education level, relevant experience, GPA/test score, training hours).
  3. Send the prediction status, probability score, and identified weaknesses to the **Google Gemini API** to generate polite, constructive, and personalized career feedback along with an actionable upskilling learning list in Indonesian.
- **Reference Dataset:** Kaggle HR Analytics: Job Change of Data Scientists (`arashnic/hr-analytics-job-change-of-data-scientists`), adapted with applicant qualification features (Education, Experience, GPA/Test Score, Training Hours, City Development Index).

---

## 2. Decoupled Pipeline Architecture (CRITICAL RULE)
This project strictly enforces a **Split-Pipeline Strategy** between Model Training and Web Application Runtime:

1. **Training Environment (`colab_pipeline/`):**
   - All Exploratory Data Analysis (EDA), data cleaning, feature engineering, model training (Random Forest / XGBoost), and hyperparameter tuning happen **exclusively in Google Colab**.
   - The Colab pipeline exports three serialized artifacts into the local `models/` directory:
     - `models/classifier_model.joblib`: The trained classification model.
     - `models/preprocessor_pipeline.joblib`: The fitted scikit-learn preprocessing pipeline (`ColumnTransformer` / `Pipeline`).
     - `models/feature_metadata.json`: Feature names, exact column ordering, categorical options, and ideal benchmark thresholds for weakness detection.

2. **Application Environment (`app/` in Antigravity):**
   - **NEVER train or fit a Machine Learning model inside `app/` during web runtime.**
   - The local web application MUST only **load** the pre-trained artifacts from `models/` using `joblib.load()` and run `.transform()` and `.predict_proba()`.
   - If `models/*.joblib` files do not exist yet during early development, create a clean **Mock/Fallback Mode** in `app/services/ml_service.py` (with a clear UI warning banner) so frontend and Gemini API integration can still be tested independently.

---

## 3. SDLC Waterfall Workflow Compliance
You must follow the **SDLC Waterfall** sequence strictly. Before writing or modifying application code, verify which phase the user is working on and ensure alignment with `docs/`:

1. **Phase 1 — Requirements (`docs/01_requirements_spec.md`):**
   - Do not add unrequested features outside the defined Functional Requirements (FR-01 to FR-07).
2. **Phase 2 — System & Software Design (`docs/02_system_design.md`):**
   - Ensure feature names, data types, and probability thresholds match the design specification before implementing services.
   - **3-Tier Probability Threshold Mapping:**
     - `probability >= 0.75` -> **"Terima"**
     - `0.45 <= probability < 0.75` -> **"Pertimbangan"**
     - `probability < 0.45` -> **"Tolak"**
3. **Phase 3 — Implementation (`colab_pipeline/` & `app/`):**
   - Implement modules cleanly following the separated directory structure.
   - Maintain strict Separation of Concerns: UI components (`app/main_ui.py`) must never call `joblib` or the Gemini SDK directly; they must call service functions inside `app/services/`.
4. **Phase 4 — Verification & Testing (`tests/` & `docs/04_testing_report.md`):**
   - Every core service (`ml_service.py`, `gap_analyzer.py`, `gemini_service.py`) must have corresponding unit tests in `tests/` using `pytest`.

---

## 4. Directory Structure & File Responsibilities
Always place new files or edits in their designated directories:

- `colab_pipeline/`: Jupyter notebooks (`.ipynb`) or standalone training scripts meant to be executed in Google Colab.
- `models/`: Read-only directory for the web app containing `.joblib` and `.json` artifacts exported from Colab.
- `app/core/config.py`: Environment variable loading (`python-dotenv`), file paths, and threshold constants.
- `app/schemas/applicant_schema.py`: Pydantic models for validating applicant input and structured outputs.
- `app/services/ml_service.py`: Artifact loading (cached), feature transformation, probability calculation, and status mapping.
- `app/services/gap_analyzer.py`: Rule-based and metadata-driven comparison to extract weak qualification attributes from an applicant's input.
- `app/services/gemini_service.py`: Prompt construction and API calls to Gemini (`google-genai` SDK) with graceful error/fallback handling.
- `app/utils/prompt_templates.py`: System instructions and prompt templates in formal, supportive Indonesian.
- `app/main_ui.py`: Streamlit web interface (or `app/api_server.py` if using FastAPI).
- `tests/`: Automated unit and integration tests.
- `docs/`: Waterfall SDLC documentation files.

---

## 5. Coding Standards & Technical Rules

### A. Python & Code Quality
- Use **Python 3.10+** syntax with explicit **Type Hints** (`typing`, `pydantic`) on all function signatures and return values.
- Write clear docstrings explaining inputs, outputs, and edge cases.
- Handle exceptions gracefully (`try...except` blocks with informative error messages) especially around file I/O (`joblib.load`) and external API calls.

### B. Machine Learning Inference Rules (`app/services/ml_service.py`)
- Convert incoming validated Pydantic applicant data into a `pandas.DataFrame` with **exact column names and order** as specified in `models/feature_metadata.json`.
- Always apply `preprocessor_pipeline.transform(df)` (NEVER `.fit_transform()`) before passing features to `classifier_model.predict_proba()`.
- Cache loaded `.joblib` models in memory (e.g., using `@st.cache_resource` in Streamlit or `functools.lru_cache` in Python/FastAPI) to prevent disk I/O bottlenecks on every request.

### C. Gemini API Integration Rules (`app/services/gemini_service.py`)
- **Use the official Google GenAI SDK:** `from google import genai` (package: `google-genai`). Do NOT use the deprecated `google-generativeai` package.
- Default model to use: `gemini-2.5-flash`.
- Read the API key strictly from environment variables (`os.getenv("GEMINI_API_KEY")`). Never hardcode API keys.
- **Prompt Engineering Requirements:**
  - Role: Senior HR Career Coach & Technical Recruiter.
  - Language: Professional, polite, empathetic, and constructive **Bahasa Indonesia**.
  - Required Sections in Output:
    1. **Ringkasan Evaluasi Kualifikasi** (Tailored to whether the status is *Terima*, *Pertimbangan*, or *Tolak*).
    2. **Area Pengembangan Prioritas** (Directly addressing the weaknesses detected by `gap_analyzer.py`).
    3. **Rekomendasi Materi Belajar & Upskilling** (3 to 5 concrete courses, topics, certifications, or portfolio project ideas).
- **Fallback Mechanism:** If `GEMINI_API_KEY` is missing or the API request fails/times out, return a clean, deterministic fallback Markdown template based on the detected weaknesses so the application never crashes.

---

## 6. Step-by-Step Execution Order for Agent Tasks
When asked to build or scaffold this project from scratch, execute in this exact order:
1. Create `requirements.txt`, `.env.example`, and `.gitignore`.
2. Create `models/feature_metadata.json` with the standard qualification feature schema and benchmark thresholds.
3. Build `colab_pipeline/` script/notebook code that trains the model on the CSV dataset and exports the exact artifacts expected by `models/`.
4. Build `app/core/config.py` and `app/schemas/applicant_schema.py`.
5. Build `app/services/gap_analyzer.py`, `app/services/ml_service.py`, and `app/services/gemini_service.py` (along with `app/utils/prompt_templates.py`).
6. Build `app/main_ui.py` (Streamlit UI) connecting the input form, ML prediction cards, weakness breakdown, and Gemini career feedback display.
7. Write unit tests inside `tests/` and verify all modules run without import or runtime errors.