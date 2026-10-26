# Project Plan & SDLC Waterfall Execution Roadmap

**Proyek:** Sistem Klasifikasi Kesesuaian Kualifikasi Kerja & Umpan Balik Karir (ML + Gemini API)  
**Model SDLC:** Classic Waterfall (Sequential Phase-Gate)  
**Strategi Teknis:** Pemisahan Pipeline (*Google Colab Training* -> *Artifact Handoff* -> *Antigravity Local App*)  

---

## 1. Aturan Fase-Gate SDLC Waterfall (Phase-Gate Governance)
Proyek ini menggunakan metode **SDLC Waterfall**, di mana setiap fase memiliki **Kriteria Masuk (*Entry Criteria*)**, **Deliverables (Luaran)**, dan **Kriteria Keluar (*Exit Gate*)**. Fase berikutnya tidak boleh dieksekusi sebelum artefak pada fase sebelumnya dinyatakan lengkap dan valid.

```text
[FASE 1: Requirements] ──► [FASE 2: System Design] ──► [FASE 3A: Colab ML Pipeline]
                                                                │
                                            (Handoff: .joblib & feature_metadata.json)
                                                                ▼
[FASE 5: Deployment/Docs] ◄── [FASE 4: Verification/Test] ◄── [FASE 3B: Local Web App]