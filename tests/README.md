# Automated Test Suite & Validation Audit

Comprehensive automated test suite and validation audit for `handwritten-digit-ensemble` (48-class classical ML character recognition ensemble).

---

## 🚀 How to Run

All commands assume a PowerShell terminal on Windows within the project root:

```powershell
cd handwritten-digit-ensemble
```

### 1. Fast Smoke Suite (Pre-Commit / Fast Verification)
Executes all critical-path tests in **< 5 seconds** (budget was < 60s):
```powershell
.\venv\Scripts\python.exe -m pytest -m smoke
```
Or via the system verification runner:
```powershell
.\venv\Scripts\python.exe scratch/verify_all_systems.py
```

### 2. Standard Unit & Integration Test Suite
Runs all tests except end-to-end browser/server tests:
```powershell
.\venv\Scripts\python.exe -m pytest
```

### 3. Full Coverage Report
Calculates exact line coverage across the `src/` modules:
```powershell
.\venv\Scripts\python.exe -m pytest --cov=src --cov-report=term-missing
```

### 4. End-to-End (E2E) Server Tests
Requires the Streamlit server running on `http://localhost:8501`:
```powershell
# In terminal 1:
.\venv\Scripts\python.exe -m streamlit run src/app.py

# In terminal 2:
.\venv\Scripts\python.exe -m pytest -m e2e
```

### 5. Running Specific Test Modules
```powershell
.\venv\Scripts\python.exe -m pytest tests/test_preprocessing.py
.\venv\Scripts\python.exe -m pytest tests/test_validation.py
.\venv\Scripts\python.exe -m pytest tests/test_prediction.py
.\venv\Scripts\python.exe -m pytest tests/test_disambiguation.py
.\venv\Scripts\python.exe -m pytest tests/test_features.py
.\venv\Scripts\python.exe -m pytest tests/test_feedback.py
.\venv\Scripts\python.exe -m pytest tests/test_model_data.py
.\venv\Scripts\python.exe -m pytest tests/test_security.py
.\venv\Scripts\python.exe -m pytest tests/test_ui_canvas.py
```

---

## 🏷️ Test Markers (`pytest.ini`)

| Marker | Description | Typical Runtime | Default Execution |
| :--- | :--- | :--- | :--- |
| **`smoke`** | Fast critical-path tests (bounding box, auto-invert, empty gate, multipart symbols, label map, baseline check). | ~3.7 seconds | Included |
| **`slow`** | Extensive benchmarks, accuracy regressions, multi-sample homoglyph audits, latency benchmarks. | ~15-20 seconds | Included |
| **`e2e`** | End-to-end integration tests checking live Streamlit endpoints (`http://localhost:8501`). | ~1 second | Excluded (`addopts = -m "not e2e"`) |

---

## 📁 Test Architecture & Layout

```
tests/
├── conftest.py             # Shared fixtures: draw_helpers, session-scoped loaded_model, temp_feedback_dir
├── test_preprocessing.py   # PP-01 to PP-10: Inversion, faint pixels, aspect ratio, centering, thickening
├── test_validation.py      # VA-01 to VA-12, V-08, V-09, V-10: Empty, graphics, multi-character, OOD, multipart
├── test_prediction.py      # PR-01 to PR-11: 48 classes, weights blend, hard voting, category, latency
├── test_disambiguation.py  # DS-01 to DS-06: Morphology twins (P/9, O/0, 6/b, Z/2, S/5), star heuristic
├── test_features.py        # HOG descriptor, L2-Hys normalization, projection profiles, feature benchmark
├── test_feedback.py       # FB-01 to FB-10: Feedback saving, concurrency, retraining promotion, isolation
├── test_model_data.py      # MD-01 to MD-10: Dataset balance, row leakage, weights, accuracy floors, robustness
├── test_security.py        # SR-01 to SR-04: XSS sanitization, path traversal, memory stability, health check
├── test_ui_canvas.py       # UI-01 to UI-13: Canvas size, brush presets, clear, sliders, quality warnings
├── BUG_REPORT.md           # Full audit report of bugs, anomalies, and structural defects discovered
└── README.md               # Test documentation, markers, and execution guide
```

---

## 📋 Validation Rules to Test Matrix (V-01 through V-11)

| Rule ID | Architectural Specification | Implementation Function / Location | Concrete Automated Tests |
| :--- | :--- | :--- | :--- |
| **V-01** | Empty Canvas Rejection (ink pixels < 20) | `src/validation.py::validate_digit_input` (Gate 1) | `test_VA_01_blank_canvas_rejection_no_model_call`, `test_VA_02_empty_gate_boundary_19_vs_20_pixels`, `test_VA_09_tiny_speck_noise_treated_as_empty` |
| **V-02** | Solid Block / Photo Rejection (fill > 82% AND ink > 4,000) | `src/validation.py::validate_digit_input` (Gate 2) | `test_VA_03_solid_block_graphic_rejection`, `test_VA_04_large_letter_bold_brush_accepted` |
| **V-03** | Multi-Character Rejection (horizontal comps $\ge 2$, width $> 70$, aspect $> 1.6$) | `src/validation.py::validate_digit_input` (Gate 3) | `test_VA_05_multi_character_rejection` |
| **V-04** | Legitimate Multi-Part Symbols Allowed (`i`, `!`, `=`, `%`, `?`) | `src/validation.py::validate_digit_input` (Gate 3 false-positive guard) | `test_VA_06_multipart_symbols_not_rejected` (parametrized across all 5 symbols) |
| **V-05** | Scribble & Complex Artifact Rejection | `src/validation.py` + `src/predict.py` (OOD gate) | `test_VA_07_random_scribble_rejected` |
| **V-06** | Natural Photo / Noise Rejection | `src/validation.py` (Gate 2 & 3) | `test_VA_08_photo_like_noisy_image_rejected` |
| **V-07** | OOD Confidence Floor Rejection (ensemble max prob < 16.0%) | `src/app.py` & `src/validation.py` | `test_VA_12_ood_confidence_floor_boundary_15_9_vs_16_1` |
| **V-08** | File Upload Guard (valid format PNG/JPG/WebP, size limit) | `src/validation.py::validate_uploaded_file` | `test_V08_upload_valid_formats`, `test_VA_10_corrupt_non_image_upload_handling`, `test_VA_11_huge_image_resizing_guard` |
| **V-09** | Live Sandbox Weights Range (0..10, total > 0 fallback) | `src/predict.py::predict_digit` | `test_V09_weights_range_and_zero_fallback`, `test_PR_06_all_weights_zero_graceful_fallback`, `test_UI_06_all_zero_weights_fallback_graceful` |
| **V-10** | Feedback Label Integrity (must be one of 48 valid classes) | `src/feedback_manager.py::get_class_id_from_label` | `test_V10_feedback_label_validation`, `test_FB_04_label_outside_48_classes_rejected` |
| **V-11** | Consensus Guard (heuristics invoked only on disagreement / margin < 12%) | `src/disambiguation.py::disambiguate_homoglyphs` | `test_DS_03_consensus_guard_heuristics_not_invoked` |
