# Comprehensive System Validation & Bug Audit Report

**Project**: `handwritten-digit-ensemble` (48-Class Classical ML Handwritten Character Recognition)  
**Date of Audit**: 2026-11-15  
**Test Suite**: 88 automated tests across 9 test modules  
**Auditor**: Senior QA & ML Validation Engineer

---

## Executive Summary

A rigorous automated test harness (`tests/`) was constructed adhering strictly to real APIs, non-weakened assertions, and synthetic isolation. 
During the audit, **7 distinct bugs and architectural anomalies** were identified across data pipelines, core inference logic, web security, and preprocessing heuristics.

| Bug ID | Component | Severity | Description | Status |
| :--- | :--- | :--- | :--- | :--- |
| **BUG-01** | `src/predict.py` | 🔴 High | Division symbol `"/"` categorized as `Alphabet` instead of `Symbol` | **FIXED** (Commit `09da14a`) |
| **BUG-02** | `src/ui_helpers.py` | 🔴 Critical (Security) | Unescaped template kwargs passed to `st.markdown(..., unsafe_allow_html=True)` (XSS) | **FIXED** (Commit `0a7637d`) |
| **BUG-03** | `src/retrain_with_feedback.py` | 🟡 Medium | Silent overwrite of production model without candidate promotion check | **FIXED** (Step 4) |
| **BUG-04** | `data/character_dataset.npz` | 🔴 High (Data) | 2,963 duplicate rows & 1,760 all-zero images causing 240-row train/test leakage | **DOCUMENTED** (`test_MD_03`) |
| **BUG-05** | `models/character_ensemble.pkl` | 🟡 Medium (ML) | Classical model lacks translation invariance; 1px shift causes 41% accuracy drop | **DOCUMENTED** (`test_MD_09`) |
| **BUG-06** | `src/preprocessing.py` | 🟡 Low (Edge Case) | `arr_28 > 45` threshold fails to detect 1px downsampled strokes for auto-thickening | **DOCUMENTED** (`BUG-06`) |
| **BUG-07** | `src/validation.py` | 🟡 Low (Robustness) | Dense textured noise bypasses Gate 2 & Gate 3, relying solely on OOD gate | **DOCUMENTED** (`BUG-07`) |

---

## Detailed Bug Findings & Remediation

### BUG-01: Division Symbol Categorized as Alphabet (`src/predict.py`)
- **Severity**: High (Classification Logic)
- **Root Cause**: In `src/predict.py::get_character_category`, the condition `elif "/" in char_str:` was evaluated *before* checking membership in `SPECIAL_SYMBOLS_SET`. Because merged alphabet labels have the format `"A / a"`, any string containing `"/"` was classified as `Alphabet`. When a user drew or tested the math division symbol `"/"`, it was erroneously categorized as an Alphabet.
- **Automated Test**: `tests/test_prediction.py::test_PR_08_category_assignment_digit_alphabet_symbol`
- **Resolution**: Reordered conditionals to evaluate `char_str in SPECIAL_SYMBOLS_SET` first.
- **Commit**: `09da14a`

---

### BUG-02: Cross-Site Scripting (XSS) Vulnerability in HTML Templates (`src/ui_helpers.py`)
- **Severity**: Critical (Security)
- **Root Cause**: `render_template(template_name, **kwargs)` used `template.format(**safe_kwargs)` and injected the result directly into Streamlit via `st.markdown(rendered, unsafe_allow_html=True)`. Variables such as `digit`, `title`, and `description` were never HTML-escaped. Injected malicious strings (e.g. `<script>alert(1)</script>`) were executed directly in the user's browser.
- **Automated Test**: `tests/test_security.py::test_SR_01_xss_prevention_in_render_template`
- **Resolution**: Updated `render_template` to run all string keyword arguments through `html.escape(v)` before rendering.
- **Commit**: `0a7637d`

---

### BUG-03: Unverified Model Replacement in Feedback Loop (`src/retrain_with_feedback.py`)
- **Severity**: Medium (MLOps / Reliability)
- **Root Cause**: The feedback retraining script previously saved retrained weights directly onto `models/character_ensemble.pkl` without verifying whether validation accuracy improved or degraded against the production baseline.
- **Automated Test**: `tests/test_feedback.py::test_FB_07_retrain_sample_weight_and_candidate_promotion`
- **Resolution**: Implemented candidate promotion architecture: retrained model is saved to `models/candidate_ensemble.pkl`, evaluated against held-out validation data, promoted to production only if `val_acc >= baseline_acc`, and automatically rolled back if performance regresses.
- **Commit**: `0a7637d`

---

### BUG-04: Dataset Row Duplication, Blank Images & Train/Test Leakage (`data/character_dataset.npz`)
- **Severity**: High (Data Quality & Evaluation Integrity)
- **Root Cause**: Forensic analysis of the 33,600-sample dataset revealed:
  1. **Only 30,637 unique rows**: 2,963 duplicate rows exist across 519 unique hash groups.
  2. **1,760 all-zero completely blank images** ($28\times 28$ array sum $= 0.0$) exist across 22 alphabet classes (`S/s`, `X/x`, `H/h`, `I/i`, `O/o`, `C/c`, `K/k`, `P/p`, `L/l`, `W/w`, `T/t`, `Q/q`, `Z/z`, `F/f`, `G/g`, `V/v`, `E/e`, `Y/y`, `R/r`, `B/b`, `J/j`, `N/n`).
  3. **Cross-class blank collision**: Hash group `d7c27185...` (blank image) is simultaneously labeled as `W/w`, `Q/q`, `P/p`, and `F/f`.
  4. **Data Leakage**: Under standard 80/20 stratified split (`random_state=42`), **240 identical rows** appear simultaneously in both `X_train` and `X_test`.
- **Automated Test**: `tests/test_model_data.py::test_MD_03_zero_data_leakage_between_train_and_test` (Fails honestly to prevent weakening assertions).
- **Recommended Action**: Re-filter `data/character_dataset.npz` using `dataset_builder.py` with an explicit blank-image filter (`np.sum(row) > 10.0`) and hash-based deduplication before the next major training run.

---

### BUG-05: Spatial Translation Brittleness of Raw-Pixel Classifiers (`models/character_ensemble.pkl`)
- **Severity**: Medium (Model Robustness)
- **Root Cause**: Because classical classifiers (Logistic Regression, Random Forest, MLP) operate on flat 784-dimensional pixel vectors without convolutional receptive fields or spatial pooling, shifting an image by just 1 pixel diagonally (`shift(1, -1)`) moves every stroke off its expected coordinates. Accuracy drops from **87.3% to 45.6%** (a 41.67% drop).
- **Automated Test**: `tests/test_model_data.py::test_MD_09_model_robustness_perturbations`
- **Mitigation in Pipeline**: The production application mitigates this by mandating center-of-mass centering (`preprocessing.py::center_image`) to (14, 14) on all inputs before inference.
- **Future ML Recommendation**: Incorporate translation-augmented samples ($\pm 1$ px shifts) during retraining, or add translation-invariant spatial features (HOG, horizontal/vertical projection profiles).

---

### BUG-06: Auto-Thicken Binarization Threshold Edge Case (`src/preprocessing.py`)
- **Severity**: Low (Heuristic Sensitivity)
- **Root Cause**: Line 253 of `src/preprocessing.py` computes stroke width using `bin_mask = arr_28 > 45`. When a 1px thin stroke drawn on a $280\times 280$ canvas is resized to $20\times 20$ via bilinear interpolation, its peak pixel intensity is attenuated to $\approx 42.0$. As a result, `bin_mask` contains 0 pixels, and the heuristic defaults to `avg_w = 3.0`, incorrectly determining that the stroke is already thick enough and skipping morphological dilation.
- **Automated Test**: `tests/test_preprocessing.py::test_PP_08_auto_thicken_under_2_2px`
- **Recommended Fix**: Change line 253 to `bin_mask = arr_28 > 25`.

---

### BUG-07: Dense Texture Noise Bypass in Structural Validation (`src/validation.py`)
- **Severity**: Low (Adversarial Robustness)
- **Root Cause**: Gate 2 requires `fill_ratio > 0.82 AND ink_count > 4000` to reject solid blocks. Synthetic dense noise images (e.g. 15%–30% fill spread across the full canvas) merge into a single large connected component under $5\times 5$ morphological dilation. Consequently, they pass Gate 2 (`fill < 0.82`) and Gate 3 (`components == 1`), reaching the classifier where they must be rejected by Gate 4 (OOD confidence floor < 16%).
- **Automated Test**: `tests/test_validation.py::test_VA_08_photo_like_noisy_image_rejected`
- **Recommended Fix**: Add a bounding-box dispersion check in Gate 2 (`total_ink_pixels > 2500 AND bbox_area > 0.7 * canvas_area`).

