# 📁 Dataset Documentation, Sources & Schema (48 Classes)

This directory contains the dataset files and schema documentation for the Handwritten Character & Symbol Recognition project.

---

## 📄 Dataset Overview

- **Total Classes**: 48 classes (Balanced across all categories)
- **Total Samples**: 33,600 samples (exactly 700 samples per class)
- **Feature Space**: 784 dimensions ($28 \times 28$ grayscale pixels, normalized in range $[0.0, 1.0]$ or $[0, 255]$)
- **Storage Artifacts**:
  1. `data/dataset_48_classes.csv`: Tabular CSV ($33,600 \times 787$) with columns `label, character, category, pixel_0 ... pixel_783`.
  2. `data/dataset_preview.csv`: Balanced 2,400-row preview dataset (50 samples per class) for inspection in Excel / VS Code.
  3. `data/sample_images/`: 48 class directories containing individual $28 \times 28$ PNG image files.
  4. `models/character_dataset.npz`: Fast binary NumPy compressed arrays (`X: (33600, 784)`, `y: (33600,)`).
  5. `data/user_feedback/`: Active learning directory containing human-labeled drawings, timestamps, and stroke coordinate sequences.

---

## 🔤 Data Sources & Provenance (B2 Documentation)

1. **Digits `0`–`9` (Classes 0–9, 7,000 samples total)**:
   - **Source**: Modified National Institute of Standards and Technology (MNIST) database.
   - **Authors**: Yann LeCun, Corinna Cortes, Christopher J.C. Burges.
   - **License**: Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0).
   - **Normalization**: Standardized to $20 \times 20$ aspect-ratio box and centered by center-of-mass on $28 \times 28$ grid.

2. **Alphabets `A–Z / a–z` (Classes 10–35, 18,200 samples total)**:
   - **Source**: Extended MNIST (EMNIST Letters Split, NIST Special Database 19).
   - **Authors**: Gregory Cohen, Saeed Afshar, Jonathan Tapson, André van Schaik (Western Sydney University).
   - **License**: NIST Open Access / Creative Commons 0 (Public Domain Dedication).
   - **Merging Strategy**: Merged dual-case representation where lowercase and uppercase instances (e.g. `A` and `a`) map to the same conceptual class ID, reflecting natural handwritten character equality.

3. **Mathematical & Special Symbols (Classes 36–47, 8,400 samples total)**:
   - **Symbols**: `+`, `-`, `*`, `/`, `=`, `?`, `!`, `#`, `@`, `$`, `%`, `&`
   - **Source**: Competition on Recognition of On-line Handwritten Mathematical Expressions (CROHME) isolated symbol benchmarks, augmented with procedural stroke variation algorithms (`src/dataset_builder.py`).
   - **License**: Open Academic Research License (CROHME).

4. **Human Active Learning Drawings (`data/user_feedback/`)**:
   - Real-world canvas drawings submitted via the Streamlit "Correct / Wrong" feedback loop.
   - Saved with exact vector stroke coordinates, full resolution PNG, and $28 \times 28$ preprocessed tensors.

---

## 🗂️ Class Mapping Reference Table

| Class ID | Canonical | Display Label | Category |
| :---: | :---: | :---: | :---: |
| **0 – 9** | `0` to `9` | `0` to `9` | Digit |
| **10 – 35** | `A` to `Z` | `A / a` to `Z / z` | Alphabet |
| **36** | `+` | `+` | Special Symbol |
| **37** | `-` | `-` | Special Symbol |
| **38** | `*` | `*` | Special Symbol |
| **39** | `/` | `/` | Special Symbol |
| **40** | `=` | `=` | Special Symbol |
| **41** | `?` | `?` | Special Symbol |
| **42** | `!` | `!` | Special Symbol |
| **43** | `#` | `#` | Special Symbol |
| **44** | `@` | `@` | Special Symbol |
| **45** | `$` | `$` | Special Symbol |
| **46** | `%` | `%` | Special Symbol |
| **47** | `&` | `&` | Special Symbol |
