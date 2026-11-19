# Project Requirements & Environment Specification

This document outlines the software dependencies, system prerequisites, and setup instructions for the **Handwritten Character & Symbol Recognition System (Heterogeneous Ensemble)**.

---

## 1. System Requirements

- **Operating System**: Windows 10/11, Linux (Ubuntu 20.04+ recommended), or macOS (12+)
- **Python**: Python 3.10 – 3.13 (64-bit recommended)
- **RAM**: Minimum 8 GB (16 GB recommended for full training runs)
- **Disk Space**: ~2 GB (including virtual environment, dependencies, and model files)
- **Hardware Acceleration**: CPU-only execution (No dedicated GPU required)

---

## 2. Core Python Dependencies

| Package | Version | Purpose |
|---|---|---|
| `streamlit` | `1.65.0` | Interactive web dashboard and UI framework |
| `scikit-learn` | `1.9.1` | Logistic Regression, Random Forest, MLP, Ensemble Voting |
| `scipy` | `1.18.1` | Scientific computing and optimization routines |
| `numpy` | `2.5.3` | Numerical matrix and tensor operations |
| `pandas` | `3.0.6` | Data wrangling and preview generation |
| `pillow` | `12.3.0` | Image processing, resizing, format conversion |
| `matplotlib` | `3.11.2` | Visualization and confusion matrix rendering |
| `joblib` | `1.6.0` | Compressed model serialization & deserialization |
| `altair` | `6.3.0` | Interactive chart and confidence visualizer |
| `uvicorn` | `0.54.0` | ASGI web server (API endpoints / microservices) |
| `websockets` | `17.2` | WebSocket transport support |

---

## 3. UI Component Notice

- **Canvas Component**: An optimized, internal HTML5 Canvas component is integrated under `src/components/canvas/` to provide high-performance, real-time touch, stylus, and mouse stroke rendering with full compatibility for Streamlit `1.30+` up to `1.65+`.

---

## 4. Setup & Installation Guide

### Step 1: Clone Repository
```bash
git clone https://github.com/Omkarbhure/ensemble-learning.git
cd ensemble-learning/handwritten-digit-ensemble
```

### Step 2: Create and Activate Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Packages
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
streamlit run src/app.py
```
The application will launch locally at `http://localhost:8501`.

---

## 5. Development & Testing Requirements

To run the automated test suite and regression benchmarks:

```bash
pip install pytest pytest-cov
pytest tests/ -v
```
