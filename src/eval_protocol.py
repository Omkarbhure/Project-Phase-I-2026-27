import os
import sys
import json
import time
import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
from scipy.stats import chi2, binomtest

# Path setup
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
DATASET_PATH = os.path.join(MODELS_DIR, "character_dataset.npz")
PRODUCTION_MODEL_PATH = os.path.join(MODELS_DIR, "character_ensemble.pkl")
BASELINE_REPORT_PATH = os.path.join(MODELS_DIR, "baseline_report.json")
BENCHMARK_METRICS_PATH = os.path.join(MODELS_DIR, "benchmark_metrics.json")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import LABEL_MAP, CANONICAL_MAP, get_character_category
from features import transform_dataset_features


def get_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """Loads the canonical 48-class dataset."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    data = np.load(DATASET_PATH)
    return data["X"], data["y"]


def get_stratified_split(
    X: Optional[np.ndarray] = None,
    y: Optional[np.ndarray] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns the standard stratified train/test split.
    At 33,600 samples, test_size=0.2 gives exactly 26,880 train and 6,720 test (140/class).
    """
    if X is None or y is None:
        X, y = get_dataset()
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def create_production_estimators(random_state: int = 42) -> Dict[str, Any]:
    """
    Creates fresh instances of the three production base models
    with the exact hyperparameters used in the deployed system.
    """
    lr = LogisticRegression(
        max_iter=200,
        solver="lbfgs",
        random_state=random_state,
    )
    rf = RandomForestClassifier(
        n_estimators=120,
        max_depth=30,
        random_state=random_state,
        n_jobs=-1,
    )
    mlp = MLPClassifier(
        hidden_layer_sizes=(384, 192),
        activation="relu",
        solver="adam",
        max_iter=75,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=random_state,
    )
    return {"lr": lr, "rf": rf, "mlp": mlp}


def create_production_ensemble(
    estimators: Optional[Dict[str, Any]] = None,
    random_state: int = 42,
) -> VotingClassifier:
    """Creates a fresh soft-voting ensemble with production weights (1:3:5)."""
    if estimators is None:
        estimators = create_production_estimators(random_state=random_state)
    return VotingClassifier(
        estimators=[
            ("lr", estimators["lr"]),
            ("rf", estimators["rf"]),
            ("mlp", estimators["mlp"]),
        ],
        voting="soft",
        weights=[1, 3, 5],
        n_jobs=1,
    )


def evaluate_model(
    clf: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "",
) -> Tuple[Dict[str, Any], np.ndarray]:
    """
    Evaluates a single fitted model or ensemble on X_test, y_test.
    Computes accuracy, weighted precision/recall/F1, macro precision/recall/F1,
    and per-sample inference latency.
    """
    t0 = time.perf_counter()
    y_pred = clf.predict(X_test)
    elapsed = time.perf_counter() - t0
    latency_ms = (elapsed / len(X_test)) * 1000.0

    acc = float(accuracy_score(y_test, y_pred))
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "name": model_name or type(clf).__name__,
        "accuracy": round(acc * 100.0, 3),
        "precision_weighted": round(float(p_w) * 100.0, 3),
        "recall_weighted": round(float(r_w) * 100.0, 3),
        "f1_weighted": round(float(f1_w) * 100.0, 3),
        "precision_macro": round(float(p_m) * 100.0, 3),
        "recall_macro": round(float(r_m) * 100.0, 3),
        "f1_macro": round(float(f1_m) * 100.0, 3),
        "latency_ms_per_sample": round(latency_ms, 4),
        "total_inference_time_sec": round(elapsed, 3),
    }
    return metrics, y_pred


def compute_category_breakdown(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, Dict[str, Any]]:
    """Breaks down accuracy and F1 across Digits, Alphabets, and Special Symbols."""
    digit_mask = y_true < 10
    letter_mask = (y_true >= 10) & (y_true < 36)
    symbol_mask = y_true >= 36

    categories = {}
    for name, mask in [
        ("Digits (0-9)", digit_mask),
        ("Alphabets (A-Z / a-z)", letter_mask),
        ("Special Symbols", symbol_mask),
    ]:
        if np.any(mask):
            y_t = y_true[mask]
            y_p = y_pred[mask]
            acc = float(accuracy_score(y_t, y_p))
            p_w, r_w, f1_w, _ = precision_recall_fscore_support(
                y_t, y_p, average="weighted", zero_division=0
            )
            categories[name] = {
                "sample_count": int(np.sum(mask)),
                "accuracy": round(acc * 100.0, 2),
                "precision": round(float(p_w) * 100.0, 2),
                "recall": round(float(r_w) * 100.0, 2),
                "f1_score": round(float(f1_w) * 100.0, 2),
            }
    return categories


def compute_top_confusions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    top_n: int = 15,
) -> List[Dict[str, Any]]:
    """Identifies the top N confused pairs (true -> predicted) in the predictions."""
    num_classes = max(int(np.max(y_true)), int(np.max(y_pred))) + 1
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    np.fill_diagonal(cm, 0)

    top_pairs = []
    flat_indices = np.argsort(cm.ravel())[::-1]
    for idx in flat_indices:
        t_idx, p_idx = divmod(int(idx), num_classes)
        count = int(cm[t_idx, p_idx])
        if count <= 0:
            break
        top_pairs.append({
            "true_idx": t_idx,
            "pred_idx": p_idx,
            "true_label": LABEL_MAP.get(t_idx, str(t_idx)),
            "predicted_label": LABEL_MAP.get(p_idx, str(p_idx)),
            "count": count,
        })
        if len(top_pairs) >= top_n:
            break
    return top_pairs


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    name_a: str = "Ensemble",
    name_b: str = "MLP",
) -> Dict[str, Any]:
    """
    Performs Edwards continuity-corrected McNemar test and exact 2-sided binomial test
    between two models evaluated on the exact same test instances.
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    n11 = int(np.sum(correct_a & correct_b))
    n00 = int(np.sum((~correct_a) & (~correct_b)))
    n10 = int(np.sum(correct_a & (~correct_b)))  # Model A correct, B wrong
    n01 = int(np.sum((~correct_a) & correct_b))  # Model A wrong, B correct
    total_discordant = n10 + n01

    if total_discordant > 0:
        chi2_stat = ((abs(n10 - n01) - 1.0) ** 2) / float(total_discordant)
        p_chi2 = float(chi2.sf(chi2_stat, 1))
        res_binom = binomtest(min(n10, n01), total_discordant, 0.5, alternative="two-sided")
        p_exact = float(res_binom.pvalue)
    else:
        chi2_stat = 0.0
        p_chi2 = 1.0
        p_exact = 1.0

    return {
        "model_a": name_a,
        "model_b": name_b,
        "contingency_table": {
            "both_correct_n11": n11,
            "both_wrong_n00": n00,
            f"{name_a}_correct_{name_b}_wrong_n10": n10,
            f"{name_a}_wrong_{name_b}_correct_n01": n01,
        },
        "total_discordant": total_discordant,
        "net_gain": n10 - n01,
        "chi2_continuity_corrected": round(chi2_stat, 4),
        "p_value_chi2": round(p_chi2, 4),
        "p_value_exact_binomial": round(p_exact, 4),
        "statistically_significant_05": p_exact < 0.05,
    }


def sync_benchmark_metrics_from_baseline() -> Dict[str, Any]:
    """
    Generates/synchronizes models/benchmark_metrics.json directly from
    models/baseline_report.json so there is only one source of truth.
    Converts metrics into the schema expected by the Streamlit UI.
    """
    if not os.path.exists(BASELINE_REPORT_PATH):
        raise FileNotFoundError(f"Baseline report not found at {BASELINE_REPORT_PATH}")

    with open(BASELINE_REPORT_PATH, "r", encoding="utf-8") as f:
        baseline = json.load(f)

    meta = baseline.get("metadata", {})
    models_raw = baseline.get("models", {})
    categories_raw = baseline.get("category_summary", {})
    confused_raw = baseline.get("top_confused_pairs", [])

    # Map models to benchmark_metrics.json schema
    models_out = {}
    for k in ["lr", "rf", "mlp", "ensemble"]:
        if k in models_raw:
            m = models_raw[k]
            models_out[k] = {
                "name": m.get("name", k),
                "accuracy": round(m.get("accuracy", 0.0), 2),
                "precision": round(m.get("precision_weighted", 0.0), 2),
                "recall": round(m.get("recall_weighted", 0.0), 2),
                "f1_score": round(m.get("f1_weighted", 0.0), 2),
                "latency_ms_per_sample": round(m.get("latency_ms_per_sample", 0.0), 3),
            }

    # Map categories
    cats_out = {}
    cat_mapping = {
        "Digits (0-9)": "Digits (0–9)",
        "Alphabets (A-Z / a-z)": "Alphabets (A–Z / a–z)",
        "Special Symbols": "Special Symbols",
    }
    for orig_name, target_name in cat_mapping.items():
        if orig_name in categories_raw:
            c = categories_raw[orig_name]
            cats_out[target_name] = {
                "sample_count": c.get("sample_count", 0),
                "accuracy": round(c.get("accuracy", 0.0), 2),
            }

    # Map top confused pairs
    confused_out = []
    for pair in confused_raw[:8]:
        confused_out.append({
            "true_label": pair.get("true_label", ""),
            "predicted_label": pair.get("predicted_label", ""),
            "count": pair.get("count", 0),
        })

    payload = {
        "timestamp": meta.get("evaluated_at", time.strftime("%Y-%m-%d %H:%M:%S")),
        "source": "Synced from models/baseline_report.json (Single Source of Truth)",
        "total_eval_samples": meta.get("test_samples", 6720),
        "total_classes": meta.get("num_classes", 48),
        "models": models_out,
        "categories": cats_out,
        "top_confused_pairs": confused_out,
    }

    with open(BENCHMARK_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return payload


def evaluate_production_system() -> Dict[str, Any]:
    """
    Evaluates the currently deployed production model across the full 6,720 held-out test split
    and writes models/baseline_report.json, then syncs models/benchmark_metrics.json.
    """
    import joblib

    if not os.path.exists(PRODUCTION_MODEL_PATH):
        raise FileNotFoundError(f"Production model not found at {PRODUCTION_MODEL_PATH}")

    model = joblib.load(PRODUCTION_MODEL_PATH)
    _, X_test, _, y_test = get_stratified_split(test_size=0.2, random_state=42)

    estimators = getattr(model, "named_estimators_", {})
    eval_targets = {
        "lr": ("Logistic Regression", estimators.get("lr")),
        "rf": ("Random Forest (120 Trees)", estimators.get("rf")),
        "mlp": ("Deep Neural Net (MLP 384x192)", estimators.get("mlp")),
        "ensemble": ("Soft Voting Ensemble (1:3:5)", model),
    }

    models_results = {}
    preds = {}

    for key, (label_name, clf) in eval_targets.items():
        if clf is not None:
            m, p = evaluate_model(clf, X_test, y_test, model_name=label_name)
            models_results[key] = m
            preds[key] = p

    cats = compute_category_breakdown(y_test, preds["ensemble"])
    confused = compute_top_confusions(y_test, preds["ensemble"], top_n=15)
    mcnemar = mcnemar_test(
        y_test,
        preds["ensemble"],
        preds["mlp"],
        name_a="Ensemble",
        name_b="MLP",
    )

    report = {
        "metadata": {
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "dataset_file": os.path.basename(DATASET_PATH),
            "total_samples": 33600,
            "test_samples": len(y_test),
            "num_classes": 48,
            "random_state": 42,
            "test_size": 0.2,
        },
        "models": models_results,
        "mcnemar_ensemble_vs_mlp": mcnemar,
        "category_summary": cats,
        "top_confused_pairs": confused,
    }

    with open(BASELINE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    # Automatically sync benchmark_metrics.json
    sync_benchmark_metrics_from_baseline()

    return report

