import os
import sys
import time
import json
import numpy as np
from typing import Dict, Any, List

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
MODELS_DIR = os.path.join(PROJECT_DIR, "models")
BENCHMARK_REPORT_PATH = os.path.join(MODELS_DIR, "hog_feature_benchmark.json")

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from eval_protocol import (
    get_dataset,
    get_stratified_split,
    create_production_estimators,
    create_production_ensemble,
    evaluate_model,
)
from features import transform_dataset_features


def run_feature_benchmark(random_state: int = 42) -> List[Dict[str, Any]]:
    print("=" * 85)
    print("      RIGOROUS HOG FEATURE BENCHMARK (PRODUCTION MODEL CONFIGURATIONS)")
    print("      Features Tested: Raw Pixels (784) | HOG (324) | Raw + HOG (1108)")
    print("=" * 85)

    print("\n[1/4] Loading compiled dataset (33,600 samples, 48 classes)...")
    X_raw, y = get_dataset()

    print(f"[2/4] Creating stratified train/test split (test_size=0.2, random_state={random_state})...")
    X_train_raw, X_test_raw, y_train, y_test = get_stratified_split(
        X_raw, y, test_size=0.2, random_state=random_state
    )
    print(f"  * Training set: {len(y_train):,} samples")
    print(f"  * Testing set : {len(y_test):,} samples")

    feature_modes = ["raw", "hog", "raw_plus_hog"]
    feature_data = {}

    print("\n[3/4] Transforming feature representations...")
    for mode in feature_modes:
        t0 = time.time()
        print(f"  * Computing '{mode}' features for train & test splits...")
        X_tr = transform_dataset_features(X_train_raw, mode=mode)
        X_te = transform_dataset_features(X_test_raw, mode=mode)
        dur = time.time() - t0
        print(f"    -> Mode '{mode}': train shape {X_tr.shape}, test shape {X_te.shape} ({dur:.1f}s)")
        feature_data[mode] = (X_tr, X_te)

    results = []

    print("\n[4/4] Training & evaluating production models across feature representations...")
    for mode in feature_modes:
        X_tr, X_te = feature_data[mode]
        n_features = X_tr.shape[1]
        print(f"\n==================== FEATURE SET: {mode.upper()} ({n_features} dims) ====================")

        # 1. Base estimators
        estimators = create_production_estimators(random_state=random_state)
        fitted_estimators = {}

        for key, name in [
            ("lr", "Logistic Regression"),
            ("rf", "Random Forest (120 Trees)"),
            ("mlp", "Deep Neural Net (MLP 384x192)"),
        ]:
            clf = estimators[key]
            t_fit = time.time()
            clf.fit(X_tr, y_train)
            fit_time = time.time() - t_fit

            m, _ = evaluate_model(clf, X_te, y_test, model_name=name)
            fitted_estimators[key] = clf
            
            entry = {
                "feature_mode": mode,
                "n_features": n_features,
                "model_key": key,
                "model_name": name,
                "accuracy": m["accuracy"],
                "precision_weighted": m["precision_weighted"],
                "recall_weighted": m["recall_weighted"],
                "f1_weighted": m["f1_weighted"],
                "train_time_sec": round(fit_time, 2),
                "latency_ms_per_sample": m["latency_ms_per_sample"],
            }
            results.append(entry)
            print(f"  * {name:<28}: Acc = {m['accuracy']:>6.2f}% | F1 = {m['f1_weighted']:>6.2f}% | Fit: {fit_time:.1f}s | Lat: {m['latency_ms_per_sample']:.3f} ms")

        # 2. Soft Voting Ensemble
        ens = create_production_ensemble(estimators=fitted_estimators, random_state=random_state)
        t_fit_ens = time.time()
        ens.fit(X_tr, y_train)
        fit_time_ens = time.time() - t_fit_ens

        m_ens, _ = evaluate_model(ens, X_te, y_test, model_name="Soft Voting Ensemble (1:3:5)")
        ens_entry = {
            "feature_mode": mode,
            "n_features": n_features,
            "model_key": "ensemble",
            "model_name": "Soft Voting Ensemble (1:3:5)",
            "accuracy": m_ens["accuracy"],
            "precision_weighted": m_ens["precision_weighted"],
            "recall_weighted": m_ens["recall_weighted"],
            "f1_weighted": m_ens["f1_weighted"],
            "train_time_sec": round(fit_time_ens, 2),
            "latency_ms_per_sample": m_ens["latency_ms_per_sample"],
        }
        results.append(ens_entry)
        print(f"  * {'Soft Voting Ensemble (1:3:5)':<28}: Acc = {m_ens['accuracy']:>6.2f}% | F1 = {m_ens['f1_weighted']:>6.2f}% | Fit: {fit_time_ens:.1f}s | Lat: {m_ens['latency_ms_per_sample']:.3f} ms")

    # Save benchmark report to models/hog_feature_benchmark.json
    with open(BENCHMARK_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Display comparison table
    print("\n" + "=" * 92)
    print(f"  {'Model':<30} | {'Features':<12} | {'Dims':<5} | {'Accuracy':<10} | {'F1-Score':<10} | {'Train (s)'} | {'Latency'}")
    print("-" * 92)
    for r in results:
        print(
            f"  {r['model_name']:<30} | {r['feature_mode']:<12} | {r['n_features']:<5} | "
            f"{r['accuracy']:>6.2f}%    | {r['f1_weighted']:>6.2f}%    | {r['train_time_sec']:>7.1f}s  | {r['latency_ms_per_sample']:.3f} ms"
        )
    print("=" * 92)

    return results


if __name__ == "__main__":
    run_feature_benchmark()
