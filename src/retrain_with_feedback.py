import os
import sys
import csv
import time
import joblib
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

from dataset_builder import DATASET_PATH
from predict import load_model, CHARACTER_MODEL_PATH
from feedback_manager import FEEDBACK_LOG_CSV, BASE_DIR


def retrain_ensemble_with_feedback(feedback_weight: float = 4.0, random_state: int = 42):
    print("=" * 70)
    print("      RETRAINING ENSEMBLE MODEL WITH USER FEEDBACK SAMPLES")
    print("=" * 70)

    # 1. Load original dataset
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Base dataset not found at {DATASET_PATH}")

    base_data = np.load(DATASET_PATH)
    X_base = base_data["X"].astype(np.float32)
    y_base = base_data["y"].astype(np.int64)
    weights_base = np.ones(len(y_base), dtype=np.float32)
    print(f"\n[1/4] Loaded base dataset: {len(y_base):,} samples across {len(np.unique(y_base))} classes.")

    # 2. Load user feedback samples
    X_feed = []
    y_feed = []
    feed_count = 0

    if os.path.exists(FEEDBACK_LOG_CSV):
        with open(FEEDBACK_LOG_CSV, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                prep_rel = row.get("preprocessed_img_path", "")
                full_prep_path = os.path.join(BASE_DIR, prep_rel) if not os.path.isabs(prep_rel) else prep_rel
                
                if os.path.exists(full_prep_path):
                    try:
                        img = Image.open(full_prep_path).convert("L")
                        arr = np.array(img, dtype=np.float32) / 255.0
                        if arr.shape == (28, 28):
                            X_feed.append(arr.flatten())
                            y_feed.append(int(row["class_id"]))
                            feed_count += 1
                    except Exception as e:
                        pass

    print(f"[2/4] Discovered {feed_count} validated user feedback drawings.")

    if feed_count > 0:
        X_feed_arr = np.array(X_feed, dtype=np.float32)
        y_feed_arr = np.array(y_feed, dtype=np.int64)
        weights_feed = np.full(len(y_feed_arr), feedback_weight, dtype=np.float32)

        X_merged = np.vstack([X_base, X_feed_arr])
        y_merged = np.concatenate([y_base, y_feed_arr])
        weights_merged = np.concatenate([weights_base, weights_feed])
        print(f"  * Merged training pool: {len(y_merged):,} samples (Feedback sample weight: {feedback_weight}x)")
    else:
        print("  * No user feedback drawings available yet. Retraining on base dataset.")
        X_merged = X_base
        y_merged = y_base
        weights_merged = weights_base

    # 3. Train / Validation split
    print("\n[3/4] Creating stratified validation split (80/20)...")
    X_train, X_val, y_train, y_val, w_train, _ = train_test_split(
        X_merged, y_merged, weights_merged, test_size=0.2, random_state=random_state, stratify=y_merged
    )

    # 4. Fit Calibrated Voting Ensemble
    print("\n[4/4] Fitting base models and calibrated voting ensemble...")
    lr = LogisticRegression(max_iter=250, solver="lbfgs", random_state=random_state)
    rf = RandomForestClassifier(n_estimators=120, max_depth=30, random_state=random_state, n_jobs=-1)
    mlp = MLPClassifier(
        hidden_layer_sizes=(384, 192),
        activation="relu",
        solver="adam",
        max_iter=75,
        early_stopping=True,
        n_iter_no_change=8,
        random_state=random_state,
    )

    ensemble = VotingClassifier(
        estimators=[("lr", lr), ("rf", rf), ("mlp", mlp)],
        voting="soft",
        weights=[1, 3, 5],
        n_jobs=1,
    )
    print("  * Fitting Weighted Soft Voting Ensemble...")
    sys.stdout.flush()
    ensemble.fit(X_train, y_train, sample_weight=w_train)

    # Evaluate
    val_pred = ensemble.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred) * 100.0
    print(f"\n  ⭐ Validation Accuracy after retraining: {val_acc:.2f}%")

    # Save to model artifact path
    joblib.dump(ensemble, CHARACTER_MODEL_PATH, compress=3)
    print(f"  * Updated model artifact saved to: {os.path.basename(CHARACTER_MODEL_PATH)}")

    print("\n" + "=" * 70)
    print("           FEEDBACK RETRAINING COMPLETE")
    print("=" * 70)
    return val_acc, feed_count


if __name__ == "__main__":
    retrain_ensemble_with_feedback()
