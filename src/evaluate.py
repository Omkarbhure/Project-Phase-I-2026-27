import os
import sys
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)

from predict import load_model, LABEL_MAP, CANONICAL_MAP, get_character_category
from dataset_builder import DATASET_PATH
from eval_protocol import (
    get_dataset,
    get_stratified_split,
    evaluate_model,
    compute_category_breakdown,
    compute_top_confusions,
    mcnemar_test,
    sync_benchmark_metrics_from_baseline,
    MODELS_DIR,
    BASELINE_REPORT_PATH,
    BENCHMARK_METRICS_PATH,
)

CONFUSION_MATRIX_PATH = os.path.join(MODELS_DIR, "baseline_confusion_matrix.png")


def evaluate_baseline(test_size=0.2, random_state=42):
    print("=" * 75)
    print("       BASELINE EVALUATION REPORT FOR CURRENT ENSEMBLE MODEL")
    print("=" * 75)

    print(f"\n[1/5] Loading compiled dataset from '{os.path.basename(DATASET_PATH)}'...")
    X, y = get_dataset()
    num_samples = len(y)
    num_classes = len(np.unique(y))
    print(f"  * Total dataset size: {num_samples:,} samples | {num_classes} classes")

    print(f"\n[2/5] Creating stratified test split (test_size={test_size}, random_state={random_state})...")
    X_train, X_test, y_train, y_test = get_stratified_split(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"  * Training set: {len(y_train):,} samples")
    print(f"  * Held-out test set: {len(y_test):,} samples ({len(y_test) // num_classes} per class)")

    print("\n[3/5] Loading model artifact...")
    model = load_model()
    estimators = getattr(model, "named_estimators_", {})
    print(f"  * Loaded model: {type(model).__name__}")
    print(f"  * Base estimators: {list(estimators.keys())}")

    # Evaluate all estimators and ensemble
    models_to_eval = {}
    if "lr" in estimators:
        models_to_eval["lr"] = ("Logistic Regression", estimators["lr"])
    if "rf" in estimators:
        models_to_eval["rf"] = ("Random Forest (120 Trees)", estimators["rf"])
    if "mlp" in estimators:
        models_to_eval["mlp"] = ("Deep Neural Net (MLP 384x192)", estimators["mlp"])
    models_to_eval["ensemble"] = ("Soft Voting Ensemble (1:3:5)", model)

    results_models = {}
    predictions = {}

    print("\n[4/5] Running inference across held-out test set...")
    for key, (label_name, clf) in models_to_eval.items():
        m_dict, y_pred = evaluate_model(clf, X_test, y_test, model_name=label_name)
        results_models[key] = m_dict
        predictions[key] = y_pred

    # Display summary table
    print("\n" + "-" * 88)
    print(f"  {'Model':<32} | {'Accuracy':<10} | {'Prec (W)':<10} | {'Recall (W)':<10} | {'F1 (W)':<10} | {'Latency'}")
    print("-" * 88)
    for key, info in results_models.items():
        print(
            f"  {info['name']:<32} | {info['accuracy']:>6.2f}%    | "
            f"{info['precision_weighted']:>6.2f}%    | {info['recall_weighted']:>6.2f}%    | "
            f"{info['f1_weighted']:>6.2f}%    | {info['latency_ms_per_sample']:.3f} ms/samp"
        )
    print("-" * 88)

    # Detailed Ensemble Per-Class Metrics
    ens_pred = predictions["ensemble"]
    report_dict = classification_report(
        y_test, ens_pred, output_dict=True, zero_division=0
    )

    per_class_metrics = {}
    for class_id in range(num_classes):
        str_id = str(class_id)
        c_metrics = report_dict.get(str_id, {})
        canonical = CANONICAL_MAP.get(class_id, str(class_id))
        display_label = LABEL_MAP.get(class_id, str(class_id))
        cat = get_character_category(display_label)

        per_class_metrics[class_id] = {
            "class_id": class_id,
            "label": display_label,
            "canonical": canonical,
            "category": cat,
            "precision": round(c_metrics.get("precision", 0.0) * 100.0, 2),
            "recall": round(c_metrics.get("recall", 0.0) * 100.0, 2),
            "f1_score": round(c_metrics.get("f1-score", 0.0) * 100.0, 2),
            "support": int(c_metrics.get("support", 0)),
        }

    # Category breakdowns using shared eval protocol logic
    category_summary = compute_category_breakdown(y_test, ens_pred)
    print("\n  Category Breakdown (Ensemble):")
    for cat_name, data in category_summary.items():
        print(f"    - {cat_name:<24}: Acc: {data['accuracy']:>6.2f}% | Prec: {data['precision']:>6.2f}% | F1: {data['f1_score']:>6.2f}% ({data['sample_count']:,} samples)")

    # Top confused pairs
    top_15_confused = compute_top_confusions(y_test, ens_pred, top_n=15)
    print("\n  Top 15 Most Confused Pairs (Empirical Error Frequency):")
    print("  " + "-" * 72)
    print(f"  {'Rank':<5} | {'True Class':<18} | {'Mispredicted As':<18} | {'Errors':<8}")
    print("  " + "-" * 72)
    for rank, p in enumerate(top_15_confused, start=1):
        print(
            f"  #{rank:<4} | {p['true_label']:<18} | {p['predicted_label']:<18} | {p['count']:<8}"
        )
    print("  " + "-" * 72)

    # McNemar Significance Test (Ensemble vs MLP)
    mcnemar_res = mcnemar_test(
        y_test, ens_pred, predictions["mlp"], name_a="Ensemble", name_b="MLP"
    )
    print("\n  McNemar Significance Test (Ensemble vs. MLP):")
    print(f"    * Concordant Correct (Both Correct)     : {mcnemar_res['contingency_table']['both_correct_n11']}")
    print(f"    * Concordant Error   (Both Wrong)       : {mcnemar_res['contingency_table']['both_wrong_n00']}")
    print(f"    * Discordant: Ensemble Correct, MLP Wrong: {mcnemar_res['contingency_table']['Ensemble_correct_MLP_wrong_n10']}")
    print(f"    * Discordant: MLP Correct, Ensemble Wrong: {mcnemar_res['contingency_table']['Ensemble_wrong_MLP_correct_n01']}")
    print(f"    * Net Gain (Ensemble - MLP)             : +{mcnemar_res['net_gain']} samples")
    print(f"    * Edwards Chi-Squared (Continuity Corr) : {mcnemar_res['chi2_continuity_corrected']:.4f}")
    print(f"    * Chi-Squared p-value                   : {mcnemar_res['p_value_chi2']:.4f}")
    print(f"    * Exact Binomial p-value                : {mcnemar_res['p_value_exact_binomial']:.4f}")
    print(f"    * Statistically Significant (alpha=0.05): {mcnemar_res['statistically_significant_05']}")

    # Plot Confusion Matrix
    print(f"\n[5/5] Generating and saving confusion matrix plot to '{os.path.basename(CONFUSION_MATRIX_PATH)}'...")
    cm = confusion_matrix(y_test, ens_pred, labels=list(range(num_classes)))
    plt.figure(figsize=(18, 16), dpi=150)
    plt.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("Baseline Ensemble Confusion Matrix (48 Classes)", fontsize=16, pad=15)
    plt.colorbar(fraction=0.046, pad=0.04)

    class_names = [CANONICAL_MAP.get(i, str(i)) for i in range(num_classes)]
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, fontsize=8, rotation=45)
    plt.yticks(tick_marks, class_names, fontsize=8)
    plt.xlabel("Predicted Class", fontsize=12, labelpad=10)
    plt.ylabel("True Class", fontsize=12, labelpad=10)
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH, bbox_inches="tight")
    plt.close()
    print("  * Confusion matrix PNG successfully saved.")

    # Save Baseline Report JSON
    baseline_report = {
        "metadata": {
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "dataset_file": os.path.basename(DATASET_PATH),
            "total_samples": num_samples,
            "test_samples": len(y_test),
            "num_classes": num_classes,
            "random_state": random_state,
            "test_size": test_size,
        },
        "models": results_models,
        "mcnemar_ensemble_vs_mlp": mcnemar_res,
        "category_summary": category_summary,
        "top_confused_pairs": top_15_confused,
        "per_class_metrics": per_class_metrics,
    }

    with open(BASELINE_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(baseline_report, f, indent=2)
    print(f"  * Baseline JSON report saved to '{os.path.basename(BASELINE_REPORT_PATH)}'.")

    # Sync benchmark_metrics.json directly from baseline_report.json
    sync_benchmark_metrics_from_baseline()
    print("  * Synced 'models/benchmark_metrics.json' with single source of truth.")

    print("\n" + "=" * 75)
    print("                      BASELINE EVALUATION COMPLETE")
    print("=" * 75)

    return baseline_report


if __name__ == "__main__":
    evaluate_baseline()
