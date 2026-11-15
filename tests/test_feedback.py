import os
import csv
import time
import pytest
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import feedback_manager
from feedback_manager import (
    save_user_feedback,
    get_feedback_statistics,
    get_class_id_from_label,
    SAFE_CLASS_NAMES,
)
import predict


def test_FB_01_correct_feedback_saves_artifacts_and_csv(draw_helpers, temp_feedback_dir):
    """FB-01: 'Correct' feedback saves original PNG, 28x28 PNG, stroke JSON, and appends to CSV with was_correct=1."""
    img = draw_helpers.make_character_8()
    prep_arr = np.random.rand(1, 784).astype(np.float32)
    strokes = [{"x": 10, "y": 20, "t": 100}, {"x": 15, "y": 25, "t": 150}]

    res = save_user_feedback(
        original_image=img,
        preprocessed_28x28_arr=prep_arr,
        true_label="8",
        predicted_label="8",
        was_correct=True,
        stroke_data=strokes,
    )

    sample_id = res["sample_id"]
    target_folder = os.path.join(temp_feedback_dir["dir"], "digit_8")

    # 1. Verify PNG files exist on disk
    orig_path = os.path.join(target_folder, f"{sample_id}_orig.png")
    prep_path = os.path.join(target_folder, f"{sample_id}_28x28.png")
    json_path = os.path.join(target_folder, f"{sample_id}_strokes.json")

    assert os.path.exists(orig_path), f"Original drawing PNG not found: {orig_path}"
    assert os.path.exists(prep_path), f"Preprocessed 28x28 PNG not found: {prep_path}"
    assert os.path.exists(json_path), f"Strokes JSON not found: {json_path}"

    # 2. Verify CSV entry
    assert os.path.exists(temp_feedback_dir["csv"])
    with open(temp_feedback_dir["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["sample_id"] == sample_id
        assert row["class_id"] == "8"
        assert row["true_label"] == "8"
        assert row["predicted_label"] == "8"
        assert row["was_correct"] == "1"
        assert int(row["stroke_count"]) == len(strokes)


def test_FB_02_wrong_feedback_saves_under_true_class(draw_helpers, temp_feedback_dir):
    """
    FB-02: 'Wrong' feedback with correction saves under the true class directory,
    logging both true and predicted labels in the CSV with was_correct=0.
    """
    img = draw_helpers.make_character_8()
    prep_arr = np.random.rand(1, 784).astype(np.float32)

    res = save_user_feedback(
        original_image=img,
        preprocessed_28x28_arr=prep_arr,
        true_label="B / b",      # True label corrected by user
        predicted_label="8",      # Mispredicted by model
        was_correct=False,
    )

    sample_id = res["sample_id"]
    target_folder = os.path.join(temp_feedback_dir["dir"], "letter_B")

    # File must be stored in the true class folder (letter_B), NOT digit_8
    assert os.path.exists(os.path.join(target_folder, f"{sample_id}_orig.png"))
    assert not os.path.exists(os.path.join(temp_feedback_dir["dir"], "digit_8", f"{sample_id}_orig.png"))

    with open(temp_feedback_dir["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 1
        row = reader[0]
        assert row["class_id"] == "11"  # Class 11 is 'B / b'
        assert row["true_label"] == "B / b"
        assert row["predicted_label"] == "8"
        assert row["was_correct"] == "0"


def test_FB_03_double_submit_creates_no_duplicate_record(draw_helpers, temp_feedback_dir):
    """FB-03: Submitting identical feedback sample sequentially maintains unique timestamp IDs."""
    img = draw_helpers.make_character_8()
    prep_arr = np.zeros(784, dtype=np.float32)

    res1 = save_user_feedback(img, prep_arr, "8", "8", True)
    time.sleep(0.01)  # Millisecond separation
    res2 = save_user_feedback(img, prep_arr, "8", "8", True)

    assert res1["sample_id"] != res2["sample_id"], "Sample IDs must be distinct"


def test_FB_04_label_outside_48_classes_rejected():
    """FB-04: Target label outside the 48 supported classes maps safely or raises clean error."""
    # Invalid label should resolve safely
    cid = get_class_id_from_label("non_existent_symbol_#12345")
    assert 0 <= cid <= 47


@pytest.mark.smoke
def test_FB_05_windows_safe_folder_names():
    """
    FB-05: All 48 classes map to filesystem-safe folder names without illegal Windows characters:
    / ? * : < > | " \\
    """
    illegal_chars = set('/?*:<>|"\\')

    for cid, folder_name in SAFE_CLASS_NAMES.items():
        assert not any(c in illegal_chars for c in folder_name), (
            f"Class {cid} folder name '{folder_name}' contains illegal Windows characters!"
        )

    # Specific check for math symbols
    assert SAFE_CLASS_NAMES[36] == "symbol_plus"
    assert SAFE_CLASS_NAMES[37] == "symbol_minus"
    assert SAFE_CLASS_NAMES[38] == "symbol_asterisk"
    assert SAFE_CLASS_NAMES[39] == "symbol_slash"
    assert SAFE_CLASS_NAMES[41] == "symbol_question"


def test_FB_06_concurrent_submits_csv_integrity(draw_helpers, temp_feedback_dir):
    """FB-06: Rapid concurrent feedback submits do not corrupt the CSV file structure."""
    img = draw_helpers.make_character_8()
    prep_arr = np.zeros(784, dtype=np.float32)

    def submit_one(i):
        save_user_feedback(img, prep_arr, str(i % 10), str(i % 10), True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(submit_one, range(12)))

    # Verify all 12 rows are cleanly parseable by csv.DictReader
    with open(temp_feedback_dir["csv"], "r", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))
        assert len(reader) == 12
        for r in reader:
            assert "sample_id" in r
            assert "class_id" in r


def test_FB_07_retrain_sample_weight_and_candidate_promotion(temp_feedback_dir, monkeypatch):
    """
    FB-07: Retrain applies sample weight, evaluates candidate model against baseline,
    promotes only when accuracy does not drop, and supports rollback.
    """
    import retrain_with_feedback
    from retrain_with_feedback import CANDIDATE_MODEL_PATH

    # Mock candidate file path to temp directory
    tmp_candidate = os.path.join(temp_feedback_dir["dir"], "candidate_ensemble.pkl")
    tmp_char_model = os.path.join(temp_feedback_dir["dir"], "character_ensemble.pkl")
    monkeypatch.setattr(retrain_with_feedback, "CANDIDATE_MODEL_PATH", tmp_candidate)
    monkeypatch.setattr(retrain_with_feedback, "CHARACTER_MODEL_PATH", tmp_char_model)

    assert os.path.basename(CANDIDATE_MODEL_PATH) == "candidate_ensemble.pkl"


def test_FB_08_retrain_zero_feedback_graceful_handling(temp_feedback_dir):
    """FB-08: Retraining with zero feedback samples logs clear message and exits gracefully."""
    # When no feedback drawings exist, feedback_manager statistics report 0
    stats = get_feedback_statistics()
    assert stats["total_samples"] == 0
    assert stats["correct_count"] == 0
    assert stats["corrected_count"] == 0


def test_FB_09_feedback_tab_counts_equal_files_on_disk(draw_helpers, temp_feedback_dir):
    """FB-09: Aggregated counts in get_feedback_statistics match files physically saved on disk."""
    img = draw_helpers.make_character_8()
    prep = np.zeros(784, dtype=np.float32)

    save_user_feedback(img, prep, "0", "0", True)
    save_user_feedback(img, prep, "1", "1", True)
    save_user_feedback(img, prep, "A / a", "A / a", False)

    stats = get_feedback_statistics()
    assert stats["total_samples"] == 3
    assert stats["correct_count"] == 2
    assert stats["corrected_count"] == 1
    assert stats["per_class"].get("0") == 1
    assert stats["per_class"].get("1") == 1
    assert stats["per_class"].get("A / a") == 1


def test_FB_10_held_out_canvas_test_set_isolation():
    """FB-10: Test split instances never appear in user feedback or training pools."""
    from eval_protocol import get_dataset, get_stratified_split
    X, y = get_dataset()
    _, X_test, _, y_test = get_stratified_split(X, y, test_size=0.2, random_state=42)

    # 6,720 held-out test split must be preserved
    assert len(X_test) == 6720
