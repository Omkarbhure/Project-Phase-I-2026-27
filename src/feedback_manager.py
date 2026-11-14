import os
import sys
import csv
import time
import json
import numpy as np
from PIL import Image

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from predict import LABEL_MAP, CANONICAL_MAP

BASE_DIR = os.path.dirname(CURRENT_DIR)
FEEDBACK_DIR = os.path.join(BASE_DIR, "data", "user_feedback")
FEEDBACK_LOG_CSV = os.path.join(FEEDBACK_DIR, "feedback_log.csv")

# Safe folder naming for Windows filesystems (avoids / * ? : etc.)
SAFE_CLASS_NAMES = {
    0: "digit_0", 1: "digit_1", 2: "digit_2", 3: "digit_3", 4: "digit_4",
    5: "digit_5", 6: "digit_6", 7: "digit_7", 8: "digit_8", 9: "digit_9",
    **{10 + i: f"letter_{chr(65 + i)}" for i in range(26)},
    36: "symbol_plus",
    37: "symbol_minus",
    38: "symbol_asterisk",
    39: "symbol_slash",
    40: "symbol_equals",
    41: "symbol_question",
    42: "symbol_exclamation",
    43: "symbol_hash",
    44: "symbol_at",
    45: "symbol_dollar",
    46: "symbol_percent",
    47: "symbol_ampersand",
}

LABEL_TO_ID = {v: k for k, v in LABEL_MAP.items()}
# Also map canonical uppercase and single chars
for k, v in CANONICAL_MAP.items():
    LABEL_TO_ID[v] = k
    if v.isalpha():
        LABEL_TO_ID[v.lower()] = k


def get_class_id_from_label(label: str) -> int:
    """Robustly resolve any label variant to its 0-47 class ID."""
    clean = label.strip()
    if clean in LABEL_TO_ID:
        return LABEL_TO_ID[clean]
    # Check partial match in LABEL_MAP values
    for cid, val in LABEL_MAP.items():
        if clean in val or val in clean:
            return cid
    return 0


def ensure_feedback_dirs():
    os.makedirs(FEEDBACK_DIR, exist_ok=True)
    if not os.path.exists(FEEDBACK_LOG_CSV):
        with open(FEEDBACK_LOG_CSV, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "sample_id",
                "class_id",
                "true_label",
                "predicted_label",
                "was_correct",
                "original_img_path",
                "preprocessed_img_path",
                "stroke_count",
            ])


def save_user_feedback(
    original_image: Image.Image,
    preprocessed_28x28_arr: np.ndarray,
    true_label: str,
    predicted_label: str,
    was_correct: bool,
    stroke_data: list = None,
) -> dict:
    """
    Saves a user feedback sample:
      - Original full resolution canvas drawing (PNG)
      - Preprocessed 28x28 model input (PNG)
      - Metadata record in data/user_feedback/feedback_log.csv
    """
    ensure_feedback_dirs()

    class_id = get_class_id_from_label(true_label)
    safe_folder = SAFE_CLASS_NAMES.get(class_id, f"class_{class_id}")
    class_dir = os.path.join(FEEDBACK_DIR, safe_folder)
    os.makedirs(class_dir, exist_ok=True)

    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    sample_id = f"{timestamp_str}_{int(time.time() * 1000) % 1000:03d}"

    orig_filename = f"{sample_id}_orig.png"
    prep_filename = f"{sample_id}_28x28.png"

    orig_path = os.path.join(class_dir, orig_filename)
    prep_path = os.path.join(class_dir, prep_filename)

    # 1. Save original
    if isinstance(original_image, Image.Image):
        original_image.save(orig_path, format="PNG")
    elif isinstance(original_image, np.ndarray):
        Image.fromarray(original_image.astype(np.uint8)).save(orig_path, format="PNG")

    # 2. Save preprocessed 28x28
    if preprocessed_28x28_arr is not None:
        p_arr = np.array(preprocessed_28x28_arr).reshape(28, 28)
        if p_arr.max() <= 1.0:
            p_arr = (p_arr * 255.0).astype(np.uint8)
        else:
            p_arr = p_arr.astype(np.uint8)
        Image.fromarray(p_arr, mode="L").save(prep_path, format="PNG")

    # 3. Save stroke data if present
    stroke_count = len(stroke_data) if stroke_data else 0
    if stroke_data:
        json_path = os.path.join(class_dir, f"{sample_id}_strokes.json")
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(stroke_data, f)
        except Exception:
            pass

    # 4. Append to CSV
    rel_orig = os.path.relpath(orig_path, BASE_DIR).replace("\\", "/")
    rel_prep = os.path.relpath(prep_path, BASE_DIR).replace("\\", "/")

    with open(FEEDBACK_LOG_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            time.strftime("%Y-%m-%d %H:%M:%S"),
            sample_id,
            class_id,
            LABEL_MAP.get(class_id, true_label),
            predicted_label,
            1 if was_correct else 0,
            rel_orig,
            rel_prep,
            stroke_count,
        ])

    return {
        "sample_id": sample_id,
        "class_id": class_id,
        "saved_to": rel_orig,
        "was_correct": was_correct,
    }


def get_feedback_statistics() -> dict:
    """Returns aggregated feedback counts and recent logs."""
    ensure_feedback_dirs()
    if not os.path.exists(FEEDBACK_LOG_CSV):
        return {"total_samples": 0, "correct_count": 0, "corrected_count": 0, "per_class": {}, "recent": []}

    rows = []
    with open(FEEDBACK_LOG_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    total = len(rows)
    correct = sum(1 for r in rows if r.get("was_correct") == "1")
    corrected = total - correct

    per_class = {}
    for r in rows:
        cid = int(r.get("class_id", 0))
        label = LABEL_MAP.get(cid, str(cid))
        per_class[label] = per_class.get(label, 0) + 1

    recent = rows[-10:][::-1]  # Most recent 10

    return {
        "total_samples": total,
        "correct_count": correct,
        "corrected_count": corrected,
        "per_class": per_class,
        "recent": recent,
    }

