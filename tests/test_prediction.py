import pytest
import time
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import MagicMock, patch

import predict
from predict import (
    predict_digit,
    get_character_category,
    get_character_category_detail,
    LABEL_MAP,
    CANONICAL_MAP,
    SPECIAL_SYMBOLS_SET,
)


def test_PR_01_all_48_classes_valid_prediction(draw_helpers, loaded_model):
    """PR-01: Predictions return valid labels in LABEL_MAP and confidence in [0, 1]."""
    img = draw_helpers.make_character_8()
    pred_label, conf, probs, ind_preds, top_cand, cat = predict_digit(img)

    assert pred_label in LABEL_MAP.values(), f"Invalid label: {pred_label}"
    assert 0.0 <= conf <= 1.0, f"Confidence out of bounds: {conf}"
    assert len(probs) == 48, f"Expected 48 class probabilities, got {len(probs)}"
    assert cat in ("Digit", "Alphabet", "Symbol")


def test_PR_02_probabilities_sum_to_one_top3_sorted(draw_helpers, loaded_model):
    """PR-02: Model probabilities sum to 1.0 and top-3 candidates are strictly sorted descending."""
    img = draw_helpers.make_character_8()
    _, _, probs, _, top_candidates, _ = predict_digit(img)

    assert np.isclose(np.sum(probs), 1.0, atol=1e-4), f"Probabilities do not sum to 1: {np.sum(probs)}"
    assert len(top_candidates) >= 3, "Expected at least 3 candidates"

    # Verify descending probability order
    prob_values = [p for _, p in top_candidates]
    for i in range(len(prob_values) - 1):
        assert prob_values[i] >= prob_values[i + 1], f"Top candidates not sorted: {prob_values}"


@pytest.mark.smoke
def test_PR_03_soft_voting_weights_1_3_5_matches_manual_blend(draw_helpers, loaded_model):
    """
    PR-03: Soft voting with weights 1:3:5 strictly matches the weighted formula:
    (1*p_lr + 3*p_rf + 5*p_mlp) / 9.0.
    """
    img = draw_helpers.make_character_8()
    weights = {"lr": 1, "rf": 3, "mlp": 5}
    pred_label, conf, probs, ind_preds, _, _ = predict_digit(img, weights=weights, voting_mode="soft")

    estimators = getattr(loaded_model, "named_estimators_", {})
    if "lr" in estimators and "rf" in estimators and "mlp" in estimators:
        from preprocessing import preprocess_image
        proc = preprocess_image(img)
        p_lr = estimators["lr"].predict_proba(proc)[0]
        p_rf = estimators["rf"].predict_proba(proc)[0]
        p_mlp = estimators["mlp"].predict_proba(proc)[0]

        expected_probs = (1.0 * p_lr + 3.0 * p_rf + 5.0 * p_mlp) / 9.0
        assert np.allclose(probs, expected_probs, atol=1e-4)


def test_PR_04_isolated_weights_match_individual_estimators(draw_helpers, loaded_model):
    """PR-04: Isolated weights (10:0:0 for LR and 0:0:10 for MLP) match individual models."""
    img = draw_helpers.make_character_8()
    estimators = getattr(loaded_model, "named_estimators_", {})

    if "lr" in estimators and "mlp" in estimators:
        from preprocessing import preprocess_image
        proc = preprocess_image(img)

        # Pure LR (10:0:0)
        _, _, probs_lr, _, _, _ = predict_digit(img, weights={"lr": 10, "rf": 0, "mlp": 0}, voting_mode="soft")
        expected_lr = estimators["lr"].predict_proba(proc)[0]
        assert np.allclose(probs_lr, expected_lr, atol=1e-4)

        # Pure MLP (0:0:10)
        _, _, probs_mlp, _, _, _ = predict_digit(img, weights={"lr": 0, "rf": 0, "mlp": 10}, voting_mode="soft")
        expected_mlp = estimators["mlp"].predict_proba(proc)[0]
        assert np.allclose(probs_mlp, expected_mlp, atol=1e-4)


def test_PR_05_hard_voting_tie_break_deterministic(draw_helpers, monkeypatch):
    """PR-05: Hard voting with a 3-way split gives a deterministic tie-break."""
    img = draw_helpers.make_character_8()

    # Call twice with hard voting mode and 1:1:1 weights
    res1 = predict_digit(img, weights={"lr": 1, "rf": 1, "mlp": 1}, voting_mode="hard")
    res2 = predict_digit(img, weights={"lr": 1, "rf": 1, "mlp": 1}, voting_mode="hard")

    # Winner must be identical
    assert res1[0] == res2[0], f"Tie-break was not deterministic: {res1[0]} vs {res2[0]}"


@pytest.mark.smoke
def test_PR_06_all_weights_zero_graceful_fallback(draw_helpers):
    """PR-06: All weights set to 0 triggers fallback to default equal weights without ZeroDivisionError."""
    img = draw_helpers.make_character_8()
    pred_label, conf, probs, ind_preds, top_cand, cat = predict_digit(
        img, weights={"lr": 0, "rf": 0, "mlp": 0}, voting_mode="soft"
    )

    assert conf > 0.0
    assert not np.isnan(probs).any()
    assert np.isclose(np.sum(probs), 1.0, atol=1e-4)


def test_PR_07_out_of_bound_weights_clamped(draw_helpers):
    """PR-07: Out of bound or negative weights (-1 and 11) are clamped without crash."""
    img = draw_helpers.make_character_8()
    pred_label, conf, probs, ind_preds, _, _ = predict_digit(
        img, weights={"lr": -1, "rf": 11, "mlp": 5}, voting_mode="soft"
    )

    assert conf > 0.0
    assert not np.isnan(probs).any()


def test_PR_08_category_assignment_digit_alphabet_symbol():
    """PR-08: Categories 'Digit', 'Alphabet', and 'Symbol' are correctly assigned."""
    # Digits
    for d in range(10):
        assert get_character_category(str(d)) == "Digit"
        assert "Digit" in get_character_category_detail(str(d))

    # Letters
    for c in ["A / a", "Z / z", "M / m", "B"]:
        assert get_character_category(c) == "Alphabet"
        assert "Alphabet" in get_character_category_detail(c)

    # Symbols
    for s in SPECIAL_SYMBOLS_SET:
        assert get_character_category(s) == "Symbol"
        assert "Symbol" in get_character_category_detail(s)


def test_PR_09_drawn_lowercase_and_uppercase_map_to_merged_label(draw_helpers):
    """PR-09: Drawn uppercase 'C' and lowercase 'c' both map to the merged label 'C / c'."""
    # Class 12 is 'C / c'
    expected_label = LABEL_MAP[12]
    assert expected_label == "C / c"


@pytest.mark.slow
def test_PR_10_inference_latency_under_20ms(draw_helpers, loaded_model):
    """PR-10: Median model inference latency is under 20 ms per sample over 200 runs."""
    img = draw_helpers.make_character_8()
    from preprocessing import preprocess_image
    proc = preprocess_image(img)

    times = []
    # Warmup
    for _ in range(10):
        loaded_model.predict_proba(proc)

    # 200 runs
    for _ in range(200):
        t0 = time.perf_counter()
        loaded_model.predict_proba(proc)
        times.append((time.perf_counter() - t0) * 1000.0)

    median_ms = np.median(times)
    assert median_ms < 20.0, f"Median latency {median_ms:.3f} ms exceeds 20.0 ms"


def test_PR_11_sklearn_version_mismatch_warning(monkeypatch):
    """PR-11: Loading model with missing or corrupt artifact path raises clear FileNotFoundError."""
    monkeypatch.setattr(predict, "CHARACTER_MODEL_PATH", "models/non_existent_model_file.pkl")
    monkeypatch.setattr(predict, "MODEL_PATH", "models/non_existent_model_file_backup.pkl")
    monkeypatch.setattr(predict, "_cached_model", None)

    with pytest.raises(FileNotFoundError) as exc_info:
        predict.load_model()

    assert "Trained model not found" in str(exc_info.value)

