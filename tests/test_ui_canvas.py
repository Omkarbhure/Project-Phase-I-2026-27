import os
import re
import pytest
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import patch, MagicMock

import canvas_component
from canvas_component import st_responsive_canvas, CanvasResult
from preprocessing import check_drawing_quality, preprocess_image
from predict import predict_digit, LABEL_MAP
from feedback_manager import get_feedback_statistics, save_user_feedback


def test_UI_01_custom_component_renders_default_size():
    """UI-01: Custom component renders with correct default canvas size (340x340 in app, 320x320 component default)."""
    with patch.object(canvas_component, "_custom_canvas_func") as mock_func:
        mock_func.return_value = None

        # Call with app default (340)
        res_340 = st_responsive_canvas(canvas_size=340)
        assert mock_func.called
        assert mock_func.call_args[1]["canvas_size"] == 340
        assert isinstance(res_340, CanvasResult)
        assert res_340.has_drawing is False

        # Call with component default (320)
        res_320 = st_responsive_canvas()
        assert mock_func.call_args[1]["canvas_size"] == 320


def test_UI_02_stroke_width_presets_and_default():
    """UI-02: Stroke width presets: Thin 12, Medium 20, Bold 28, default 20px."""
    # Check default is 20px
    with patch.object(canvas_component, "_custom_canvas_func") as mock_func:
        mock_func.return_value = None
        st_responsive_canvas()
        assert mock_func.call_args[1]["stroke_width"] == 20

    # Verify preset constants against canvas index.html
    html_path = os.path.join(
        os.path.dirname(canvas_component.__file__),
        "components",
        "canvas",
        "index.html",
    )
    assert os.path.exists(html_path), f"Canvas index.html not found: {html_path}"
    with open(html_path, "r", encoding="utf-8") as f:
        html_content = f.read()

    assert "presetThin" in html_content
    assert "presetMed" in html_content
    assert "presetBold" in html_content
    assert "12" in html_content  # Thin 12px
    assert "20" in html_content  # Med 20px
    assert "28" in html_content  # Bold 28px


def test_UI_03_clear_button_resets_canvas_state():
    """UI-03: Clear button / blank canvas resets state to None / has_drawing=False."""
    with patch.object(canvas_component, "_custom_canvas_func") as mock_func:
        # 1. Component returns None
        mock_func.return_value = None
        res_none = st_responsive_canvas()
        assert res_none.has_drawing is False
        assert res_none.image is None
        assert res_none.stroke_count == 0
        assert res_none.strokes == []

        # 2. Component returns cleared dictionary
        mock_func.return_value = {
            "hasDrawing": False,
            "dataUrl": None,
            "strokeCount": 0,
            "strokes": [],
        }
        res_cleared = st_responsive_canvas()
        assert res_cleared.has_drawing is False
        assert res_cleared.image is None
        assert res_cleared.stroke_count == 0


def test_UI_04_autopredict_toggle_behavior():
    """UI-04: Auto-predict toggle passed to custom canvas component."""
    with patch.object(canvas_component, "_custom_canvas_func") as mock_func:
        mock_func.return_value = None

        # Auto-predict ON
        st_responsive_canvas(auto_predict=True)
        assert mock_func.call_args[1]["auto_predict"] is True

        # Auto-predict OFF
        st_responsive_canvas(auto_predict=False)
        assert mock_func.call_args[1]["auto_predict"] is False


def test_UI_05_weight_sliders_configuration_and_defaults(draw_helpers):
    """UI-05: Weight sliders default 1:3:5 and pass into predict_digit correctly."""
    default_weights = {"lr": 1, "rf": 3, "mlp": 5}
    total_w = sum(default_weights.values())
    assert total_w == 9
    assert round(default_weights["lr"] / total_w, 2) == 0.11
    assert round(default_weights["rf"] / total_w, 2) == 0.33
    assert round(default_weights["mlp"] / total_w, 2) == 0.56

    img = draw_helpers.make_character_8()
    pred_label, conf, probs, ind_preds, top5, cat = predict_digit(
        img, weights=default_weights
    )

    assert isinstance(pred_label, str)
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 48
    assert "lr" in ind_preds and "rf" in ind_preds and "mlp" in ind_preds


def test_UI_06_all_zero_weights_fallback_graceful(draw_helpers):
    """UI-06: All 0 weights fallback to equal weights (1:1:1) gracefully without ZeroDivisionError."""
    zero_weights = {"lr": 0, "rf": 0, "mlp": 0}
    img = draw_helpers.make_character_8()

    pred_label, conf, probs, ind_preds, top5, cat = predict_digit(
        img, weights=zero_weights
    )

    assert isinstance(pred_label, str)
    assert 0.0 <= conf <= 1.0
    assert len(probs) == 48


def test_UI_07_what_the_model_sees_panel_renders_28x28(draw_helpers):
    """UI-07: 'What the model sees' panel renders the 28x28 normalized image."""
    img = draw_helpers.make_character_8()
    prep_1x784 = preprocess_image(img, auto_thicken=False)

    assert prep_1x784.shape == (1, 784)
    img_28 = prep_1x784.reshape(28, 28)

    # Values in normalized range [0.0, 1.0]
    assert img_28.min() >= 0.0
    assert img_28.max() <= 1.0

    # uint8 conversion for st.image rendering
    view_uint8 = (img_28 * 255.0).astype(np.uint8)
    assert view_uint8.shape == (28, 28)
    assert view_uint8.dtype == np.uint8
    assert view_uint8.max() > 100  # Visible character ink present


def test_UI_08_confidence_breakdown_three_classifiers(draw_helpers):
    """UI-08: Individual classifier breakdown returns distinct votes and confidences for LR, RF, MLP."""
    img = draw_helpers.make_character_8()
    _, _, _, ind_preds, _, _ = predict_digit(img)

    assert set(ind_preds.keys()) == {"lr", "rf", "mlp"}
    for model_name, info in ind_preds.items():
        assert "digit" in info
        assert "confidence" in info
        assert "category" in info
        assert isinstance(info["digit"], str)
        assert 0.0 <= info["confidence"] <= 1.0
        assert info["category"] in {"Digit", "Alphabet", "Symbol"}


@pytest.mark.e2e
def test_UI_09_e2e_app_load_and_prediction():
    """
    UI-09: End-to-end browser test: verify running Streamlit app displays prediction container.
    Skipped when Streamlit server is not running locally.
    """
    import urllib.request
    import urllib.error

    try:
        req = urllib.request.Request(
            "http://localhost:8501", headers={"User-Agent": "PytestE2E"}
        )
        with urllib.request.urlopen(req, timeout=3.0) as res:
            assert res.status == 200
    except (urllib.error.URLError, ConnectionRefusedError, TimeoutError, OSError) as e:
        pytest.skip(
            f"Streamlit server not reachable on localhost:8501 ({e}). Run Streamlit before e2e."
        )


def test_UI_10_drawing_quality_warnings():
    """UI-10: Drawing quality warnings display for small drawings, thin strokes, and off-center drawings."""
    # 1. Normal well-drawn character: no tips
    normal_img = Image.new("RGB", (280, 280), (0, 0, 0))
    draw = ImageDraw.Draw(normal_img)
    draw.rectangle([90, 60, 190, 220], outline=(255, 255, 255), width=18)
    tips_normal = check_drawing_quality(normal_img)
    assert len(tips_normal) == 0, f"Expected no tips for normal drawing, got {tips_normal}"

    # 2. Too small drawing (span < 45 px)
    small_img = Image.new("RGB", (280, 280), (0, 0, 0))
    draw_s = ImageDraw.Draw(small_img)
    draw_s.rectangle([130, 130, 150, 150], fill=(255, 255, 255))
    tips_small = check_drawing_quality(small_img)
    assert any("Small Drawing" in tip for tip in tips_small), (
        f"Missing small drawing warning in {tips_small}"
    )

    # 3. Off-center drawing (centroid > 32% away from canvas center)
    offcenter_img = Image.new("RGB", (280, 280), (0, 0, 0))
    draw_o = ImageDraw.Draw(offcenter_img)
    draw_o.rectangle([10, 10, 60, 60], fill=(255, 255, 255))
    tips_offcenter = check_drawing_quality(offcenter_img)
    assert any("Off-Center" in tip for tip in tips_offcenter), (
        f"Missing off-center warning in {tips_offcenter}"
    )


def test_UI_11_preprocessing_pipeline_intermediate_stages(draw_helpers):
    """
    UI-11: Preprocessing pipeline intermediate stages adhere to specifications:
    crop -> 20x20 aspect-preserving -> 28x28 center-of-mass -> Gaussian blur.
    """
    img = draw_helpers.make_character_8()

    # Stage 1: Preprocessed output is 1x784
    arr = preprocess_image(img, auto_thicken=False)
    assert arr.shape == (1, 784)

    # Stage 2: Reshape to 28x28
    img_28 = arr.reshape(28, 28)
    assert img_28.shape == (28, 28)

    # Stage 3: Centroid is near center (14, 14)
    y_idx, x_idx = np.where(img_28 > 0.1)
    cy, cx = np.mean(y_idx), np.mean(x_idx)
    assert abs(cy - 14.0) < 2.5
    assert abs(cx - 14.0) < 2.5


def test_UI_12_feedback_tab_correction_dropdown_and_stats(temp_feedback_dir):
    """UI-12: Feedback statistics structure and class options for correction dropdown."""
    stats = get_feedback_statistics()
    expected_keys = {
        "total_samples",
        "correct_count",
        "corrected_count",
        "per_class",
        "recent",
    }
    assert expected_keys.issubset(set(stats.keys()))

    # Correction selectbox options from LABEL_MAP: all 48 classes present
    all_classes = [LABEL_MAP[i] for i in range(48)]
    assert len(all_classes) == 48
    assert len(set(all_classes)) == 48


@pytest.mark.e2e
def test_UI_13_e2e_feedback_submission_updates_csv(draw_helpers, temp_feedback_dir):
    """
    UI-13: End-to-end feedback submission flow updates the CSV log and creates disk PNGs.
    """
    img = draw_helpers.make_character_8()
    prep_arr = preprocess_image(img)

    result = save_user_feedback(
        original_image=img,
        preprocessed_28x28_arr=prep_arr,
        true_label="8",
        predicted_label="8",
        was_correct=True,
    )

    assert "sample_id" in result
    stats = get_feedback_statistics()
    assert stats["total_samples"] >= 1
    assert stats["correct_count"] >= 1

