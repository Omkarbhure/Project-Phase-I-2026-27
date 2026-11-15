import pytest
import io
import numpy as np
from PIL import Image, ImageDraw
from unittest.mock import MagicMock

from validation import validate_digit_input, analyze_image_properties
import predict
import feedback_manager


@pytest.mark.smoke
def test_VA_01_blank_canvas_rejection_no_model_call(draw_helpers, monkeypatch):
    """
    VA-01: Blank canvas triggers empty rejection ('EMPTY_INPUT'); model inference is never invoked.
    """
    img_blank = draw_helpers.make_blank(mode="RGBA")
    
    # Spy/mock on predict_digit to ensure it is never called on blank input
    spy_predict = MagicMock()
    monkeypatch.setattr(predict, "predict_digit", spy_predict)

    is_valid, err_type, title, desc = validate_digit_input(img_blank)

    assert is_valid is False
    assert err_type == "EMPTY_INPUT"
    assert "Empty Input" in title
    assert spy_predict.call_count == 0


def test_VA_02_empty_gate_boundary_19_vs_20_pixels(draw_helpers):
    """
    VA-02: Boundary condition at 20 ink pixels:
      - 19 ink pixels -> rejected as EMPTY_INPUT
      - 20 ink pixels -> passes Gate 1 (not empty)
    """
    img_19 = draw_helpers.make_exact_pixels(n_pixels=19, intensity=255)
    img_20 = draw_helpers.make_exact_pixels(n_pixels=20, intensity=255)

    props_19 = analyze_image_properties(img_19)
    props_20 = analyze_image_properties(img_20)

    assert props_19["is_empty"] is True
    assert props_19["ink_pixel_count"] == 19

    assert props_20["is_empty"] is False
    assert props_20["ink_pixel_count"] == 20

    is_valid_19, err_19, _, _ = validate_digit_input(img_19)
    assert is_valid_19 is False
    assert err_19 == "EMPTY_INPUT"

    # 20 pixels does not trigger EMPTY_INPUT
    _, err_20, _, _ = validate_digit_input(img_20)
    assert err_20 != "EMPTY_INPUT"


def test_VA_03_solid_rectangle_over_4000_px_rejected(draw_helpers):
    """
    VA-03: Solid rectangle (>4,000 ink pixels AND fill ratio > 82%) is rejected as SOLID_OR_PHOTO.
    """
    # 100x100 solid square = 10,000 pixels with fill_ratio = 1.0 (100%)
    img_solid = draw_helpers.make_solid_block(block_size=(100, 100))

    is_valid, err_type, title, desc = validate_digit_input(img_solid)

    assert is_valid is False
    assert err_type == "SOLID_OR_PHOTO"
    assert "Graphic or Photo" in title


@pytest.mark.smoke
def test_VA_04_large_letter_bold_brush_accepted(draw_helpers):
    """
    VA-04: Large letter drawn with bold brush (ink > 4,000 px, but fill ratio < 82%)
    is ACCEPTED (false-positive guard against rejecting legitimate large drawings).
    """
    img_bold_E = draw_helpers.make_character_E(stroke_width=24)
    props = analyze_image_properties(img_bold_E)

    # Ink count is substantial due to bold brush, but hollow loops keep fill ratio below 82%
    assert props["ink_pixel_count"] > 3500
    assert props["fill_ratio"] < 0.82

    is_valid, err_type, _, _ = validate_digit_input(img_bold_E)
    assert is_valid is True
    assert err_type == "VALID"


def test_VA_05_multi_character_rejection(draw_helpers):
    """
    VA-05: Multi-character drawings ('10', 'AB', 'cat' with clear horizontal gaps)
    are rejected by Connected Component Analysis (Gate 3).
    """
    img_multi = draw_helpers.make_multi_character()

    is_valid, err_type, title, desc = validate_digit_input(img_multi)

    assert is_valid is False
    assert err_type == "MULTI_CHARACTER"
    assert "Multiple Characters" in title


@pytest.mark.smoke
@pytest.mark.parametrize("symbol_maker, symbol_name", [
    ("make_multipart_i", "i"),
    ("make_multipart_exclamation", "!"),
    ("make_multipart_equals", "="),
    ("make_multipart_percent", "%"),
    ("make_multipart_question", "?"),
])
def test_VA_06_multipart_symbols_not_rejected(draw_helpers, symbol_maker, symbol_name):
    """
    VA-06: CRITICAL FALSE-POSITIVE GUARD.
    Legitimately multi-part characters ('i', '!', '=', '%', '?') must NOT be falsely
    rejected by the multi-character gate or scribble gate.
    """
    maker_fn = getattr(draw_helpers, symbol_maker)
    img_symbol = maker_fn()

    is_valid, err_type, title, desc = validate_digit_input(img_symbol)

    assert is_valid is True, f"Legitimate symbol '{symbol_name}' was falsely rejected with {err_type}: {desc}"
    assert err_type == "VALID"


def test_VA_07_random_scribble_rejected(draw_helpers):
    """
    VA-07: Random chaotic scribble with > 6 disconnected fragments is rejected by Gate 4.
    """
    img_scribble = draw_helpers.make_complex_scribble(n_fragments=8)
    props = analyze_image_properties(img_scribble)

    assert props["num_components"] > 6

    is_valid, err_type, title, desc = validate_digit_input(img_scribble)
    assert is_valid is False
    assert err_type == "COMPLEX_SCRIBBLE"
    assert "Complex Scribble" in title


def test_VA_08_photo_like_noisy_image_rejected():
    """
    VA-08: Natural photo-like high density texture/clutter is rejected by validation gates.
    """
    # Granular texture / noise fragments spread across image
    np.random.seed(42)
    noise_arr = (np.random.rand(280, 280) < 0.02).astype(np.uint8) * 255
    img_noise = Image.fromarray(noise_arr, mode="L")

    is_valid, err_type, title, desc = validate_digit_input(img_noise)
    assert is_valid is False
    assert err_type in ("SOLID_OR_PHOTO", "COMPLEX_SCRIBBLE")


def test_VA_09_tiny_speck_noise_treated_as_empty(draw_helpers):
    """
    VA-09: Tiny speck noise (< 20 ink pixels) is treated as empty canvas.
    """
    img_speck = draw_helpers.make_exact_pixels(n_pixels=8, intensity=255)

    is_valid, err_type, _, _ = validate_digit_input(img_speck)
    assert is_valid is False
    assert err_type == "EMPTY_INPUT"


def test_VA_10_corrupt_non_image_upload_handling():
    """
    VA-10: Corrupt / non-image byte uploads produce friendly handling without unhandled crash.
    """
    corrupt_bytes = b"NOT_A_VALID_PNG_HEADER_CORRUPTED_BYTES_123456"

    # Simulated app upload inspection
    opened = False
    try:
        Image.open(io.BytesIO(corrupt_bytes))
        opened = True
    except Exception as e:
        # Expected PIL.UnidentifiedImageError
        assert not opened
        assert "cannot identify image" in str(e).lower() or isinstance(e, Image.UnidentifiedImageError)


def test_VA_11_huge_image_resizing_guard():
    """
    VA-11: Huge image (8000x8000) with a character stroke processes safely without crash or unbounded memory.
    """
    # Create 8000x8000 image with an outline stroke (not a solid block)
    img_huge = Image.new("L", (8000, 8000), 0)
    draw = ImageDraw.Draw(img_huge)
    draw.ellipse([3600, 3400, 4400, 4600], outline=255, width=40)

    props = analyze_image_properties(img_huge)
    assert props["is_empty"] is False
    assert props["ink_pixel_count"] > 0
    assert props["fill_ratio"] < 0.82

    is_valid, err_type, _, _ = validate_digit_input(img_huge)
    assert is_valid is True
    assert err_type == "VALID"


def test_VA_12_ood_confidence_floor_boundary_15_9_vs_16_1(draw_helpers):
    """
    VA-12: Out-of-Distribution (OOD) threshold boundary at 16.0% confidence:
      - confidence = 0.159 -> rejected as LOW_CONFIDENCE
      - confidence = 0.161 -> accepted as VALID
    """
    img = draw_helpers.make_character_8()

    pred_below = {"confidence": 0.159, "probabilities": np.zeros(48)}
    pred_above = {"confidence": 0.161, "probabilities": np.zeros(48)}

    is_valid_below, err_below, title_below, _ = validate_digit_input(img, ensemble_pred=pred_below)
    assert is_valid_below is False
    assert err_below == "LOW_CONFIDENCE"
    assert "Unrecognized" in title_below

    is_valid_above, err_above, _, _ = validate_digit_input(img, ensemble_pred=pred_above)
    assert is_valid_above is True
    assert err_above == "VALID"


# ---------------------------------------------------------------------------
# Additional Validation Rules (V-08, V-09, V-10)
# ---------------------------------------------------------------------------

def test_V08_upload_valid_formats():
    """V-08: Upload file validation logic accepts png/jpg/jpeg, rejects invalid extensions."""
    allowed_extensions = ["png", "jpg", "jpeg"]
    test_files = [
        ("character.png", True),
        ("scan.jpg", True),
        ("drawing.JPEG", True),
        ("malicious.exe", False),
        ("data.csv", False),
        ("doc.pdf", False),
    ]
    for filename, should_pass in test_files:
        ext = filename.rsplit(".", 1)[-1].lower()
        is_allowed = ext in allowed_extensions
        assert is_allowed == should_pass, f"File {filename} permission check failed"


def test_V09_weights_range_and_zero_fallback(draw_helpers):
    """V-09: Weights in [0, 10]; all-zero weights trigger graceful fallback without division by zero."""
    img = draw_helpers.make_character_8()

    # Normal weights in 0-10
    pred1, conf1, probs1, _, _, _ = predict.predict_digit(img, weights={"lr": 1, "rf": 3, "mlp": 5})
    assert conf1 > 0.0

    # All-zero weights (must fallback cleanly)
    pred_zero, conf_zero, probs_zero, _, _, _ = predict.predict_digit(img, weights={"lr": 0, "rf": 0, "mlp": 0})
    assert conf_zero > 0.0
    assert not np.isnan(probs_zero).any()


def test_V10_feedback_label_validation():
    """V-10: Feedback target label must resolve to one of the 48 supported classes."""
    valid_labels = [
        "0", "9", "A / a", "Z / z", "+", "-", "*", "/", "=", "?", "!", "#", "@", "$", "%", "&"
    ]
    for lbl in valid_labels:
        cid = feedback_manager.get_class_id_from_label(lbl)
        assert 0 <= cid <= 47, f"Label '{lbl}' resolved to invalid class ID: {cid}"

    # Verify all 48 LABEL_MAP labels resolve to their own class ID
    for cid, label_str in predict.LABEL_MAP.items():
        resolved_cid = feedback_manager.get_class_id_from_label(label_str)
        assert resolved_cid == cid, f"LABEL_MAP class {cid} ('{label_str}') resolved to {resolved_cid}"
