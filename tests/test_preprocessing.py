import pytest
import numpy as np
import scipy.ndimage
from PIL import Image, ImageDraw

from preprocessing import preprocess_image
from features import extract_hog_features, transform_dataset_features


def test_PP_01_transparent_rgba_white_strokes(draw_helpers):
    """PP-01: Transparent RGBA with white strokes is alpha-composited on black, returning valid (1, 784) in [0, 1]."""
    img = Image.new("RGBA", (280, 280), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.line([(50, 140), (230, 140)], fill=(255, 255, 255, 255), width=20)

    res = preprocess_image(img)

    assert isinstance(res, np.ndarray)
    assert res.shape == (1, 784)
    assert res.dtype == np.float32
    assert np.all((res >= 0.0) & (res <= 1.0))
    assert np.max(res) > 0.5  # Foreground ink preserved


def test_PP_02_black_ink_auto_inverted(draw_helpers):
    """PP-02: Black ink on white background is auto-detected and inverted to standard MNIST format."""
    img = Image.new("L", (280, 280), 255)  # White background
    draw = ImageDraw.Draw(img)
    draw.rectangle([100, 50, 180, 230], fill=0)  # Black ink

    res = preprocess_image(img, invert_if_needed=True)

    arr_28 = res.reshape(28, 28)
    # Background inverted to black (approx 0.0 within Gaussian tail tolerance)
    assert np.isclose(arr_28[0, 0], 0.0, atol=1e-4)
    assert np.max(arr_28) > 0.8  # Digit inverted to white foreground (> 0.8)


def test_PP_03_faint_pixels_removed(draw_helpers):
    """PP-03: Faint noise pixels below luminance 30 are zeroed out and do not taint the bounding box."""
    # Image with faint noise (luminance 22 < 30) in top-left
    img_faint = Image.new("L", (280, 280), 0)
    d_faint = ImageDraw.Draw(img_faint)
    d_faint.rectangle([10, 10, 50, 50], fill=22)  # Faint noise < 30
    d_faint.rectangle([100, 100, 180, 180], fill=200)  # Main character stroke

    # Identical image WITHOUT the faint noise
    img_clean = Image.new("L", (280, 280), 0)
    d_clean = ImageDraw.Draw(img_clean)
    d_clean.rectangle([100, 100, 180, 180], fill=200)

    res_faint = preprocess_image(img_faint)
    res_clean = preprocess_image(img_clean)

    # Preprocessed output must be identical because faint noise was zeroed out
    assert np.allclose(res_faint, res_clean, atol=1e-4)


@pytest.mark.smoke
def test_PP_04_horizontal_line_not_stretched(draw_helpers):
    """
    PP-04: Horizontal line stays wide and flat; NOT stretched into a square.
    (Verifies aspect-ratio-preserving scaling fix for '-' vs 'I' confusion).
    """
    img = draw_helpers.make_horizontal_line(width=200, height=10)
    res = preprocess_image(img).reshape(28, 28)

    # Active ink in 28x28
    bin_ink = res > 0.15
    y_idx, x_idx = np.where(bin_ink)
    assert len(x_idx) > 0, "No ink detected in preprocessed horizontal line"

    w = np.max(x_idx) - np.min(x_idx) + 1
    h = np.max(y_idx) - np.min(y_idx) + 1
    aspect_ratio = w / float(max(1, h))

    # The horizontal line must span the 20px box horizontally but remain thin vertically
    assert w >= 16, f"Expected horizontal span >= 16, got {w}"
    assert h <= 6, f"Expected height <= 6 (not stretched), got {h}"
    assert aspect_ratio >= 3.0, f"Aspect ratio must be >= 3.0, got {aspect_ratio:.2f}"


def test_PP_05_vertical_line_tall_and_narrow(draw_helpers):
    """PP-05: Vertical line stays tall and narrow (height ~20px, width narrow)."""
    img = draw_helpers.make_vertical_line(width=10, height=200)
    res = preprocess_image(img).reshape(28, 28)

    bin_ink = res > 0.15
    y_idx, x_idx = np.where(bin_ink)
    assert len(y_idx) > 0

    w = np.max(x_idx) - np.min(x_idx) + 1
    h = np.max(y_idx) - np.min(y_idx) + 1
    aspect_ratio = h / float(max(1, w))

    assert h >= 16, f"Expected vertical span >= 16, got {h}"
    assert w <= 6, f"Expected width <= 6, got {w}"
    assert aspect_ratio >= 3.0, f"Vertical aspect ratio must be >= 3.0, got {aspect_ratio:.2f}"


@pytest.mark.smoke
def test_PP_06_position_invariance_center_of_mass(draw_helpers):
    """
    PP-06: Same character at different canvas positions yields near-identical output,
    with Center-of-Mass aligned at (14, 14) +/- 1.5 px.
    """
    img_center = draw_helpers.make_character_8(offset=(0, 0))
    img_offset = draw_helpers.make_character_8(offset=(60, -50))

    res_center = preprocess_image(img_center).reshape(28, 28)
    res_offset = preprocess_image(img_offset).reshape(28, 28)

    cy1, cx1 = scipy.ndimage.center_of_mass(res_center)
    cy2, cx2 = scipy.ndimage.center_of_mass(res_offset)

    # Center-of-mass must be tightly centered
    assert abs(cy1 - 14.0) <= 1.5, f"Centroid y {cy1:.2f} off target"
    assert abs(cx1 - 14.0) <= 1.5, f"Centroid x {cx1:.2f} off target"
    assert abs(cy2 - 14.0) <= 1.5, f"Offset centroid y {cy2:.2f} off target"
    assert abs(cx2 - 14.0) <= 1.5, f"Offset centroid x {cx2:.2f} off target"

    # Normalized vectors should have high cosine similarity (> 0.90)
    v1 = res_center.flatten()
    v2 = res_offset.flatten()
    cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6)
    assert cos_sim >= 0.90, f"Position invariance cosine similarity too low: {cos_sim:.4f}"


def test_PP_07_scale_invariance_small_vs_large(draw_helpers):
    """PP-07: Same character drawn small (45px) vs large (220px) normalizes to near-identical output."""
    # Small '8'
    img_small = Image.new("L", (280, 280), 0)
    d_sm = ImageDraw.Draw(img_small)
    d_sm.ellipse([115, 115, 165, 140], outline=255, width=6)
    d_sm.ellipse([112, 138, 168, 168], outline=255, width=6)

    # Large '8'
    img_large = Image.new("L", (280, 280), 0)
    d_lg = ImageDraw.Draw(img_large)
    d_lg.ellipse([60, 40, 220, 140], outline=255, width=22)
    d_lg.ellipse([50, 135, 230, 245], outline=255, width=22)

    res_small = preprocess_image(img_small).flatten()
    res_large = preprocess_image(img_large).flatten()

    cos_sim = np.dot(res_small, res_large) / (np.linalg.norm(res_small) * np.linalg.norm(res_large) + 1e-6)
    assert cos_sim >= 0.80, f"Scale invariance cosine similarity too low: {cos_sim:.4f}"


def test_PP_08_auto_thicken_under_2_2px(draw_helpers):
    """PP-08: Stroke under 2.2 px: auto-thicken on dilates strokes, off leaves them unchanged."""
    # 2px circular stroke (thin stroke measuring ~2.0px in 28x28 space, below the 2.2px threshold)
    img = Image.new("L", (280, 280), 0)
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 80, 200, 200], outline=255, width=2)

    res_thin = preprocess_image(img, auto_thicken=False).reshape(28, 28)
    res_thick = preprocess_image(img, auto_thicken=True).reshape(28, 28)

    ink_thin = np.sum(res_thin > 0.20)
    ink_thick = np.sum(res_thick > 0.20)

    # Auto-thicken must dilate the stroke, increasing active foreground mass
    assert ink_thick > ink_thin, f"Auto-thicken should increase ink mass: {ink_thick} vs {ink_thin}"


def test_PP_09_deterministic_output(draw_helpers):
    """PP-09: Preprocessing is strictly deterministic on identical inputs."""
    img = draw_helpers.make_character_8()
    res1 = preprocess_image(img)
    res2 = preprocess_image(img)

    assert np.array_equal(res1, res2), "preprocess_image output was not bitwise deterministic"


def test_PP_10_hog_and_raw_plus_hog_dimensions():
    """PP-10: HOG gives 324 dims; Raw+HOG gives 1,108 dims."""
    dummy_img = np.zeros(784, dtype=np.float32)

    hog_feat = extract_hog_features(dummy_img)
    assert hog_feat.shape == (324,), f"HOG feature dimension must be 324, got {hog_feat.shape}"

    batch = np.zeros((3, 784), dtype=np.float32)
    raw_plus_hog = transform_dataset_features(batch, mode="raw_plus_hog")
    assert raw_plus_hog.shape == (3, 1108), f"Raw+HOG must be 1,108, got {raw_plus_hog.shape}"
