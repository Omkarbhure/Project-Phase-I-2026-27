import pytest
import numpy as np

from features import (
    extract_hog_features,
    extract_projection_profile,
    transform_dataset_features,
)


def test_features_extract_hog_dimensions():
    """Verifies Dalal-Triggs HOG returns exactly 324 dimensions for 1D, 2D, and list inputs."""
    # 1D array
    arr_1d = np.zeros(784, dtype=np.float32)
    h_1d = extract_hog_features(arr_1d)
    assert h_1d.shape == (324,)
    assert not np.isnan(h_1d).any()

    # 2D array
    arr_2d = np.zeros((28, 28), dtype=np.float32)
    h_2d = extract_hog_features(arr_2d)
    assert h_2d.shape == (324,)
    assert np.array_equal(h_1d, h_2d)

    # List input
    list_input = [0.0] * 784
    h_list = extract_hog_features(list_input)
    assert h_list.shape == (324,)


def test_features_hog_l2hys_normalization():
    """Verifies HOG L2-Hys block normalization clips and scales values into [0.0, 1.0]."""
    # Create synthetic gradient
    img = np.tile(np.linspace(0, 1, 28, dtype=np.float32), (28, 1))
    h = extract_hog_features(img)

    assert h.dtype == np.float32
    assert np.all(h >= 0.0)
    assert np.all(h <= 1.0)
    assert np.max(h) > 0.0


def test_features_hog_intensity_invariance():
    """
    Verifies that Dalal-Triggs HOG features are invariant to global contrast/illumination scaling
    due to block-level normalization.
    """
    img1 = np.zeros((28, 28), dtype=np.float32)
    img1[8:20, 10:18] = 0.40  # Moderate intensity

    img2 = np.zeros((28, 28), dtype=np.float32)
    img2[8:20, 10:18] = 0.80  # Double intensity (same geometry)

    h1 = extract_hog_features(img1)
    h2 = extract_hog_features(img2)

    # Normalized HOG vectors should be near-identical
    cos_sim = np.dot(h1, h2) / (np.linalg.norm(h1) * np.linalg.norm(h2) + 1e-6)
    assert cos_sim >= 0.98, f"HOG intensity invariance cosine similarity too low: {cos_sim:.4f}"


def test_features_projection_profile():
    """Verifies projection profile extracts 28 horizontal + 28 vertical + 16 zoning = 72 features."""
    img = np.zeros((28, 28), dtype=np.float32)
    img[14, :] = 1.0  # Horizontal line across middle row

    proj = extract_projection_profile(img)

    assert proj.shape == (72,)
    assert not np.isnan(proj).any()

    # The 14th horizontal row should have average = 1.0
    assert np.isclose(proj[14], 1.0)
    # Other horizontal rows should be 0.0
    assert np.isclose(proj[0], 0.0)


def test_features_transform_dataset_features_modes():
    """Verifies transform_dataset_features handles all modes and dimension contracts."""
    X = np.random.rand(4, 784).astype(np.float32)

    # 1. Raw mode (784)
    X_raw = transform_dataset_features(X, mode="raw")
    assert X_raw.shape == (4, 784)

    # 2. HOG mode (324)
    X_hog = transform_dataset_features(X, mode="hog")
    assert X_hog.shape == (4, 324)

    # 3. Raw + HOG mode (1108)
    X_combo = transform_dataset_features(X, mode="raw_plus_hog")
    assert X_combo.shape == (4, 1108)
    assert np.array_equal(X_combo[:, :784], X_raw)
    assert np.array_equal(X_combo[:, 784:], X_hog)

    # 4. Full mode (1180)
    X_full = transform_dataset_features(X, mode="full")
    assert X_full.shape == (4, 1180)

    # 5. Invalid mode raises ValueError
    with pytest.raises(ValueError) as exc:
        transform_dataset_features(X, mode="non_existent_mode")
    assert "Unknown feature mode" in str(exc.value)

