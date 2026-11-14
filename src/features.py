import numpy as np
import scipy.ndimage as ndi

try:
    from skimage.feature import hog as skimage_hog
    _HAS_SKIMAGE = True
except ImportError:
    _HAS_SKIMAGE = False


def extract_hog_features(
    image,
    pixels_per_cell=(7, 7),
    cells_per_block=(2, 2),
    orientations=9
) -> np.ndarray:
    """
    Computes Dalal-Triggs Histogram of Oriented Gradients (HOG) features.
    Uses skimage.feature.hog if installed, otherwise uses high-speed vectorized scipy/numpy.
    For a 28x28 image with 7x7 cells, returns a 324-dimensional feature vector.
    """
    if isinstance(image, list):
        image = np.array(image, dtype=np.float32)
    if image.ndim == 1:
        image = image.reshape(28, 28)

    # Normalize image to [0, 1] if needed
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    if _HAS_SKIMAGE:
        return skimage_hog(
            image,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm="L2-Hys",
            visualize=False
        ).astype(np.float32)

    # Fast vectorized fallback via Scipy
    gx = ndi.sobel(image, axis=1, mode="constant")
    gy = ndi.sobel(image, axis=0, mode="constant")
    mag = np.hypot(gx, gy)
    ori = np.arctan2(gy, gx) % np.pi  # unsigned gradients in [0, pi]

    h, w = image.shape
    c_y, c_x = pixels_per_cell
    n_cells_y = h // c_y
    n_cells_x = w // c_x

    histograms = np.zeros((n_cells_y, n_cells_x, orientations), dtype=np.float32)
    bin_width = np.pi / orientations
    bin_idx = (ori / bin_width).astype(int) % orientations

    for i in range(n_cells_y):
        for j in range(n_cells_x):
            cell_mag = mag[i * c_y:(i + 1) * c_y, j * c_x:(j + 1) * c_x]
            cell_bin = bin_idx[i * c_y:(i + 1) * c_y, j * c_x:(j + 1) * c_x]
            for b in range(orientations):
                histograms[i, j, b] = np.sum(cell_mag[cell_bin == b])

    b_y, b_x = cells_per_block
    n_blocks_y = n_cells_y - b_y + 1
    n_blocks_x = n_cells_x - b_x + 1

    blocks = []
    eps = 1e-6
    for i in range(n_blocks_y):
        for j in range(n_blocks_x):
            block = histograms[i:i + b_y, j:j + b_x, :].ravel()
            norm = np.sqrt(np.sum(block ** 2) + eps)
            block = np.clip(block / norm, 0.0, 0.2)
            blocks.append(block / np.sqrt(np.sum(block ** 2) + eps))

    return np.concatenate(blocks).astype(np.float32)


def extract_projection_profile(image) -> np.ndarray:
    """
    Computes horizontal and vertical pixel projection profiles + quadrant zoning:
      - 28 horizontal row sums
      - 28 vertical col sums
      - 16 quadrant grid sums (4x4 zones)
    Total: 72 features.
    """
    if image.ndim == 1:
        image = image.reshape(28, 28)
    if image.max() > 1.0:
        image = image.astype(np.float32) / 255.0

    h_proj = np.sum(image, axis=1) / 28.0
    v_proj = np.sum(image, axis=0) / 28.0

    # 4x4 zoning
    zones = []
    for i in range(4):
        for j in range(4):
            zone = image[i*7:(i+1)*7, j*7:(j+1)*7]
            zones.append(np.mean(zone))

    return np.concatenate([h_proj, v_proj, np.array(zones, dtype=np.float32)])


def transform_dataset_features(X, mode: str = "raw_plus_hog") -> np.ndarray:
    """
    Transforms a batch of (N, 784) raw samples according to feature mode:
      - 'raw': returns (N, 784) normalized pixels
      - 'hog': returns (N, 324) Dalal-Triggs HOG gradients
      - 'raw_plus_hog': returns (N, 1108) concatenated raw pixels + HOG
      - 'full': returns (N, 1180) raw + HOG + projection profiles
    """
    if mode == "raw":
        return X.astype(np.float32)

    X_hog = np.array([extract_hog_features(img) for img in X], dtype=np.float32)

    if mode == "hog":
        return X_hog
    elif mode == "raw_plus_hog":
        return np.hstack([X.astype(np.float32), X_hog])
    elif mode == "full":
        X_proj = np.array([extract_projection_profile(img) for img in X], dtype=np.float32)
        return np.hstack([X.astype(np.float32), X_hog, X_proj])
    else:
        raise ValueError(f"Unknown feature mode: '{mode}'")

