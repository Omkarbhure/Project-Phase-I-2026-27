import numpy as np
import scipy.ndimage as ndi


def augment_sample(img_flat: np.ndarray, rng: np.random.RandomState = None) -> np.ndarray:
    """
    Applies classical offline data augmentation strictly post-split:
      1. Random Affine: rotation (+/-12 deg), scale (0.88-1.12), shear (+/-0.12), translation (+/-2 px)
      2. Elastic distortion with Gaussian smoothed vector fields
      3. Morphological dilation or erosion (simulating bold / fine pen nibs)
      4. Subtle Gaussian sensor/scanner noise
    Returns: augmented flat (784,) float32 vector in [0.0, 1.0].
    """
    if rng is None:
        rng = np.random.RandomState()

    img = img_flat.reshape(28, 28).astype(np.float32)

    # 1. Random Affine Transformation
    angle = np.deg2rad(rng.uniform(-12, 12))
    scale = rng.uniform(0.88, 1.12)
    shear = rng.uniform(-0.12, 0.12)
    tx = rng.uniform(-2, 2)
    ty = rng.uniform(-2, 2)

    c = 13.5
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    S = np.array([[scale, shear], [0, scale]])
    M = R @ S

    offset = np.array([c, c]) - M @ np.array([c, c]) + np.array([ty, tx])
    aff = ndi.affine_transform(img, M, offset=offset, order=1, mode="constant", cval=0.0)

    # 2. Elastic distortion (probability = 0.35)
    if rng.rand() < 0.35:
        shape = aff.shape
        dx = ndi.gaussian_filter((rng.rand(*shape) * 2 - 1), 2.5, mode="constant", cval=0) * 6.0
        dy = ndi.gaussian_filter((rng.rand(*shape) * 2 - 1), 2.5, mode="constant", cval=0) * 6.0
        y, x = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        aff = ndi.map_coordinates(aff, indices, order=1, mode="constant", cval=0.0).reshape(shape)

    # 3. Morphological dilation / erosion (probability = 0.30)
    p = rng.rand()
    if p < 0.18:
        aff = ndi.grey_dilation(aff, size=(2, 2))
    elif p < 0.28:
        aff = ndi.grey_erosion(aff, size=(2, 2))

    # 4. Subtle Gaussian noise
    if rng.rand() < 0.4:
        aff = aff + rng.normal(0, 0.02, aff.shape)

    # 5. Normalization & clipping
    aff = np.clip(aff, 0.0, 1.0)
    return aff.flatten().astype(np.float32)


def augment_training_set(X_train: np.ndarray, y_train: np.ndarray, multiplier: int = 1, random_state: int = 42):
    """
    Augments the training partition post-split to avoid data leakage.
    multiplier=1 doubles the training set with realistic synthetic variations.
    """
    rng = np.random.RandomState(random_state)
    n_samples = len(X_train)
    
    aug_X = []
    aug_y = []
    
    for _ in range(multiplier):
        for i in range(n_samples):
            aug_X.append(augment_sample(X_train[i], rng))
            aug_y.append(y_train[i])

    X_combined = np.vstack([X_train, np.array(aug_X, dtype=np.float32)])
    y_combined = np.concatenate([y_train, np.array(aug_y, dtype=np.int64)])
    
    # Shuffle combined set
    perm = rng.permutation(len(y_combined))
    return X_combined[perm], y_combined[perm]

