
import numpy as np

MODALITIES = ["t1", "t1ce", "t2", "flair"]

ORIGINAL_LABELS = {0, 1, 2, 4}


def normalize_mri(volume, modality_name="unknown"):
    """
    Chuẩn hóa cường độ của một modality MRI.

    Input:
        volume: NumPy array 3D.

    Output:
        NumPy array 3D, dtype float32.

    Các voxel bằng 0 được giữ nguyên là 0.
    """

    volume = np.asarray(volume, dtype=np.float32)

    if volume.ndim != 3:
        raise ValueError(
            f"{modality_name}: expected 3D volume, "
            f"received shape {volume.shape}"
        )

    if not np.isfinite(volume).all():
        raise ValueError(
            f"{modality_name}: found NaN or Inf values"
        )

    foreground = volume != 0

    if not foreground.any():
        raise ValueError(
            f"{modality_name}: the entire volume is zero"
        )

    values = volume[foreground]

    mean = float(values.mean(dtype=np.float64))
    std = float(values.std(dtype=np.float64))

    if not np.isfinite(std) or std < 1e-6:
        raise ValueError(
            f"{modality_name}: invalid or near-zero standard deviation"
        )

    normalized = np.zeros_like(volume, dtype=np.float32)

    normalized[foreground] = (
        volume[foreground] - mean
    ) / std

    return normalized


def remap_brats_labels(segmentation):
    """
    Chuyển nhãn BraTS gốc thành nhãn liên tục:

        0 -> 0: Background
        1 -> 1
        2 -> 2
        4 -> 3

    Output dtype: uint8.
    """

    if not np.isfinite(segmentation).all():
        raise ValueError("Segmentation contains NaN or Inf")

    rounded = np.rint(segmentation)

    if not np.allclose(segmentation, rounded):
        raise ValueError("Segmentation contains non-integer labels")

    rounded = rounded.astype(np.uint8)

    unique_labels = set(
        np.unique(rounded).astype(int).tolist()
    )

    unexpected = unique_labels - ORIGINAL_LABELS

    if unexpected:
        raise ValueError(
            f"Unexpected original labels: {sorted(unexpected)}"
        )

    remapped = np.zeros(rounded.shape, dtype=np.uint8)

    remapped[rounded == 1] = 1
    remapped[rounded == 2] = 2
    remapped[rounded == 4] = 3

    return remapped
