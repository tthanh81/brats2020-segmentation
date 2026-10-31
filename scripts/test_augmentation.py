
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.augmentation import BraTSPatchAugmentation


def create_sample():
    """
    Create a small synthetic 3D MRI and segmentation mask.
    """
    label = torch.zeros((16, 16, 16), dtype=torch.uint8)

    # Two separate tumor regions
    label[2:8, 3:9, 4:10] = 1
    label[10:14, 10:15, 4:10] = 3

    image = torch.zeros((4, 16, 16, 16), dtype=torch.float32)

    # Four synthetic MRI modalities
    for channel in range(4):
        image[channel] = (
            label.float() * (1.0 + channel * 0.5)
        )

    return {
        "image": image,
        "label": label,
        "subject_id": "synthetic_001"
    }


def main():
    torch.manual_seed(42)

    original = create_sample()

    print("=" * 55)
    print("BraTS2020 — 3D Augmentation Unit Tests")
    print("=" * 55)

    # Test 1: No augmentation
    identity = BraTSPatchAugmentation(
        flip_prob=0.0,
        intensity_prob=0.0
    )

    unchanged = identity(original)

    assert torch.equal(
        unchanged["image"], original["image"]
    )
    assert torch.equal(
        unchanged["label"], original["label"]
    )

    print("[PASS] Identity transformation")

    # Test 2: Force both spatial flips
    spatial = BraTSPatchAugmentation(
        flip_prob=1.0,
        intensity_prob=0.0
    )

    flipped = spatial(original)

    expected_image = torch.flip(
        original["image"], dims=(1, 2)
    )
    expected_label = torch.flip(
        original["label"], dims=(0, 1)
    )

    assert torch.equal(
        flipped["image"], expected_image
    )
    assert torch.equal(
        flipped["label"], expected_label
    )

    print("[PASS] Synchronized spatial flips")

    # Test 3: Force intensity augmentation
    intensity = BraTSPatchAugmentation(
        flip_prob=0.0,
        intensity_prob=1.0
    )

    changed = intensity(original)

    assert torch.equal(
        changed["label"], original["label"]
    )

    assert not torch.equal(
        changed["image"], original["image"]
    )

    print("[PASS] Intensity transforms affect MRI only")

    # Test 4: Background preservation
    background = original["image"] == 0

    assert torch.all(
        changed["image"][background] == 0
    )

    print("[PASS] Zero background preserved")

    # Test 5: Check shapes
    for sample in (unchanged, flipped, changed):
        assert sample["image"].shape == (4, 16, 16, 16)
        assert sample["label"].shape == (16, 16, 16)

    print("[PASS] Tensor shapes preserved")

    # Test 6: Check segmentation labels
    for sample in (unchanged, flipped, changed):
        unique_labels = set(
            torch.unique(sample["label"]).tolist()
        )
        assert unique_labels.issubset({0, 1, 2, 3})

    print("[PASS] Segmentation labels valid")

    # Test 7: Numerical stability
    for sample in (unchanged, flipped, changed):
        assert torch.isfinite(sample["image"]).all()

    print("[PASS] No NaN or Inf")

    # Test 8: Original sample was not modified
    reference = create_sample()

    assert torch.equal(
        original["image"], reference["image"]
    )
    assert torch.equal(
        original["label"], reference["label"]
    )

    print("[PASS] Original data not modified")

    # Test 9: Metadata preservation
    assert flipped["subject_id"] == "synthetic_001"

    print("[PASS] Metadata preserved")

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        slice_index = 6

        fig, axes = plt.subplots(2, 2, figsize=(9, 9))

        axes[0, 0].imshow(
            original["image"][0, :, :, slice_index],
            cmap="gray"
        )
        axes[0, 0].set_title("Original MRI")

        axes[0, 1].imshow(
            original["label"][:, :, slice_index],
            cmap="viridis", vmin=0, vmax=3
        )
        axes[0, 1].set_title("Original Mask")

        axes[1, 0].imshow(
            flipped["image"][0, :, :, slice_index],
            cmap="gray"
        )
        axes[1, 0].set_title("Augmented MRI — Flipped")

        axes[1, 1].imshow(
            flipped["label"][:, :, slice_index],
            cmap="viridis", vmin=0, vmax=3
        )
        axes[1, 1].set_title("Augmented Mask — Flipped")

        for ax in axes.flat:
            ax.axis("off")

        fig.tight_layout()

        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)

        output_path = (
            output_dir / "augmentation_test_preview.png"
        )

        fig.savefig(output_path, dpi=150)
        plt.close(fig)

        print("[PASS] Visualization saved:", output_path)

    except ImportError:
        print("[SKIP] Matplotlib not installed")

    print()
    print("=" * 55)
    print("ALL AUGMENTATION TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
