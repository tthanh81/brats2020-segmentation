
from pathlib import Path
import sys

import torch
from torch.utils.data import Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.patches import RandomPatchDataset
from src.data.augmentation import BraTSPatchAugmentation


class SyntheticVolumeDataset(Dataset):
    """One synthetic 3D MRI volume with a tumor mask."""

    def __len__(self):
        return 1

    def __getitem__(self, index):
        image = torch.zeros((4, 20, 20, 20), dtype=torch.float32)
        label = torch.zeros((20, 20, 20), dtype=torch.uint8)

        label[4:14, 5:15, 6:16] = 1
        label[8:12, 8:12, 8:12] = 3

        for channel in range(4):
            image[channel] = label.float() * (channel + 1)

        return {
            "image": image,
            "label": label,
            "subject_id": "synthetic_001",
        }


class TrackingAugmentation:
    """Record how many times augmentation was called."""

    def __init__(self):
        self.calls = 0
        self.transform = BraTSPatchAugmentation(
            flip_prob=1.0,
            intensity_prob=0.0,
        )

    def __call__(self, sample):
        self.calls += 1
        return self.transform(sample)


def main():
    print("=" * 55)
    print("BraTS2020 — Patch Augmentation Integration Test")
    print("=" * 55)

    base_dataset = SyntheticVolumeDataset()

    # Original patch pipeline: no augmentation.
    plain_dataset = RandomPatchDataset(
        base_dataset=base_dataset,
        patch_size=(8, 8, 8),
        pos_prob=1.0,
        augmentation=None,
    )

    tracker = TrackingAugmentation()

    # New patch pipeline: augmentation enabled.
    augmented_dataset = RandomPatchDataset(
        base_dataset=base_dataset,
        patch_size=(8, 8, 8),
        pos_prob=1.0,
        augmentation=tracker,
    )

    # Reset RNG to select the same patch in both cases.
    torch.manual_seed(42)
    original = plain_dataset[0]

    torch.manual_seed(42)
    augmented = augmented_dataset[0]

    # 1. Check augmentation invocation.
    assert tracker.calls == 1
    print("[PASS] Augmentation called exactly once")

    # 2. Check patch shape.
    assert augmented["image"].shape == (4, 8, 8, 8)
    assert augmented["label"].shape == (8, 8, 8)
    print("[PASS] Patch shapes correct")

    # 3. Compare augmented patch to expected spatial flips.
    expected_image = torch.flip(
        original["image"], dims=(1, 2)
    )
    expected_label = torch.flip(
        original["label"], dims=(0, 1)
    )

    assert torch.equal(augmented["image"], expected_image)
    assert torch.equal(augmented["label"], expected_label)

    print("[PASS] Spatial augmentation applied correctly")

    # 4. Check metadata.
    assert augmented["subject_id"] == original["subject_id"]
    assert augmented["center_type"] == original["center_type"]
    assert augmented["crop_start"] == original["crop_start"]

    print("[PASS] Patch metadata preserved")

    # 5. Check labels and numerical values.
    labels = set(torch.unique(augmented["label"]).tolist())
    assert labels.issubset({0, 1, 2, 3})
    assert torch.isfinite(augmented["image"]).all()

    print("[PASS] Labels and MRI values valid")

    
    # 6. Verify the original pipeline remains unchanged.
    torch.manual_seed(42)
    repeated_plain = plain_dataset[0]

    assert torch.equal(
        original["image"], repeated_plain["image"]
    )
    assert torch.equal(
        original["label"], repeated_plain["label"]
    )
    assert original["crop_start"] == repeated_plain["crop_start"]
    assert tracker.calls == 1

    print("[PASS] Pipeline supports optional augmentation")


    print()
    print("Original image shape:", tuple(original["image"].shape))
    print("Augmented image shape:", tuple(augmented["image"].shape))
    print("Unique labels:", sorted(labels))
    print("Augmentation calls:", tracker.calls)

    print()
    print("=" * 55)
    print("ALL PATCH AUGMENTATION INTEGRATION TESTS PASSED")
    print("=" * 55)


if __name__ == "__main__":
    main()
