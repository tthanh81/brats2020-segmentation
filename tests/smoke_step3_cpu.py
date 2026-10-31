"""CPU-only smoke checks for Step 3 candidate modules.

Run from the repository root using pinned dependencies:
    python tests/smoke_step3_cpu.py

Does NOT train, load MRI, read held-out test, or use Kaggle artifacts.
"""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def test_m4b():
    from src.data.patches_class_balanced import ClassBalancedRandomPatchDataset

    class SyntheticDataset:
        def __len__(self):
            return 1
        def __getitem__(self, index):
            assert index == 0
            image = torch.zeros(4, 20, 20, 20)
            label = torch.zeros(20, 20, 20, dtype=torch.long)
            label[6:12, 6:12, 6:12] = 1
            label[12:16, 6:12, 6:12] = 2
            label[6:12, 12:16, 6:12] = 3
            return dict(image=image, label=label, subject_id="synthetic")

    torch.manual_seed(42)
    ds = ClassBalancedRandomPatchDataset(SyntheticDataset(), patch_size=(8, 8, 8), pos_prob=1.0)
    for _ in range(3):
        sample = ds[0]
        assert sample["image"].shape == (4, 8, 8, 8)
        assert sample["label"].shape == (8, 8, 8)
        assert sample["center_type"] == "positive"
        assert all(0 <= n <= 12 for n in sample["crop_start"])
    print("[PASS] M4b sampler: synthetic patch shapes, positive center, crop bounds")


def test_m4a():
    from src.models.m4a_combined_loss import M4ACombinedLoss, M4A_LOSS_CONFIG

    model = M4ACombinedLoss(M4A_LOSS_CONFIG)
    torch.manual_seed(42)
    logits = torch.randn(1, 4, 8, 8, 8, requires_grad=True)
    labels = torch.randint(0, 4, (1, 1, 8, 8, 8), dtype=torch.long)
    loss = model(logits, labels)
    assert torch.isfinite(loss).item(), loss
    loss.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all().item()
    print("[PASS] M4a combined loss: forward/backward finite on synthetic data")


def test_m6():
    from src.models.swin_unetr import build_swin_unetr

    model = build_swin_unetr()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params == 15_705_646, f"Unexpected SwinUNETR parameter count: {n_params}"
    print(f"[PASS] M6 SwinUNETR factory: parameter count {n_params:,}")
    del model


if __name__ == "__main__":
    print("Checking candidate modules WITHOUT MRI, checkpoint inference, or training")
    print("Torch:", torch.__version__)
    import monai
    print("MONAI:", monai.__version__)
    test_m4b()
    test_m4a()
    test_m6()
    print("ALL STEP3 CPU SMOKE CHECKS PASS")
