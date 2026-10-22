import torch


def region_masks(label):
    """
    Create BraTS2020 evaluation masks.

    Remapped labels:
        0: Background
        1: NCR / NET
        2: Edema
        3: Enhancing tumor

    Regions:
        WT = {1, 2, 3}
        TC = {1, 3}
        ET = {3}
    """

    label = torch.as_tensor(label)

    if label.ndim != 3:
        raise ValueError(
            f"Expected 3D label, got {tuple(label.shape)}"
        )

    valid = (label >= 0) & (label <= 3)

    if not torch.all(valid):
        raise ValueError("Unexpected segmentation labels")

    return {
        "WT": label > 0,
        "TC": (label == 1) | (label == 3),
        "ET": label == 3,
    }


def binary_dice(pred_mask, true_mask):
    """
    Compute Dice score between two binary masks.

    Convention:
        Both masks empty -> Dice = 1
        One mask empty  -> Dice = 0
    """

    pred_mask = pred_mask.bool()
    true_mask = true_mask.bool()

    if pred_mask.shape != true_mask.shape:
        raise ValueError("Prediction and target shapes differ")

    intersection = torch.logical_and(
        pred_mask, true_mask
    ).sum().item()

    pred_count = pred_mask.sum().item()
    true_count = true_mask.sum().item()

    denominator = pred_count + true_count

    if denominator == 0:
        return 1.0

    return 2.0 * intersection / denominator


def brats_dice_scores(pred_label, true_label):
    """
    Compute Dice scores for WT, TC and ET.

    Inputs:
        pred_label: [H, W, D]
        true_label: [H, W, D]

    Outputs:
        {"WT": float, "TC": float, "ET": float}
    """

    pred_label = torch.as_tensor(pred_label)
    true_label = torch.as_tensor(true_label)

    if pred_label.shape != true_label.shape:
        raise ValueError("Prediction and target shapes differ")

    pred_regions = region_masks(pred_label)
    true_regions = region_masks(true_label)

    return {
        region: binary_dice(
            pred_regions[region],
            true_regions[region],
        )
        for region in ("WT", "TC", "ET")
    }