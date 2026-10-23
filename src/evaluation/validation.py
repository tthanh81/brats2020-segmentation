"""Full-volume BraTS2020 evaluation using MONAI sliding-window inference."""

import time

import pandas as pd
import torch
from monai.inferers import sliding_window_inference

from src.evaluation.metrics import brats_dice_scores

def validate_model(
    model,
    dataset,
    device,
    roi_size=(96, 96, 96),
    overlap=0.25,
    max_subjects=None,
):
    """
    Validate BraTS2020 segmentation on full MRI volumes.

    Returns:
        summary: Mean Dice scores and timing information.
        results_df: Per-subject Dice scores.
    """

    model.eval()

    num_subjects = len(dataset)

    if max_subjects is not None:
        num_subjects = min(num_subjects, max_subjects)

    results = []
    start_total = time.perf_counter()

    with torch.inference_mode():

        for i in range(num_subjects):

            start_subject = time.perf_counter()

            # 1. Load full MRI volume.
            sample = dataset[i]

            image = sample["image"].unsqueeze(0)
            target = sample["label"]

            # 2. Full-volume sliding-window inference.
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(device.type == "cuda"),
            ):

                logits = sliding_window_inference(
                    inputs=image,
                    roi_size=roi_size,
                    sw_batch_size=1,
                    predictor=model,
                    overlap=overlap,
                    mode="gaussian",
                    sw_device=device,
                    device=torch.device("cpu"),
                )

            # 3. Convert logits to segmentation labels.
            prediction = (
                logits.argmax(dim=1)
                .squeeze(0)
                .to(torch.uint8)
            )

            assert prediction.shape == target.shape

            # 4. Calculate Dice.
            scores = brats_dice_scores(
                pred_label=prediction,
                true_label=target,
            )

            # 5. Store per-subject results.
            elapsed = time.perf_counter() - start_subject

            results.append({
                "subject_id": sample["subject_id"],
                "WT": scores["WT"],
                "TC": scores["TC"],
                "ET": scores["ET"],
                "ET_present": bool((target == 3).any().item()),
                "time_seconds": elapsed,
            })

            # Release full-volume tensors.
            del logits, prediction, image, target

            # Print progress every 5 subjects.
            if (i + 1) % 5 == 0 or (i + 1) == num_subjects:

                print(
                    f"Validated {i + 1}/{num_subjects} subjects"
                )

    # 6. Aggregate metrics.
    results_df = pd.DataFrame(results)

    mean_scores = results_df[
        ["WT", "TC", "ET"]
    ].mean()

    total_time = time.perf_counter() - start_total

    summary = {
        "WT": float(mean_scores["WT"]),
        "TC": float(mean_scores["TC"]),
        "ET": float(mean_scores["ET"]),
        "mean_dice": float(mean_scores.mean()),
        "num_subjects": len(results_df),
        "total_time_seconds": total_time,
        "avg_time_seconds": total_time / len(results_df),
        "num_empty_ET_targets": int(
            (~results_df["ET_present"]).sum()
        ),
    }

    return summary, results_df
