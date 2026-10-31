
import csv
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    ROOT / "configs" / "unet3d_optimization_protocol.json"
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    print("=" * 55)
    print("BraTS2020 — Experiment Protocol Validation")
    print("=" * 55)

    # 1. Read experiment configuration
    require(
        CONFIG_PATH.is_file(),
        f"Configuration not found: {CONFIG_PATH}"
    )

    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)

    require(
        config["study_name"] == "brats2020_unet3d_optimization",
        "Unexpected study_name"
    )
    require(
        config["protocol_version"] == 1,
        "Unsupported protocol_version"
    )

    print("[PASS] JSON configuration loaded")

    # 2. Validate patient-level split
    dataset = config["dataset"]
    manifest_path = ROOT / dataset["split_manifest"]

    require(manifest_path.is_file(), "Split manifest not found")

    with manifest_path.open(
        "r", encoding="utf-8", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    subject_ids = [r["subject_id"] for r in rows]
    require(
        len(subject_ids) == len(set(subject_ids)),
        "Duplicate patient IDs detected"
    )

    split_counts = Counter(r["split"] for r in rows)

    expected_counts = {
        "train": dataset["train_subjects"],
        "internal_validation": dataset["validation_subjects"],
        "test": dataset["test_subjects"]
    }

    require(
        dict(split_counts) == expected_counts,
        f"Split mismatch: {dict(split_counts)}"
    )

    require(
        all(int(r["seed"]) == dataset["seed"] for r in rows),
        "Inconsistent split seeds"
    )

    require(
        dataset["seed"] == 42,
        "This protocol requires the seed-42 split"
    )

    require(
        "not use" in dataset["test_policy"].lower(),
        "Test policy must prohibit test-based selection"
    )

    print("[PASS] Patient-level split verified")
    print("       Train:", split_counts["train"])
    print("       Validation:", split_counts["internal_validation"])
    print("       Test:", split_counts["test"])

    # 3. Validate reference model
    reference = config["reference_model"]

    require(reference["model_id"] == "M1", "Invalid reference")
    require(reference["epochs_completed"] == 20, "Invalid epoch")
    require(reference["best_epoch"] == 20, "Invalid best epoch")

    require(
        abs(reference["validation_mean_dice"] - 0.7575443313042411)
        < 1e-9,
        "Reference Dice mismatch"
    )

    print("[PASS] Reference model M1 verified")

    # 4. Validate training protocol
    train = config["controlled_training"]

    require(
        train["starting_checkpoint_epoch"] == 10,
        "Experiments must start from epoch 10"
    )
    require(train["final_epoch"] == 20, "Invalid final epoch")
    require(
        train["validation_epochs"] == [15, 20],
        "Invalid validation schedule"
    )
    require(train["optimizer"] == "AdamW", "Invalid optimizer")
    require(train["patch_size"] == [96, 96, 96], "Invalid patch size")
    require(train["batch_size"] == 1, "Invalid batch size")
    require(
        abs(train["learning_rate"] - 1e-4) < 1e-12,
        "Invalid learning rate"
    )
    require(train["use_amp"] is True, "AMP setting mismatch")

    print("[PASS] Controlled training protocol verified")

    # 5. Validate evaluation policy
    evaluation = config["evaluation"]

    require(
        evaluation["primary_metric"] == "mean_validation_dice",
        "Unexpected primary metric"
    )
    require(
        evaluation["regions"] == ["WT", "TC", "ET"],
        "Invalid segmentation regions"
    )
    require(
        evaluation["aggregation"] == "subject_macro_average",
        "Invalid Dice aggregation"
    )
    require(
        evaluation["compare_same_validation_subjects"] is True,
        "Paired validation comparison must be enabled"
    )

    print("[PASS] Evaluation protocol verified")

    # 6. Validate experiment definitions
    experiments = config["experiments"]

    expected_ids = {"M2", "M3", "M4a", "M4b", "M5"}
    actual_ids = [e["id"] for e in experiments]

    require(
        len(actual_ids) == len(set(actual_ids)),
        "Duplicate experiment IDs"
    )
    require(
        set(actual_ids) == expected_ids,
        "Unexpected experiment definitions"
    )

    for experiment in experiments:
        require(
            experiment["status"] == "planned",
            f"Unexpected status: {experiment['id']}"
        )

    print("[PASS] Experiment definitions verified")

    # 7. Validate output policy
    output = config["output_policy"]

    require(
        output["commit_checkpoints_to_git"] is False,
        "Checkpoints must not be committed to Git"
    )
    require(
        output["save_best_checkpoint"] is True,
        "Best checkpoint saving must be enabled"
    )
    require(
        output["save_training_history"] is True,
        "Training history saving must be enabled"
    )
    require(
        output["save_subject_level_validation"] is True,
        "Subject-level validation is required"
    )

    print("[PASS] Output policy verified")

    print()
    print("=" * 55)
    print("ALL CHECKS PASSED")
    print("Study:", config["study_name"])
    print("Experiments:", ", ".join(actual_ids))
    print("Reference validation Dice:",
          reference["validation_mean_dice"])
    print("=" * 55)


if __name__ == "__main__":
    main()
