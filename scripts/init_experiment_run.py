
import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "unet3d_optimization_protocol.json"
OUTPUT_ROOT = ROOT / "outputs" / "runs"


def git_info():
    def git(*args):
        result = subprocess.run(
            ["git", *args],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("branch", "--show-current"),
            "working_tree_dirty": bool(
                git("status", "--porcelain")
            ),
        }
    except (FileNotFoundError, subprocess.CalledProcessError):
        return {
            "commit": None,
            "branch": None,
            "working_tree_dirty": None,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a BraTS2020 experiment run."
    )
    parser.add_argument(
        "--experiment",
        required=True,
        help="Experiment ID: M2, M3, M4a, M4b or M5",
    )
    args = parser.parse_args()

    # Load study protocol.
    with CONFIG_PATH.open("r", encoding="utf-8") as f:
        protocol = json.load(f)

    experiments = {
        item["id"]: item
        for item in protocol["experiments"]
    }

    if args.experiment not in experiments:
        raise ValueError(
            f"Unknown experiment: {args.experiment}. "
            f"Available: {list(experiments)}"
        )

    experiment = experiments[args.experiment]

    # Unique run ID, with UTC timestamp.
    timestamp = datetime.now(timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )

    run_id = f"{args.experiment}_{timestamp}"
    run_dir = OUTPUT_ROOT / run_id

    # Never overwrite a previous experiment.
    run_dir.mkdir(parents=True, exist_ok=False)

    for folder in ("checkpoints", "metrics", "figures"):
        (run_dir / folder).mkdir()

    # Preserve the exact study protocol.
    protocol_bytes = CONFIG_PATH.read_bytes()
    protocol_hash = hashlib.sha256(
        protocol_bytes
    ).hexdigest()

    snapshot_path = run_dir / "protocol_snapshot.json"
    snapshot_path.write_bytes(protocol_bytes)

    metadata = {
        "run_id": run_id,
        "experiment_id": experiment["id"],
        "experiment_name": experiment["name"],
        "change": experiment["change"],
        "status": "initialized",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "study": {
            "name": protocol["study_name"],
            "protocol_version": protocol["protocol_version"],
            "protocol_sha256": protocol_hash,
            "reference_model": protocol["reference_model"],
        },

        "training": protocol["controlled_training"],
        "evaluation": protocol["evaluation"],
        "dataset": protocol["dataset"],

        "environment": {
            "python_version": sys.version.split()[0],
            "platform": platform.platform(),
            "git": git_info(),
        },

        "results": {
            "training_completed": False,
            "validation_completed": False,
            "best_epoch": None,
            "best_validation_dice": None,
        },
    }

    metadata_path = run_dir / "run_metadata.json"

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print("=" * 55)
    print("BraTS2020 — Experiment Run Initialization")
    print("=" * 55)
    print("[PASS] Experiment:", experiment["id"])
    print("[PASS] Name:", experiment["name"])
    print("[PASS] Configuration snapshot saved")
    print("[PASS] Git and environment metadata recorded")
    print("[PASS] Output folders created")
    print()
    print("Run ID:", run_id)
    print("Directory:", run_dir)
    print("Status: initialized — no training performed")
    print("=" * 55)


if __name__ == "__main__":
    main()
