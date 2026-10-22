"""Save and restore BraTS2020 training state at epoch boundaries."""

import os
from pathlib import Path

import torch

def save_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    best_val_dice,
    config,
):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_dice": best_val_dice,

        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),

        "config": config,

        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_states": torch.cuda.get_rng_state_all(),
    }

    # Write to a temporary file before replacing
    # the destination to reduce incomplete writes.
    temporary_path = path.with_name(
        path.name + ".tmp"
    )

    torch.save(checkpoint, temporary_path)

    os.replace(temporary_path, path)

    return path

def load_checkpoint(
    path,
    model,
    optimizer,
    scaler,
    map_location="cpu",
):
    checkpoint = torch.load(
        path,
        map_location=map_location,
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    optimizer.load_state_dict(
        checkpoint["optimizer_state_dict"]
    )

    scaler.load_state_dict(
        checkpoint["scaler_state_dict"]
    )

    return checkpoint
