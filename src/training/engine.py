"""Single-epoch BraTS2020 training with FP16 AMP overflow handling.

Extracted from the successfully executed 10-epoch baseline notebook.
"""

import time
import torch

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    epoch,
    log_interval=25,
):
    """
    Train one epoch with safe AMP overflow handling.

    - Skip gradient clipping if gradients are non-finite
    - Let GradScaler handle Inf/NaN gradients
    - Count successful and skipped optimizer steps separately
    """

    model.train()

    total_loss = 0.0

    optimizer_steps = 0
    skipped_steps = 0

    positive_patches = 0
    tumor_patches = 0

    num_batches = len(loader)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start_time = time.perf_counter()

    for batch_idx, batch in enumerate(loader, start=1):

        # --------------------------------------
        # 1. Prepare input
        # --------------------------------------

        images = batch["image"].to(
            device,
            dtype=torch.float32,
            non_blocking=True,
        )

        targets = (
            batch["label"]
            .unsqueeze(1)
            .to(
                device,
                dtype=torch.int64,
                non_blocking=True,
            )
        )

        if batch["center_type"][0] == "positive":
            positive_patches += 1

        if (targets > 0).any().item():
            tumor_patches += 1

        # --------------------------------------
        # 2. Zero gradients
        # --------------------------------------

        optimizer.zero_grad(set_to_none=True)

        # --------------------------------------
        # 3. Forward + loss
        # --------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=scaler.is_enabled(),
        ):
            logits = model(images)

        loss = criterion(
            logits.float(),
            targets,
        )

        if not torch.isfinite(loss).item():
            raise RuntimeError(
                f"Non-finite loss at epoch {epoch}, "
                f"batch {batch_idx}"
            )

        # --------------------------------------
        # 4. Backward
        # --------------------------------------

        scaler.scale(loss).backward()

        # Records Inf/NaN gradients internally.
        scaler.unscale_(optimizer)

        # --------------------------------------
        # 5. Check gradient finiteness
        # --------------------------------------

        gradients = [
            p.grad
            for p in model.parameters()
            if p.grad is not None
        ]

        assert len(gradients) > 0, (
            "No gradients were produced."
        )

        grad_is_finite = bool(
            torch.stack([
                torch.isfinite(grad).all()
                for grad in gradients
            ]).all().item()
        )

        # Clip ONLY when gradients are finite.
        if grad_is_finite:

            grad_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=12.0,
                error_if_nonfinite=True,
            )

        else:

            grad_norm = None

        # --------------------------------------
        # 6. Optimizer step with AMP
        # --------------------------------------

        scale_before = scaler.get_scale()

        # Automatically skips optimizer.step()
        # if unscale_ detected Inf/NaN.
        scaler.step(optimizer)
        scaler.update()

        scale_after = scaler.get_scale()

        step_skipped = scale_after < scale_before

        # --------------------------------------
        # 7. Track optimizer updates
        # --------------------------------------

        if step_skipped:

            skipped_steps += 1

            print(
                f"[AMP WARNING] "
                f"Epoch {epoch}, batch {batch_idx} | "
                f"Scale: {scale_before:.0f} "
                f"-> {scale_after:.0f}",
                flush=True,
            )

        else:

            if not grad_is_finite:
                raise RuntimeError(
                    "Non-finite gradients were not "
                    "handled by GradScaler."
                )

            optimizer_steps += 1

        # Stop if overflow becomes excessive.
        if skipped_steps > max(10, num_batches // 10):
            raise RuntimeError(
                "Too many AMP-skipped steps. "
                "Numerical stability needs investigation."
            )

        # --------------------------------------
        # 8. Record loss
        # --------------------------------------

        total_loss += loss.item()

        # --------------------------------------
        # 9. Progress logging
        # --------------------------------------

        if (
            batch_idx % log_interval == 0
            or batch_idx == num_batches
        ):

            mean_loss = total_loss / batch_idx

            print(
                f"Epoch {epoch} | "
                f"Batch {batch_idx}/{num_batches} | "
                f"Mean loss: {mean_loss:.4f} | "
                f"AMP skipped: {skipped_steps}",
                flush=True,
            )

    # --------------------------------------
    # 10. Epoch summary
    # --------------------------------------

    if device.type == "cuda":
        torch.cuda.synchronize()

    epoch_time = time.perf_counter() - start_time

    assert optimizer_steps + skipped_steps == num_batches

    print("\nAMP TRAINING STATISTICS")
    print("-" * 50)
    print("Successful optimizer steps:", optimizer_steps)
    print("Skipped optimizer steps:", skipped_steps)
    print("Final AMP scale:", scaler.get_scale())

    return {
        "mean_loss": total_loss / num_batches,
        "epoch_time_seconds": epoch_time,
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "positive_patches": positive_patches,
        "tumor_patches": tumor_patches,
    }
