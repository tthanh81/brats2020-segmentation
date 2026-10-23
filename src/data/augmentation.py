
import torch


class BraTSPatchAugmentation:
    """
    Training-only augmentation for BraTS2020 3D patches.

    image: [4, H, W, D] float tensor
    label: [H, W, D] integer tensor

    Spatial transformations are synchronized
    between all MRI modalities and the mask.
    """

    def __init__(
        self,
        flip_prob=0.5,
        intensity_prob=0.5,
        scale_range=(0.9, 1.1),
        shift_range=(-0.1, 0.1),
    ):
        self.flip_prob = flip_prob
        self.intensity_prob = intensity_prob
        self.scale_range = scale_range
        self.shift_range = shift_range

        if not 0 <= flip_prob <= 1:
            raise ValueError("Invalid flip probability")

        if not 0 <= intensity_prob <= 1:
            raise ValueError("Invalid intensity probability")

        if scale_range[0] <= 0 or scale_range[0] > scale_range[1]:
            raise ValueError("Invalid intensity scale range")

        if shift_range[0] > shift_range[1]:
            raise ValueError("Invalid intensity shift range")

    @staticmethod
    def uniform(low, high):
        return low + (high - low) * torch.rand(()).item()

    def __call__(self, sample):
        image = sample["image"].clone()
        label = sample["label"].clone()

        if image.ndim != 4 or label.ndim != 3:
            raise ValueError("Unexpected MRI or mask dimensions")

        if tuple(image.shape[1:]) != tuple(label.shape):
            raise ValueError("Image/mask shape mismatch")

        # Spatial flip 1: dimension H
        if torch.rand(()).item() < self.flip_prob:
            image = torch.flip(image, dims=(1,))
            label = torch.flip(label, dims=(0,))

        # Spatial flip 2: dimension W
        if torch.rand(()).item() < self.flip_prob:
            image = torch.flip(image, dims=(2,))
            label = torch.flip(label, dims=(1,))

        # Remember foreground to preserve zero background.
        foreground = image != 0

        # Intensity scaling, per MRI modality.
        if torch.rand(()).item() < self.intensity_prob:
            scales = torch.tensor(
                [
                    self.uniform(*self.scale_range)
                    for _ in range(image.shape[0])
                ],
                dtype=image.dtype,
                device=image.device,
            ).view(-1, 1, 1, 1)

            image = image * scales

        # Intensity shifting, per MRI modality.
        if torch.rand(()).item() < self.intensity_prob:
            shifts = torch.tensor(
                [
                    self.uniform(*self.shift_range)
                    for _ in range(image.shape[0])
                ],
                dtype=image.dtype,
                device=image.device,
            ).view(-1, 1, 1, 1)

            image = torch.where(
                foreground,
                image + shifts,
                torch.zeros_like(image),
            )

        result = dict(sample)
        result["image"] = image.contiguous()
        result["label"] = label.contiguous()

        return result
