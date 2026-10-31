"""M4b class-balanced tumor-center patch sampling.

Controlled change relative to RandomPatchDataset:
choose a present tumor class uniformly before selecting
a voxel inside that class.

The random branch and crop behavior remain unchanged.
"""

import torch

from src.data.patches import RandomPatchDataset


class ClassBalancedRandomPatchDataset(RandomPatchDataset):
    """Class-balanced tumor-center sampler for M4b."""

    def __getitem__(self, index):

        # 1. Load preprocessed volume.
        sample = self.base_dataset[index]

        image = sample["image"]
        label = sample["label"]

        spatial_shape = tuple(label.shape)

        # 2. Same validation as baseline.
        if image.ndim != 4 or label.ndim != 3:
            raise ValueError(
                f"Invalid shapes: "
                f"image={image.shape}, "
                f"label={label.shape}"
            )

        if tuple(image.shape[1:]) != spatial_shape:
            raise ValueError(
                "Image and label spatial shapes differ."
            )

        for patch_dim, volume_dim in zip(
            self.patch_size,
            spatial_shape,
        ):
            if patch_dim > volume_dim:
                raise ValueError(
                    f"Patch {self.patch_size} exceeds "
                    f"volume {spatial_shape}."
                )

        # 3. M4b-specific class-balanced selection.
        available_classes = [
            class_id
            for class_id in (1, 2, 3)
            if bool((label == class_id).any().item())
        ]

        use_positive_center = (
            len(available_classes) > 0
            and torch.rand(()).item() < self.pos_prob
        )

        # 4. Select center.
        if use_positive_center:

            class_index = torch.randint(
                low=0,
                high=len(available_classes),
                size=(1,),
            ).item()

            selected_class = available_classes[
                class_index
            ]

            class_coords = torch.nonzero(
                label == selected_class,
                as_tuple=False,
            )

            chosen = torch.randint(
                low=0,
                high=class_coords.shape[0],
                size=(1,),
            ).item()

            center = class_coords[chosen].tolist()
            center_type = "positive"

        else:

            # Same random branch as baseline.
            center = [
                torch.randint(0, size, (1,)).item()
                for size in spatial_shape
            ]

            center_type = "random"

        # 5. Same crop boundaries as baseline.
        starts = []

        for c, p, size in zip(
            center,
            self.patch_size,
            spatial_shape,
        ):

            start = max(
                0,
                min(c - p // 2, size - p),
            )

            starts.append(start)

        h0, w0, d0 = starts
        ph, pw, pd = self.patch_size

        # 6. Same image/label cropping.
        image_patch = image[
            :,
            h0:h0 + ph,
            w0:w0 + pw,
            d0:d0 + pd,
        ]

        label_patch = label[
            h0:h0 + ph,
            w0:w0 + pw,
            d0:d0 + pd,
        ]

        result = {
            "image": image_patch.contiguous(),
            "label": label_patch.contiguous(),
            "subject_id": sample["subject_id"],
            "center_type": center_type,
            "crop_start": tuple(starts),
        }

        if self.augmentation is not None:
            result = self.augmentation(result)

        return result
