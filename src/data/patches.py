
import torch
from torch.utils.data import Dataset


class RandomPatchDataset(Dataset):
    """
    Lấy một patch 3D cho mỗi lần truy cập một bệnh nhân.

    pos_prob:
        Xác suất chọn tâm patch từ voxel có nhãn khối u.

    Đầu vào:
        base_dataset trả về image [C,H,W,D], label [H,W,D].

    Đầu ra:
        image patch [C,ph,pw,pd],
        label patch [ph,pw,pd].
    """

    def __init__(
        self,
        base_dataset,
        patch_size=(96, 96, 96),
        pos_prob=0.5,
        augmentation=None,
    ):
        self.base_dataset = base_dataset
        self.patch_size = tuple(patch_size)
        self.pos_prob = float(pos_prob)
        self.augmentation = augmentation

        if len(self.patch_size) != 3:
            raise ValueError(
                "patch_size must contain 3 dimensions."
            )

        if any(
            not isinstance(p, int) or p <= 0
            for p in self.patch_size
        ):
            raise ValueError(
                "All patch dimensions must be positive integers."
            )

        if not 0 <= self.pos_prob <= 1:
            raise ValueError(
                "pos_prob must be between 0 and 1."
            )

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, index):

        # 1. Lấy MRI volume đã tiền xử lý.
        sample = self.base_dataset[index]

        image = sample["image"]
        label = sample["label"]

        spatial_shape = tuple(label.shape)

        # 2. Kiểm tra shape.
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

        # 3. Tìm các voxel có nhãn khối u.
        positive_coords = torch.nonzero(
            label > 0,
            as_tuple=False,
        )

        use_positive_center = (
            positive_coords.shape[0] > 0
            and torch.rand(()).item() < self.pos_prob
        )

        # 4. Chọn tâm patch.
        if use_positive_center:

            chosen = torch.randint(
                low=0,
                high=positive_coords.shape[0],
                size=(1,),
            ).item()

            center = positive_coords[chosen].tolist()
            center_type = "positive"

        else:
            # Tâm ngẫu nhiên trong toàn bộ volume.
            # Không nhất thiết thuộc lớp background.
            center = [
                torch.randint(0, size, (1,)).item()
                for size in spatial_shape
            ]

            center_type = "random"

        # 5. Tính tọa độ bắt đầu của patch.
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

        # 6. Cắt image và label cùng tọa độ.
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

        # Apply augmentation only when provided.
        if self.augmentation is not None:
            result = self.augmentation(result)

        return result

