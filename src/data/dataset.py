
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

from .preprocessing import (
    MODALITIES,
    normalize_mri,
    remap_brats_labels,
)


class BraTSVolumeDataset(Dataset):

    def __init__(self, manifest, data_root):
        self.manifest = manifest.reset_index(drop=True).copy()
        self.data_root = Path(data_root)

        required = MODALITIES + ["seg", "subject_id"]

        missing_columns = set(required) - set(
            self.manifest.columns
        )

        if missing_columns:
            raise ValueError(
                f"Manifest is missing columns: {missing_columns}"
            )

        # Kiểm tra đường dẫn trước khi đọc volume.
        for _, row in self.manifest.iterrows():
            for field in MODALITIES + ["seg"]:

                rel_path = row[field]

                if pd.isna(rel_path) or not str(rel_path).strip():
                    raise ValueError(
                        f"{row['subject_id']} is missing {field}"
                    )

                path = self.data_root / str(rel_path)

                if not path.is_file():
                    raise FileNotFoundError(
                        f"File not found: {path}"
                    )

    def __len__(self):
        return len(self.manifest)

    def __getitem__(self, index):

        row = self.manifest.iloc[index]
        subject_id = row["subject_id"]

        # 1. Đọc bốn modality MRI.
        nifti_images = {}

        for modality in MODALITIES:
            path = self.data_root / str(row[modality])
            nifti_images[modality] = nib.load(str(path))

        # Dùng FLAIR làm ảnh tham chiếu.
        reference = nifti_images["flair"]
        reference_shape = tuple(reference.shape)
        reference_affine = reference.affine

        # 2. Kiểm tra hình học của bốn modality.
        for modality, nii_img in nifti_images.items():

            if tuple(nii_img.shape) != reference_shape:
                raise ValueError(
                    f"{subject_id}: shape mismatch in {modality}"
                )

            if not np.allclose(
                nii_img.affine,
                reference_affine,
                rtol=0,
                atol=1e-4,
            ):
                raise ValueError(
                    f"{subject_id}: affine mismatch in {modality}"
                )

        # 3. Chuẩn hóa từng modality.
        channels = []

        for modality in MODALITIES:

            volume = nifti_images[modality].get_fdata(
                dtype=np.float32,
                caching="unchanged",
            )

            normalized = normalize_mri(
                volume,
                modality_name=f"{subject_id}_{modality}",
            )

            channels.append(normalized)

        image_array = np.stack(channels, axis=0)

        # 4. Đọc segmentation mask.
        seg_path = self.data_root / str(row["seg"])
        seg_img = nib.load(str(seg_path))

        if tuple(seg_img.shape) != reference_shape:
            raise ValueError(
                f"{subject_id}: segmentation shape mismatch"
            )

        if not np.allclose(
            seg_img.affine,
            reference_affine,
            rtol=0,
            atol=1e-4,
        ):
            raise ValueError(
                f"{subject_id}: segmentation affine mismatch"
            )

        seg_raw = seg_img.get_fdata(
            dtype=np.float32,
            caching="unchanged",
        )

        label_array = remap_brats_labels(seg_raw)

        # 5. Chuyển sang PyTorch Tensor.
        image_tensor = torch.from_numpy(
            np.ascontiguousarray(
                image_array,
                dtype=np.float32,
            )
        )

        label_tensor = torch.from_numpy(
            np.ascontiguousarray(
                label_array,
                dtype=np.uint8,
            )
        )

        return {
            "image": image_tensor,
            "label": label_tensor,
            "subject_id": subject_id,
        }
