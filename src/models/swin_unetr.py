"""Post-experiment extraction of the locked M6 architecture constructor.

This module is a new, not historically executed, refactor. Its constructor
arguments match Notebook 11 / Cell 11.5. Check checkpoint load state before use.
"""

from monai.networks.nets import SwinUNETR


def build_swin_unetr():
    return SwinUNETR(
        in_channels=4, out_channels=4, spatial_dims=3,
        feature_size=24, use_checkpoint=True, patch_size=2,
        depths=(2, 2, 2, 2), num_heads=(3, 6, 12, 24),
        norm_name="instance",
    )
