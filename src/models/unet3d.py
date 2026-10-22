
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


def build_unet3d():
    """
    Tạo mô hình MONAI 3D U-Net baseline cho BraTS2020.

    Input:
        [B, 4, H, W, D]

    Output:
        [B, 4, H, W, D]

    Bốn kênh đầu vào tương ứng:
        T1, T1ce, T2, FLAIR.

    Bốn kênh đầu ra tương ứng:
        Background và ba nhãn khối u đã remap.
    """

    model = UNet(
        spatial_dims=3,
        in_channels=4,
        out_channels=4,
        channels=(16, 32, 64, 128),
        strides=(2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
    )

    return model


def build_loss():
    """
    Tạo Dice + Cross Entropy loss.

    Target yêu cầu:
        [B, 1, H, W, D]

    Các giá trị nhãn:
        0, 1, 2, 3.
    """

    return DiceCELoss(
        include_background=False,
        to_onehot_y=True,
        softmax=True,
    )
