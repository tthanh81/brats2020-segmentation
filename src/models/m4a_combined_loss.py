"""M4a loss extracted from original Notebook 08, cell 12.

The class implementation below is source-exact; imports and public config
are added for module usage. Not yet rerun against historical checkpoints.
"""
import torch
from torch import nn
from monai.losses import DiceLoss, FocalLoss

M4A_LOSS_CONFIG = {
    "name": "Dice_WeightedCE_Focal",
    "lambda_dice": 1.0,
    "lambda_ce": 1.0,
    "lambda_focal": 0.5,
    "focal_gamma": 2.0,
    "ce_class_weights": [1.0, 1.0, 1.0, 1.5],
    "dice_include_background": False,
    "focal_include_background": True,
    "focal_use_softmax": True,
}

class M4ACombinedLoss(nn.Module):

    def __init__(self, config):
        super().__init__()

        self.lambda_dice = float(
            config["lambda_dice"]
        )

        self.lambda_ce = float(
            config["lambda_ce"]
        )

        self.lambda_focal = float(
            config["lambda_focal"]
        )

        self.dice = DiceLoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
        )

        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(
                config["ce_class_weights"],
                dtype=torch.float32,
            )
        )

        self.focal = FocalLoss(
            include_background=True,
            to_onehot_y=True,
            use_softmax=True,
            gamma=float(config["focal_gamma"]),
        )

    def components(self, logits, targets):

        assert logits.ndim == 5
        assert logits.shape[1] == 4

        assert targets.ndim == 5
        assert targets.shape[1] == 1

        logits = logits.float()
        targets = targets.long()

        loss_dice = self.dice(
            logits, targets
        )

        loss_ce = self.ce(
            logits,
            targets.squeeze(1),
        )

        loss_focal = self.focal(
            logits, targets
        )

        return {
            "dice": loss_dice,
            "weighted_ce": loss_ce,
            "focal": loss_focal,
        }

    def forward(self, logits, targets):

        parts = self.components(
            logits, targets
        )

        total = (
            self.lambda_dice * parts["dice"]
            + self.lambda_ce * parts["weighted_ce"]
            + self.lambda_focal * parts["focal"]
        )

        return total
