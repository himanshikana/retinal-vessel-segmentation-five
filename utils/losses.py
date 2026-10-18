"""
Loss functions for semantic segmentation with severe class imbalance.
Implements Soft Dice Loss and Compound (BCE + Soft Dice) Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    """
    Soft Dice Loss operating directly on logits with sigmoid activation.
    Designed for binary semantic segmentation under severe foreground imbalance.
    """
    def __init__(self, smooth: float = 1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predicted raw logits of shape (B, 1, H, W) or (B, H, W)
            targets: Ground truth binary masks of matching shape with values in {0, 1}
        Returns:
            Scalar Dice loss = 1 - Dice coefficient
        """
        probs = torch.sigmoid(logits)

        # Flatten spatial dimensions per sample in batch
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1)

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


class CompoundLoss(nn.Module):
    """
    Compound Loss combining Binary Cross-Entropy with Logits and Soft Dice Loss.
    Addresses pixel-level cross-entropy gradient flow while directly optimizing
    the global Dice overlap metric for sparse retinal vessel structures.
    """
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.5,
        smooth: float = 1e-6,
        pos_weight: float = None,
    ):
        super(CompoundLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

        pos_weight_tensor = torch.tensor([pos_weight]) if pos_weight is not None else None
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        self.dice_loss = DiceLoss(smooth=smooth)

        # Cache last individual loss components for telemetry
        self.last_bce: float = 0.0
        self.last_dice: float = 0.0

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        return_components: bool = False
    ):
        """
        Args:
            logits: Tensor of shape (B, 1, H, W)
            targets: Tensor of shape (B, 1, H, W)
            return_components: If True, returns (total_loss, bce_loss, dice_loss)
        """
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)

        total = (self.bce_weight * bce) + (self.dice_weight * dice)

        self.last_bce = bce.item()
        self.last_dice = dice.item()

        if return_components:
            return total, bce, dice
        return total
