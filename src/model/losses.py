"""
Loss Functions for Oil Spill Semantic Segmentation (PRD Section 6).
Implements Dice Loss and Combo Loss (Dice Loss + Binary Cross-Entropy).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary semantic segmentation.
    Formula: Loss = 1 - (2 * |Y_hat * Y| + eps) / (|Y_hat| + |Y| + eps)
    """

    def __init__(self, smooth: float = 1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        
        # Flatten spatial dimensions: (B, C, H, W) -> (B, -1)
        probs_flat = probs.view(probs.size(0), -1)
        targets_flat = targets.view(targets.size(0), -1).float()

        intersection = (probs_flat * targets_flat).sum(dim=1)
        cardinality = probs_flat.sum(dim=1) + targets_flat.sum(dim=1)

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        return (1.0 - dice).mean()


class ComboLoss(nn.Module):
    """
    Combo Loss: Weighted combination of Binary Cross-Entropy and Dice Loss.
    Combines pixel-level log-loss with region-level overlap optimization to address class imbalance.
    """

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, pos_weight: float = 2.0):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.dice_loss = DiceLoss()
        self.pos_weight = torch.tensor([pos_weight]) if pos_weight > 1.0 else None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets_float = targets.float()
        
        # Binary Cross Entropy with Logits
        if self.pos_weight is not None and logits.is_cuda:
            pos_weight = self.pos_weight.to(logits.device)
            bce = F.binary_cross_entropy_with_logits(logits, targets_float, pos_weight=pos_weight)
        else:
            bce = F.binary_cross_entropy_with_logits(logits, targets_float)

        dice = self.dice_loss(logits, targets_float)
        return self.bce_weight * bce + self.dice_weight * dice
