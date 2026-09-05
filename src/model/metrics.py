"""
Segmentation Evaluation Metrics (PRD Section 16).
Computes Dice Score, IoU, Precision, Recall, and Specificity with numerical stability.
"""
from dataclasses import dataclass
from typing import Dict
import numpy as np
import torch


@dataclass
class SegmentationMetrics:
    """Container for quantitative segmentation benchmark metrics."""
    dice_score: float
    iou: float
    precision: float
    recall: float
    specificity: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "dice_score": round(self.dice_score, 4),
            "iou": round(self.iou, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "specificity": round(self.specificity, 4),
        }


def calculate_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    eps: float = 1e-6,
) -> SegmentationMetrics:
    """
    Calculate segmentation metrics from prediction probabilities and binary targets.
    
    Args:
        predictions: Probability array in [0.0, 1.0] or binary array.
        targets: Ground truth binary array (0 or 1).
        threshold: Binarization threshold.
        eps: Small epsilon to prevent division by zero.
        
    Returns:
        SegmentationMetrics instance.
    """
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    bin_pred = (predictions >= threshold).astype(np.uint8).flatten()
    bin_target = (targets >= 0.5).astype(np.uint8).flatten()

    tp = np.sum((bin_pred == 1) & (bin_target == 1))
    fp = np.sum((bin_pred == 1) & (bin_target == 0))
    fn = np.sum((bin_pred == 0) & (bin_target == 1))
    tn = np.sum((bin_pred == 0) & (bin_target == 0))

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall = (tp + eps) / (tp + fn + eps)
    specificity = (tn + eps) / (tn + fp + eps)

    return SegmentationMetrics(
        dice_score=float(dice),
        iou=float(iou),
        precision=float(precision),
        recall=float(recall),
        specificity=float(specificity),
    )
