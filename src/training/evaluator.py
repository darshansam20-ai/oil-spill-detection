"""
Model Evaluation and Benchmark Reporting Module (PRD Section 16).
Runs comprehensive evaluation on the independent test set split.
Computes Dice Score, IoU, Precision, Recall, Specificity across varying confidence thresholds.
"""
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.model.convnext_unet import ConvNeXtTinyUNet
from src.model.metrics import calculate_metrics, SegmentationMetrics
from src.utils.device import get_default_device
from src.utils.logger import get_logger

logger = get_logger("training.evaluator")


class ModelEvaluator:
    """Evaluates trained segmentation model checkpoints on test data."""

    def __init__(self, model: ConvNeXtTinyUNet, device: Optional[str] = None):
        self.device = torch.device(device) if device else get_default_device()
        self.model = model.to(self.device)
        self.model.eval()

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path: Path, device: Optional[str] = None) -> "ModelEvaluator":
        """Load a trained model evaluator directly from a checkpoint file."""
        model = ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=False)
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        return cls(model=model, device=device)

    @torch.no_grad()
    def evaluate(
        self,
        test_loader: DataLoader,
        threshold: float = 0.5,
    ) -> SegmentationMetrics:
        """
        Run test set evaluation at a specified segmentation threshold.
        """
        self.model.eval()
        all_preds = []
        all_targets = []

        for images, masks, _ in test_loader:
            images = images.to(self.device, dtype=torch.float32)
            probs = self.model.predict_probability(images)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(masks.numpy())

        cat_preds = np.concatenate(all_preds, axis=0)
        cat_targets = np.concatenate(all_targets, axis=0)

        metrics = calculate_metrics(cat_preds, cat_targets, threshold=threshold)
        logger.info(
            f"Test Set Evaluation (Threshold: {threshold:.2f}): "
            f"Dice: {metrics.dice_score:.4f}, IoU: {metrics.iou:.4f}, "
            f"Precision: {metrics.precision:.4f}, Recall: {metrics.recall:.4f}"
        )
        return metrics

    def sweep_thresholds(
        self,
        test_loader: DataLoader,
        thresholds: Optional[List[float]] = None,
    ) -> Dict[float, Dict[str, float]]:
        """
        Evaluate performance across a spectrum of segmentation thresholds to determine optimal operating point.
        """
        threshold_list = thresholds or [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        results = {}
        for thresh in threshold_list:
            m = self.evaluate(test_loader, threshold=thresh)
            results[thresh] = m.to_dict()
        return results
