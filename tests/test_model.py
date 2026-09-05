"""
Unit tests for ConvNeXt-Tiny + U-Net architecture, losses, and metrics (FR-16, FR-17).
"""
import pytest
import torch
import numpy as np

from src.model.convnext_unet import ConvNeXtTinyUNet
from src.model.losses import DiceLoss, ComboLoss
from src.model.metrics import calculate_metrics


def test_convnext_tiny_unet_forward_shapes():
    model = ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=False)
    model.eval()

    # Input batch of 2 patches: (B, C, H, W)
    dummy_input = torch.randn(2, 1, 256, 256)
    with torch.no_grad():
        logits = model(dummy_input)
        probs = model.predict_probability(dummy_input)

    assert logits.shape == (2, 1, 256, 256)
    assert probs.shape == (2, 1, 256, 256)
    assert probs.min() >= 0.0
    assert probs.max() <= 1.0


def test_combo_loss_numerical_stability():
    loss_fn = ComboLoss(bce_weight=0.5, dice_weight=0.5)

    logits = torch.tensor([[[[10.0, -10.0], [-10.0, 10.0]]]])  # High confidence
    targets = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])

    loss = loss_fn(logits, targets)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)
    assert loss.item() < 0.1  # Very low loss for near-perfect predictions


def test_segmentation_metrics_calculation():
    # 4x4 binary mask test
    targets = np.array([
        [1, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0],
    ])
    # Predictions matching targets exactly
    preds = targets.astype(np.float32)

    metrics = calculate_metrics(preds, targets, threshold=0.5)
    assert np.isclose(metrics.dice_score, 1.0)
    assert np.isclose(metrics.iou, 1.0)
    assert np.isclose(metrics.precision, 1.0)
    assert np.isclose(metrics.recall, 1.0)
    assert np.isclose(metrics.specificity, 1.0)
