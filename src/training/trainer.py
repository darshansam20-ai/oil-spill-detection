"""
Offline Model Training and Versioning Pipeline (PRD Section 6 & Section 16).
Trains ConvNeXt-Tiny + U-Net using Combo Loss (Dice + BCE) and AdamW optimizer.
Saves versioned model artifacts and training history. Strictly offline.
"""
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from src.config.constants import CURRENT_MODEL_VERSION, MODEL_ARCHITECTURE
from src.config.settings import settings
from src.model.convnext_unet import ConvNeXtTinyUNet
from src.model.losses import ComboLoss
from src.model.metrics import calculate_metrics, SegmentationMetrics
from src.preprocessing.sar_preprocessor import PreprocessingConfig
from src.utils.device import get_default_device
from src.utils.logger import get_logger

logger = get_logger("training.trainer")


@dataclass
class TrainingConfig:
    """Hyperparameters and configuration for offline model training."""
    model_version: str = CURRENT_MODEL_VERSION
    architecture: str = MODEL_ARCHITECTURE
    batch_size: int = 8
    epochs: int = 25
    learning_rate: float = 2e-4
    weight_decay: float = 1e-2
    bce_weight: float = 0.5
    dice_weight: float = 0.5
    patch_size: int = 256
    device: Optional[str] = None
    mixed_precision: bool = True
    save_dir: Path = settings.paths.checkpoints_dir
    metadata_dir: Path = settings.paths.metadata_dir


class ModelTrainer:
    """Offline trainer for ConvNeXt-Tiny + U-Net segmentation model."""

    def __init__(
        self,
        model: Optional[ConvNeXtTinyUNet] = None,
        config: Optional[TrainingConfig] = None,
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ):
        self.config = config or TrainingConfig()
        self.preprocessing_config = preprocessing_config or PreprocessingConfig()
        self.device = torch.device(self.config.device) if self.config.device else get_default_device()
        
        self.model = model or ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=True)
        self.model.to(self.device)

        self.criterion = ComboLoss(
            bce_weight=self.config.bce_weight,
            dice_weight=self.config.dice_weight,
        ).to(self.device)

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=self.config.epochs, eta_min=1e-6)
        self.scaler = torch.amp.GradScaler('cuda') if (self.config.mixed_precision and self.device.type == "cuda") else None

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_dice": [],
            "val_iou": [],
            "val_precision": [],
            "val_recall": [],
        }

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run one training epoch with AdamW optimizer and mixed precision."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(train_loader)

        for batch_idx, (images, masks, _) in enumerate(train_loader):
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            self.optimizer.zero_grad()

            if self.scaler is not None:
                with torch.amp.autocast('cuda'):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(images)
                loss = self.criterion(logits, masks)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

            total_loss += loss.item()

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, SegmentationMetrics]:
        """Evaluate model on validation split (inference-only)."""
        self.model.eval()
        total_loss = 0.0
        num_batches = len(val_loader)
        
        all_preds = []
        all_targets = []

        for images, masks, _ in val_loader:
            images = images.to(self.device, dtype=torch.float32)
            masks = masks.to(self.device, dtype=torch.float32)

            logits = self.model(images)
            loss = self.criterion(logits, masks)
            total_loss += loss.item()

            probs = torch.sigmoid(logits).cpu().numpy()
            targets = masks.cpu().numpy()
            all_preds.append(probs)
            all_targets.append(targets)

        avg_loss = total_loss / max(1, num_batches)
        if all_preds:
            cat_preds = np.concatenate(all_preds, axis=0)
            cat_targets = np.concatenate(all_targets, axis=0)
            metrics = calculate_metrics(cat_preds, cat_targets, threshold=0.5)
        else:
            metrics = SegmentationMetrics(0.0, 0.0, 0.0, 0.0, 0.0)

        return avg_loss, metrics

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict[str, Any]:
        """
        Execute full training loop, track metrics, and checkpoint best model.
        """
        logger.info(f"Starting offline training for {self.config.epochs} epochs on {self.device}...")
        best_val_dice = 0.0
        best_epoch = 0
        self.config.save_dir.mkdir(parents=True, exist_ok=True)
        self.config.metadata_dir.mkdir(parents=True, exist_ok=True)

        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            train_loss = self.train_epoch(train_loader)
            val_loss, val_metrics = self.validate_epoch(val_loader)
            self.scheduler.step()
            elapsed = time.time() - start_time

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_dice"].append(val_metrics.dice_score)
            self.history["val_iou"].append(val_metrics.iou)
            self.history["val_precision"].append(val_metrics.precision)
            self.history["val_recall"].append(val_metrics.recall)

            logger.info(
                f"Epoch [{epoch:02d}/{self.config.epochs:02d}] ({elapsed:.1f}s) | "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                f"Val Dice: {val_metrics.dice_score:.4f} | Val IoU: {val_metrics.iou:.4f} | "
                f"Val Prec: {val_metrics.precision:.4f} | Val Rec: {val_metrics.recall:.4f}"
            )

            # Checkpoint best model based on validation Dice score
            if val_metrics.dice_score >= best_val_dice:
                best_val_dice = val_metrics.dice_score
                best_epoch = epoch
                self.save_checkpoint(epoch, val_loss, val_metrics, is_best=True)

        logger.info(f"Training completed! Best Validation Dice: {best_val_dice:.4f} at epoch {best_epoch}.")
        return {
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "history": self.history,
        }

    def save_checkpoint(
        self,
        epoch: int,
        val_loss: float,
        metrics: SegmentationMetrics,
        is_best: bool = False,
    ) -> Path:
        """
        Save versioned model checkpoint and metadata JSON (PRD Section 6 & 19).
        """
        checkpoint_name = f"best_model.pt" if is_best else f"checkpoint_epoch_{epoch:03d}.pt"
        checkpoint_path = self.config.save_dir / checkpoint_name

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_loss": val_loss,
                "metrics": metrics.to_dict(),
                "model_version": self.config.model_version,
                "architecture": self.config.architecture,
            },
            checkpoint_path,
        )

        metadata = {
            "model_version": self.config.model_version,
            "architecture": self.config.architecture,
            "best_epoch": epoch,
            "val_loss": val_loss,
            "metrics": metrics.to_dict(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "preprocessing": asdict(self.preprocessing_config),
            "training_config": {
                "epochs": self.config.epochs,
                "batch_size": self.config.batch_size,
                "learning_rate": self.config.learning_rate,
                "patch_size": self.config.patch_size,
            },
        }

        metadata_path = self.config.metadata_dir / f"model_metadata_{self.config.model_version}.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        # Also save latest metadata
        with open(self.config.metadata_dir / "latest_model_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved model checkpoint -> {checkpoint_path}")
        return checkpoint_path
