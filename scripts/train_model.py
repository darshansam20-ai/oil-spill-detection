import argparse
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader

from src.config.settings import settings
from src.dataset.split_manager import SplitManager
from src.dataset.dataset_loader import Sentinel1OilSpillDataset
from src.model.convnext_unet import ConvNeXtTinyUNet
from src.training.trainer import ModelTrainer, TrainingConfig
from src.utils.logger import get_logger

logger = get_logger("scripts.train_model")


def main():
    parser = argparse.ArgumentParser(description="Train ConvNeXt-Tiny + U-Net on Sentinel-1 Oil Spill Dataset")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Training batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Initial learning rate")
    parser.add_argument("--patches-per-scene", type=int, default=80, help="Patches extracted per scene per epoch")
    parser.add_argument("--patch-size", type=int, default=256, help="Square patch size")
    args = parser.parse_args()

    # 1. Dataset Partitioning (Zero spatial leakage)
    split_mgr = SplitManager(random_seed=42)
    splits = split_mgr.get_scene_splits()

    logger.info(f"Loaded partitions: Train={len(splits['train'])} scenes, Val={len(splits['val'])} scenes, Test={len(splits['test'])} scenes.")

    # 2. PyTorch DataLoaders
    train_dataset = Sentinel1OilSpillDataset(
        scene_pairs=splits["train"],
        patch_size=args.patch_size,
        patches_per_scene=args.patches_per_scene,
        is_training=True,
        positive_sampling_ratio=0.65,
    )
    val_dataset = Sentinel1OilSpillDataset(
        scene_pairs=splits["val"],
        patch_size=args.patch_size,
        patches_per_scene=30,
        is_training=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # 3. Model & Trainer Setup
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patch_size=args.patch_size,
    )

    model = ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=True)
    trainer = ModelTrainer(model=model, config=config)

    # 4. Run Training Loop
    results = trainer.train(train_loader, val_loader)
    logger.info(f"Training completed successfully! Best validation Dice: {results['best_val_dice']:.4f}")


if __name__ == "__main__":
    main()
