import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from torch.utils.data import DataLoader

from src.config.settings import settings
from src.dataset.split_manager import SplitManager
from src.dataset.dataset_loader import Sentinel1OilSpillDataset
from src.training.evaluator import ModelEvaluator
from src.utils.logger import get_logger

logger = get_logger("scripts.evaluate_model")


def main():
    parser = argparse.ArgumentParser(description="Evaluate ConvNeXt-Tiny + U-Net on Test Split")
    parser.add_argument("--checkpoint", type=str, default=str(settings.paths.checkpoints_dir / "best_model.pt"))
    parser.add_argument("--patch-size", type=int, default=256)
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    # 1. Load Test Partition
    split_mgr = SplitManager(random_seed=42)
    splits = split_mgr.get_scene_splits()
    test_pairs = splits["test"]

    logger.info(f"Evaluating checkpoint {checkpoint_path.name} on {len(test_pairs)} independent test scenes.")

    test_dataset = Sentinel1OilSpillDataset(
        scene_pairs=test_pairs,
        patch_size=args.patch_size,
        patches_per_scene=50,
        is_training=False,
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=0)

    # 2. Run Evaluation
    evaluator = ModelEvaluator.load_from_checkpoint(checkpoint_path)
    
    print("\n=======================================================")
    print("      CONVNEXT-TINY + U-NET TEST BENCHMARK RESULTS     ")
    print("=======================================================")
    
    sweep_results = evaluator.sweep_thresholds(test_loader, thresholds=[0.3, 0.4, 0.5, 0.6, 0.7])
    print(f"{'Threshold':<12} | {'Dice Score':<12} | {'IoU':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 65)
    for thresh, m in sweep_results.items():
        print(f"{thresh:<12.2f} | {m['dice_score']:<12.4f} | {m['iou']:<10.4f} | {m['precision']:<10.4f} | {m['recall']:<10.4f}")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
