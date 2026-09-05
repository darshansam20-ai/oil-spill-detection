"""
Production Inference Engine (PRD Section 6, 17).
Strictly inference-only deployment pipeline.
Loads versioned ConvNeXt-Tiny + U-Net model artifacts, applies sliding-window tiled inference,
and reconstructs seamless full-scene probability maps.
"""
import json
from pathlib import Path
from typing import List, Optional, Tuple
import numpy as np
import torch

from src.config.constants import CURRENT_MODEL_VERSION
from src.config.settings import settings
from src.model.convnext_unet import ConvNeXtTinyUNet
from src.preprocessing.sar_preprocessor import SARPreprocessor, PreprocessingConfig
from src.tiling.tiler import ImageTiler, PatchTile
from src.tiling.stitcher import PatchStitcher
from src.utils.device import get_default_device
from src.utils.logger import get_logger

logger = get_logger("inference.predictor")


class OilSpillPredictor:
    """Production Inference Engine for Sentinel-1 Oil Spill Detection."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        patch_size: int = 256,
        overlap: int = 64,
        device: Optional[str] = None,
    ):
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = torch.device(device) if device else get_default_device()
        
        self.tiler = ImageTiler(patch_size=patch_size, overlap=overlap)
        self.stitcher = PatchStitcher(patch_size=patch_size, blend_window="gaussian")
        self.preprocessor = SARPreprocessor()

        # Load Model
        self.checkpoint_path = checkpoint_path or (settings.paths.checkpoints_dir / "best_model.pt")
        self.model = ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=False)
        self.model_version = CURRENT_MODEL_VERSION

        if self.checkpoint_path.exists():
            logger.info(f"Loading trained weights from checkpoint: {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model_version = checkpoint.get("model_version", CURRENT_MODEL_VERSION)
        else:
            logger.warning(f"Checkpoint not found at {self.checkpoint_path}. Initializing default model weights.")

        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def predict_patches(self, patches: List[np.ndarray], batch_size: int = 8) -> List[np.ndarray]:
        """
        Run batched neural network inference on extracted patches.
        Strictly inference-only: no gradient tracking or optimizer updates.
        """
        self.model.eval()
        predictions = []
        
        for i in range(0, len(patches), batch_size):
            batch_np = np.stack(patches[i:i + batch_size], axis=0)  # (B, H, W)
            batch_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(self.device, dtype=torch.float32)  # (B, 1, H, W)
            
            probs = self.model.predict_probability(batch_tensor)  # (B, 1, H, W)
            probs_np = probs.squeeze(1).cpu().numpy()  # (B, H, W)

            if probs_np.ndim == 2:
                probs_np = np.expand_dims(probs_np, axis=0)
            for j in range(probs_np.shape[0]):
                predictions.append(probs_np[j])

        return predictions

    def predict_scene(self, preprocessed_img: np.ndarray) -> np.ndarray:
        """
        Run full tiled sliding-window inference on a preprocessed full SAR scene.
        
        Args:
            preprocessed_img: 2D normalized SAR backscatter array (float32 in [0, 1]).
            
        Returns:
            Reconstructed full-scene probability map (float32 in [0.0, 1.0]).
        """
        H, W = preprocessed_img.shape[:2]
        padded_h, padded_w, _, _ = self.tiler.get_patch_grid_dimensions(H, W)

        # 1. Extract sliding-window tiles
        tiles = list(self.tiler.extract_patches(preprocessed_img))
        raw_patches = [t.patch for t in tiles]
        logger.info(f"Extracted {len(tiles)} patches ({self.patch_size}x{self.patch_size}, overlap={self.overlap}) for inference.")

        # 2. Run inference on patches
        predicted_patches = self.predict_patches(raw_patches)

        # 3. Reconstruct full scene probability map with distance blending
        prob_map = self.stitcher.stitch_patches(
            predicted_patches=predicted_patches,
            tiles=tiles,
            orig_h=H,
            orig_w=W,
            padded_h=padded_h,
            padded_w=padded_w,
        )

        logger.info(f"Full-scene probability map generated (shape: {prob_map.shape}, min={prob_map.min():.3f}, max={prob_map.max():.3f})")
        return prob_map
