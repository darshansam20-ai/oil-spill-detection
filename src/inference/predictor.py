"""
Production Inference Engine (PRD Section 6, 17).
Strictly inference-only deployment pipeline.
Loads versioned ConvNeXt-Tiny + U-Net model artifacts, applies sliding-window tiled inference,
and reconstructs seamless full-scene probability maps.
"""
import cv2
import gc
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

# Global in-memory cache to ensure the model weights are loaded only ONCE in RAM
_MODEL_CACHE: Dict[Tuple[str, str], torch.nn.Module] = {}
_VERSION_CACHE: Dict[Tuple[str, str], str] = {}


class OilSpillPredictor:
    """Production Inference Engine for Sentinel-1 Oil Spill Detection."""

    def __init__(
        self,
        checkpoint_path: Optional[Path] = None,
        patch_size: int = 256,
        overlap: int = 64,
        device: Optional[str] = None,
        use_bfloat16: bool = True,
    ):
        # Constrain PyTorch CPU threads to avoid memory fragmentation on container hosts
        try:
            torch.set_num_threads(1)
        except Exception:
            pass

        self.patch_size = patch_size
        self.overlap = overlap
        self.device = torch.device(device) if device else get_default_device()
        self.dtype = torch.bfloat16 if (use_bfloat16 and self.device.type == "cpu") else torch.float32
        
        self.tiler = ImageTiler(patch_size=patch_size, overlap=overlap)
        self.stitcher = PatchStitcher(patch_size=patch_size, blend_window="gaussian")
        self.preprocessor = SARPreprocessor()

        # Load Model with memory-efficient shared cache
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (settings.paths.checkpoints_dir / "best_model.pt")
        cache_key = (str(self.checkpoint_path.resolve()), str(self.device), str(self.dtype))

        if cache_key in _MODEL_CACHE:
            self.model = _MODEL_CACHE[cache_key]
            self.model_version = _VERSION_CACHE.get(cache_key, CURRENT_MODEL_VERSION)
        else:
            self.model = ConvNeXtTinyUNet(in_channels=1, num_classes=1, pretrained=False)
            if self.dtype == torch.bfloat16:
                self.model = self.model.to(torch.bfloat16)
            self.model_version = CURRENT_MODEL_VERSION

            if self.checkpoint_path.exists():
                logger.info(f"Loading trained weights from checkpoint: {self.checkpoint_path}")
                try:
                    checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)
                except Exception:
                    checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
                
                state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
                self.model.load_state_dict(state_dict)
                if isinstance(checkpoint, dict):
                    self.model_version = checkpoint.get("model_version", CURRENT_MODEL_VERSION)
                
                del checkpoint
                del state_dict
                gc.collect()
            else:
                logger.warning(f"Checkpoint not found at {self.checkpoint_path}. Initializing default model weights.")

            self.model.to(self.device)
            self.model.eval()
            _MODEL_CACHE[cache_key] = self.model
            _VERSION_CACHE[cache_key] = self.model_version

    def predict_patches(self, patches: List[np.ndarray], batch_size: int = 1) -> List[np.ndarray]:
        """
        Run batched neural network inference on extracted patches.
        Strictly inference-only: minimal memory allocation with bfloat16.
        """
        self.model.eval()
        predictions = []
        
        with torch.inference_mode():
            for i in range(0, len(patches), batch_size):
                batch_np = np.stack(patches[i:i + batch_size], axis=0)  # (B, H, W)
                batch_tensor = torch.from_numpy(batch_np).unsqueeze(1).to(self.device, dtype=self.dtype)  # (B, 1, H, W)
                
                probs = self.model.predict_probability(batch_tensor)  # (B, 1, H, W)
                probs_np = probs.squeeze(1).float().cpu().numpy()  # (B, H, W)

                if probs_np.ndim == 2:
                    probs_np = np.expand_dims(probs_np, axis=0)
                for j in range(probs_np.shape[0]):
                    predictions.append(probs_np[j])

                del batch_tensor, probs, probs_np, batch_np

        return predictions

    def predict_scene(self, preprocessed_img: np.ndarray, max_dim: int = 768) -> np.ndarray:
        """
        Run full tiled sliding-window inference on a preprocessed full SAR scene.
        Adaptively scales high-resolution satellite scenes to ensure rapid online inference.
        
        Args:
            preprocessed_img: 2D normalized SAR backscatter array (float32 in [0, 1]).
            max_dim: Maximum canvas dimension for sliding-window tiled inference.
            
        Returns:
            Reconstructed full-scene probability map (float32 in [0.0, 1.0]).
        """
        H, W = preprocessed_img.shape[:2]
        
        # Adaptive scaling for high-resolution satellite scenes
        if max(H, W) > max_dim:
            scale = max_dim / float(max(H, W))
            scaled_h, scaled_w = int(round(H * scale)), int(round(W * scale))
            scaled_img = cv2.resize(preprocessed_img, (scaled_w, scaled_h), interpolation=cv2.INTER_AREA)
        else:
            scaled_img = preprocessed_img
            scaled_h, scaled_w = H, W

        padded_h, padded_w, _, _ = self.tiler.get_patch_grid_dimensions(scaled_h, scaled_w)

        # 1. Extract sliding-window tiles
        tiles = list(self.tiler.extract_patches(scaled_img))
        raw_patches = [t.patch for t in tiles]
        logger.info(f"Extracted {len(tiles)} patches ({self.patch_size}x{self.patch_size}, overlap={self.overlap}) for inference on {scaled_h}x{scaled_w} canvas.")

        # 2. Run inference on patches with batch_size=1 for lowest possible memory footprint
        predicted_patches = self.predict_patches(raw_patches, batch_size=1)
        del raw_patches
        gc.collect()

        # 3. Reconstruct probability map with distance blending
        prob_map_scaled = self.stitcher.stitch_patches(
            predicted_patches=predicted_patches,
            tiles=tiles,
            orig_h=scaled_h,
            orig_w=scaled_w,
            padded_h=padded_h,
            padded_w=padded_w,
        )
        del predicted_patches, tiles
        gc.collect()

        # 4. Upsample back to original scene dimensions if scaled
        if (scaled_h, scaled_w) != (H, W):
            prob_map = cv2.resize(prob_map_scaled, (W, H), interpolation=cv2.INTER_LINEAR)
            del prob_map_scaled
            gc.collect()
        else:
            prob_map = prob_map_scaled

        logger.info(f"Full-scene probability map generated (shape: {prob_map.shape}, min={prob_map.min():.3f}, max={prob_map.max():.3f})")
        return prob_map
