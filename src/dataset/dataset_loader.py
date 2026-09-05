"""
PyTorch Dataset for Sentinel-1 Marine Oil-Spill Semantic Segmentation (PRD Section 5 & 16).
Extracts patches, applies data augmentations, normalizes SAR backscatter, and yields PyTorch tensors.
"""
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import numpy as np
import rasterio
from PIL import Image
import torch
from torch.utils.data import Dataset

from src.preprocessing.sar_preprocessor import SARPreprocessor, PreprocessingConfig
from src.utils.logger import get_logger

logger = get_logger("dataset.dataset_loader")


class Sentinel1OilSpillDataset(Dataset):
    """
    PyTorch Dataset yielding 256x256 (or configurable size) image/mask patches.
    """

    def __init__(
        self,
        scene_pairs: List[Tuple[Path, Path, str]],
        patch_size: int = 256,
        patches_per_scene: int = 100,
        is_training: bool = True,
        positive_sampling_ratio: float = 0.6,
        preprocessing_config: Optional[PreprocessingConfig] = None,
    ):
        self.scene_pairs = scene_pairs
        self.patch_size = patch_size
        self.patches_per_scene = patches_per_scene
        self.is_training = is_training
        self.positive_sampling_ratio = positive_sampling_ratio
        self.preprocessor = SARPreprocessor(preprocessing_config or PreprocessingConfig())

        # Preload or index scenes
        self.scenes_data = []
        self._load_scenes()

    def _load_scenes(self) -> None:
        """Load and normalize scene rasters into memory."""
        for img_path, mask_path, split_name in self.scene_pairs:
            with rasterio.open(img_path) as src_img, rasterio.open(mask_path) as src_mask:
                img_data = src_img.read(1).astype(np.float32)
                mask_data = src_mask.read(1).astype(np.float32)

            # Preprocess image backscatter
            norm_img, _ = self.preprocessor.preprocess_image(img_data)
            bin_mask = (mask_data > 0.5).astype(np.float32)

            # Find coordinates of positive oil spill pixels for targeted sampling
            oil_ys, oil_xs = np.where(bin_mask > 0)
            has_oil = len(oil_ys) > 0

            self.scenes_data.append({
                "scene_name": img_path.stem,
                "image": norm_img,
                "mask": bin_mask,
                "h": norm_img.shape[0],
                "w": norm_img.shape[1],
                "oil_ys": oil_ys,
                "oil_xs": oil_xs,
                "has_oil": has_oil,
            })
            
        logger.info(f"Loaded {len(self.scenes_data)} scenes into dataset (Training={self.is_training}).")

    def __len__(self) -> int:
        return len(self.scenes_data) * self.patches_per_scene

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, str]:
        scene_idx = idx // self.patches_per_scene
        scene = self.scenes_data[scene_idx]

        h, w = scene["h"], scene["w"]
        p_size = self.patch_size

        # Determine top-left crop coordinate (y, x)
        sample_positive = self.is_training and scene["has_oil"] and (np.random.rand() < self.positive_sampling_ratio)
        if sample_positive:
            # Center around a known oil spill pixel with random jitter
            rand_idx = np.random.randint(0, len(scene["oil_ys"]))
            center_y = scene["oil_ys"][rand_idx]
            center_x = scene["oil_xs"][rand_idx]
            y = int(np.clip(center_y - p_size // 2 + np.random.randint(-30, 31), 0, max(0, h - p_size)))
            x = int(np.clip(center_x - p_size // 2 + np.random.randint(-30, 31), 0, max(0, w - p_size)))
        else:
            # Uniform random crop
            y = np.random.randint(0, max(1, h - p_size + 1)) if h >= p_size else 0
            x = np.random.randint(0, max(1, w - p_size + 1)) if w >= p_size else 0

        img_crop = scene["image"][y:y + p_size, x:x + p_size]
        mask_crop = scene["mask"][y:y + p_size, x:x + p_size]

        # Handle edge padding if scene is smaller than patch_size
        if img_crop.shape[0] < p_size or img_crop.shape[1] < p_size:
            pad_h = p_size - img_crop.shape[0]
            pad_w = p_size - img_crop.shape[1]
            img_crop = np.pad(img_crop, ((0, pad_h), (0, pad_w)), mode="reflect")
            mask_crop = np.pad(mask_crop, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        # Data Augmentation during training
        if self.is_training:
            # Random Horizontal Flip
            if np.random.rand() > 0.5:
                img_crop = np.fliplr(img_crop)
                mask_crop = np.fliplr(mask_crop)
            # Random Vertical Flip
            if np.random.rand() > 0.5:
                img_crop = np.flipud(img_crop)
                mask_crop = np.flipud(mask_crop)
            # Random 90-degree Rotation
            k = np.random.randint(0, 4)
            if k > 0:
                img_crop = np.rot90(img_crop, k)
                mask_crop = np.rot90(mask_crop, k)

        # Convert to PyTorch tensors: (1, H, W)
        img_tensor = torch.from_numpy(img_crop.copy()).unsqueeze(0).float()
        mask_tensor = torch.from_numpy(mask_crop.copy()).unsqueeze(0).float()

        return img_tensor, mask_tensor, scene["scene_name"]
