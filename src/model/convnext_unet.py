"""
ConvNeXt-Tiny + U-Net Semantic Segmentation Architecture (PRD Section 6).
Combines a ConvNeXt-Tiny multi-scale hierarchical encoder with a custom U-Net decoder and skip connections.
Outputs a 1-channel per-pixel oil-spill probability map.
"""
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ConvBlock(nn.Module):
    """Convolutional block with LayerNorm/BatchNorm and GELU activation."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DecoderBlock(nn.Module):
    """
    U-Net Decoder block: Upsamples previous decoder stage, concatenates skip connection,
    and applies double convolution block.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        total_in = in_channels + skip_channels
        self.conv = ConvBlock(total_in, out_channels)

    def forward(self, x: torch.Tensor, skip: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.upsample(x)
        if skip is not None:
            # Handle possible slight spatial rounding differences
            if x.shape[-2:] != skip.shape[-2:]:
                x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class ConvNeXtTinyUNet(nn.Module):
    """
    ConvNeXt-Tiny Encoder + U-Net Decoder for Oil-Spill Semantic Segmentation.
    
    Encoder Stages (ConvNeXt-Tiny):
      Stage 0 (Stride 4):  96 channels
      Stage 1 (Stride 8):  192 channels
      Stage 2 (Stride 16): 384 channels
      Stage 3 (Stride 32): 768 channels (Bottleneck)
      
    Decoder Stages (U-Net with Skip Connections):
      Dec 3: 768 -> 384 (+ skip Stage 2: 384) -> 256 channels (Stride 16)
      Dec 2: 256 -> 192 (+ skip Stage 1: 192) -> 128 channels (Stride 8)
      Dec 1: 128 -> 96  (+ skip Stage 0: 96)  -> 64 channels  (Stride 4)
      Dec 0: 64  -> 32  (Upsample to Stride 1) -> 32 channels  (Stride 1)
      Head:  32  -> 1 channel logit
    """

    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        pretrained: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        # Load ConvNeXt-Tiny Backbone
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        backbone = models.convnext_tiny(weights=weights)

        # Adapt input conv for 1-channel SAR imagery if needed
        first_conv = backbone.features[0][0]
        if in_channels != 3:
            new_conv = nn.Conv2d(
                in_channels=in_channels,
                out_channels=first_conv.out_channels,
                kernel_size=first_conv.kernel_size,
                stride=first_conv.stride,
                padding=first_conv.padding,
            )
            if pretrained and in_channels == 1:
                # Initialize single channel weights from RGB average
                new_conv.weight.data = first_conv.weight.data.mean(dim=1, keepdim=True)
                new_conv.bias.data = first_conv.bias.data
            backbone.features[0][0] = new_conv

        # Encoder stages:
        # features[0] -> Stride 4 stem (96 ch)
        # features[1] -> Stage 0 blocks (96 ch)
        # features[2] -> Downsample to stride 8 (192 ch)
        # features[3] -> Stage 1 blocks (192 ch)
        # features[4] -> Downsample to stride 16 (384 ch)
        # features[5] -> Stage 2 blocks (384 ch)
        # features[6] -> Downsample to stride 32 (768 ch)
        # features[7] -> Stage 3 blocks (768 ch)
        self.encoder_stem = backbone.features[0]
        self.encoder_stage0 = backbone.features[1]
        self.downsample1 = backbone.features[2]
        self.encoder_stage1 = backbone.features[3]
        self.downsample2 = backbone.features[4]
        self.encoder_stage2 = backbone.features[5]
        self.downsample3 = backbone.features[6]
        self.encoder_stage3 = backbone.features[7]

        # Decoder Stages with Skip Connections
        self.dec3 = DecoderBlock(in_channels=768, skip_channels=384, out_channels=256)
        self.dec2 = DecoderBlock(in_channels=256, skip_channels=192, out_channels=128)
        self.dec1 = DecoderBlock(in_channels=128, skip_channels=96, out_channels=64)

        # Final upsampling from stride 4 to original resolution (stride 1)
        self.final_upsample = nn.Sequential(
            nn.Upsample(scale_factor=4, mode="bilinear", align_corners=False),
            ConvBlock(64, 32),
        )

        # Segmentation Head (1x1 conv)
        self.head = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward_encoder(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Extract multi-scale feature hierarchy from encoder."""
        # Stage 0: Stride 4 (96 ch)
        e0 = self.encoder_stage0(self.encoder_stem(x))
        # Stage 1: Stride 8 (192 ch)
        e1 = self.encoder_stage1(self.downsample1(e0))
        # Stage 2: Stride 16 (384 ch)
        e2 = self.encoder_stage2(self.downsample2(e1))
        # Stage 3: Stride 32 (768 ch)
        e3 = self.encoder_stage3(self.downsample3(e2))
        return [e0, e1, e2, e3]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning raw unactivated logits.
        Output shape: (B, num_classes, H, W)
        """
        orig_size = x.shape[-2:]
        e0, e1, e2, e3 = self.forward_encoder(x)

        d3 = self.dec3(e3, e2)  # Stride 16 (256 ch)
        d2 = self.dec2(d3, e1)  # Stride 8  (128 ch)
        d1 = self.dec1(d2, e0)  # Stride 4  (64 ch)

        out = self.final_upsample(d1)  # Stride 1 (32 ch)
        if out.shape[-2:] != orig_size:
            out = F.interpolate(out, size=orig_size, mode="bilinear", align_corners=False)

        logits = self.head(out)
        return logits

    def predict_probability(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference-only method applying Sigmoid to logits.
        Returns per-pixel oil-spill probability map in [0.0, 1.0].
        """
        logits = self.forward(x)
        return torch.sigmoid(logits)
