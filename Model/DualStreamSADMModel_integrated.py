# File: DualStreamSADMModel.py
"""
Dual-stream model with explicit grayscale generation for the frequency stream.
- Spatial stream: FullXception (RGB input)
- Frequency stream: HybridWaveletFeatureExtractor (grayscale input)
- Bidirectional attention + fusion + classification
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from typing import Tuple

# Import your modules
from Xception import FullXception  # rename the file before importing
from HybridWaveletFeatureExtractor import HybridWaveletFeatureExtractor


class BidirectionalChannelAttention(nn.Module):
    """Bidirectional Channel Attention Module"""

    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        self.mlp_S = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )
        self.mlp_F = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels)
        )
        self.sig = nn.Sigmoid()

    def forward(self, S: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b, c, _, _ = S.shape
        s_gap = nn.functional.adaptive_avg_pool2d(S, (1, 1)).view(b, c)
        f_gap = nn.functional.adaptive_avg_pool2d(F, (1, 1)).view(b, c)
        alphaS = self.sig(self.mlp_S(f_gap))
        alphaF = self.sig(self.mlp_F(s_gap))
        S_mod = S * alphaS.view(b, c, 1, 1)
        F_mod = F * alphaF.view(b, c, 1, 1)
        return S_mod, F_mod


class DualStreamSADMModel(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 1, pretrained_spatial: bool = False):
        super().__init__()
        # Spatial stream
        self.spatial = FullXception(pretrained=pretrained_spatial, num_classes=feature_dim)
        self.spatial.classifier = nn.Identity()  # Remove classification head

        # Frequency stream
        self.freq_extractor = HybridWaveletFeatureExtractor(feature_dim=feature_dim)

        # Bidirectional attention
        self.attention = BidirectionalChannelAttention(channels=feature_dim)

        # Fusion + classifier
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        """
        x_rgb: [B, 3, H, W], RGB images in range [0,1] float
        """
        B = x_rgb.size(0)

        # --- Spatial stream ---
        S = self.spatial(x_rgb)
        if S.dim() == 2:
            S = S.unsqueeze(-1).unsqueeze(-1)  # [B, C] -> [B, C, 1, 1]

        # --- Frequency stream (explicit grayscale) ---
        # Convert batch to numpy and grayscale
        freq_feats = []
        for i in range(B):
            img_np = x_rgb[i].cpu().numpy().transpose(1, 2, 0)  # [H,W,C]
            img_np = (img_np * 255).astype(np.uint8)
            # Explicit grayscale
            if img_np.ndim == 3 and img_np.shape[2] == 3:
                img_gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            else:
                img_gray = img_np
            feat = self.freq_extractor.extract_features(img_gray)
            freq_feats.append(feat)

        F_vec = torch.stack(freq_feats).to(x_rgb.device)
        F_map = F_vec.unsqueeze(-1).unsqueeze(-1)  # [B,C,1,1]

        # Expand F_map to match spatial feature map size
        _, _, h, w = S.shape
        F_map = F_map.expand(-1, -1, h, w)

        # --- Bidirectional attention ---
        S_mod, F_mod = self.attention(S, F_map)

        # --- Fusion ---
        fused = torch.cat([S_mod, F_mod], dim=1)
        fused = self.fusion_conv(fused)

        # --- Classification ---
        logits = self.classifier(fused)
        return logits
