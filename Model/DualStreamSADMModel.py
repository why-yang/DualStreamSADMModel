# File: DualStreamSADMModel.py
"""
Dual-stream architecture:
- Spatial stream: RGB input (Xception or lightweight conv)
- Frequency stream: Grayscale input (HybridWavelet / FrequencyNet)
- Bidirectional channel attention fusion
- Classification head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

try:
    from torchvision.models import xception as _xception_loader
except Exception:
    _xception_loader = None

# -------------------- Bidirectional Channel Attention --------------------
class BidirectionalChannelAttention(nn.Module):
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
        s_gap = F.adaptive_avg_pool2d(S, (1, 1)).view(b, c)
        f_gap = F.adaptive_avg_pool2d(F, (1, 1)).view(b, c)
        alphaS = self.sig(self.mlp_S(f_gap))  # freq -> spatial
        alphaF = self.sig(self.mlp_F(s_gap))  # spatial -> freq
        S_mod = S * alphaS.view(b, c, 1, 1)
        F_mod = F * alphaF.view(b, c, 1, 1)
        return S_mod, F_mod

# -------------------- Dual Stream Model --------------------
class DualStreamSADMModel(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 2, pretrained_spatial: bool = False):
        super().__init__()
        # Spatial backbone (Xception or fallback conv)
        if _xception_loader is not None:
            self.spatial = _xception_loader(pretrained=pretrained_spatial)
            if hasattr(self.spatial, 'fc'):
                self.spatial.fc = nn.Identity()  # remove classifier
        else:
            self.spatial = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )

        # Frequency stream (HybridWavelet / FrequencyNet)
        from sadm_frequency_stream import FrequencyNet
        self.freq_stream = FrequencyNet(feature_dim=feature_dim)

        # Bidirectional attention
        self.attention = BidirectionalChannelAttention(channels=feature_dim)

        # Fusion and classification head
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(feature_dim * 2, feature_dim, kernel_size=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x_rgb: torch.Tensor, freq_feat_vec: torch.Tensor = None) -> torch.Tensor:
        """
        x_rgb: [B,3,H,W]
        freq_feat_vec: optional precomputed frequency vector
        """
        # ---------------- 1. Explicit grayscale conversion ----------------
        x_gray = torch.mean(x_rgb, dim=1, keepdim=True)  # [B,1,H,W]

        # ---------------- 2. Forward two streams ----------------
        S = self._spatial_forward(x_rgb)  # RGB -> spatial features
        F = self.freq_stream(x_gray)      # Grayscale -> frequency features

        # ---------------- 3. Optional channel alignment ----------------
        if F.shape[1] != S.shape[1]:
            proj = nn.Conv2d(F.shape[1], S.shape[1], kernel_size=1).to(F.device)
            F = proj(F)

        # ---------------- 4. Bidirectional attention ----------------
        S_mod, F_mod = self.attention(S, F)

        # ---------------- 5. Concatenate, fuse, classify ----------------
        fused = torch.cat([S_mod, F_mod], dim=1)
        fused = self.fusion_conv(fused)
        logits = self.classifier(fused)

        return logits

    def _spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for spatial backbone, returns feature map"""
        out = self.spatial(x)
        if out.dim() == 2:  # vector output -> (B,C,1,1)
            b, dim = out.shape
            out = out.view(b, dim, 1, 1)
        return out
