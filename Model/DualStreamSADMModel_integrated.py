# File: DualStreamSADMModel.py
"""
Dual-stream model, adapted as follows:
- Use HybridWaveletFeatureExtractor as the frequency stream
- Use FullXception from Xception.py as the spatial stream
- Interface aligned, supporting end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

# Use your custom Xception
from Xception import FullXception  # Note: rename the file before importing
from HybridWaveletFeatureExtractor import HybridWaveletFeatureExtractor


class BidirectionalChannelAttention(nn.Module):
    """Bidirectional Channel Attention Module (unchanged)"""

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
        alphaS = self.sig(self.mlp_S(f_gap))
        alphaF = self.sig(self.mlp_F(s_gap))
        alphaS_map = alphaS.view(b, c, 1, 1)
        alphaF_map = alphaF.view(b, c, 1, 1)
        S_mod = S * alphaS_map
        F_mod = F * alphaF_map
        return S_mod, F_mod


class DualStreamSADMModel(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 1, pretrained_spatial: bool = False):
        super().__init__()

        # Use your FullXception as the spatial stream
        self.spatial = FullXception(pretrained=pretrained_spatial, num_classes=feature_dim)
        # Remove the final classification head, keep only the feature extractor
        self.spatial.classifier = nn.Identity()

        # Frequency stream: use your HybridWaveletFeatureExtractor
        self.freq_extractor = HybridWaveletFeatureExtractor(feature_dim=feature_dim)

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

    def forward(self, x_rgb: torch.Tensor) -> torch.Tensor:
        # Spatial stream forward
        S = self.spatial(x_rgb)
        if S.dim() == 2:
            S = S.unsqueeze(-1).unsqueeze(-1)  # [B, C] -> [B, C, 1, 1]

        # Frequency stream forward (note: requires image data, not feature vectors)
        # We need to extract frequency features from x_rgb
        # Convert tensor to numpy and extract features (pay attention to device conversion)
        freq_feats = []
        for i in range(x_rgb.size(0)):
            img_np = x_rgb[i].cpu().numpy().transpose(1, 2, 0)  # C,H,W -> H,W,C
            img_np = (img_np * 255).astype(np.uint8)
            feat = self.freq_extractor.extract_features(img_np)
            freq_feats.append(feat)

        F_vec = torch.stack(freq_feats).to(x_rgb.device)
        F_map = F_vec.unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]

        # Expand F_map to match the spatial dimensions of S
        _, _, h, w = S.shape
        F_map = F_map.expand(-1, -1, h, w)

        # Bidirectional attention
        S_mod, F_mod = self.attention(S, F_map)

        # Fusion
        fused = torch.cat([S_mod, F_mod], dim=1)
        fused = self.fusion_conv(fused)

        # Classification
        logits = self.classifier(fused)
        return logits
