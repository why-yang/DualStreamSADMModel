# File: sadm_dualstream_model.py
"""
Dual-stream architecture: Xception spatial backbone + FrequencyNet from file above +
Bidirectional Channel Attention Fusion Module + classification head.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import xception

try:
    from torchvision.models import xception as _xception_loader
except Exception:
    _xception_loader = None


class BidirectionalChannelAttention(nn.Module):
    """实现了本文所述的双向通道注意力。
给定空间特征S（B，C，H，W）和频率特征F（B，C.H，W或B，C.1,1），
通过GAP计算信道向量并产生交叉引导权重。
    """
    def __init__(self, channels: int, hidden: int = 128):
        super().__init__()
        # MLPs for cross-domain mapping
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
        # small temperature to stabilize sigmoid
        self.sig = nn.Sigmoid()

    def forward(self, S: torch.Tensor, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # S: (B,C,H,W), F: (B,C,h,w) where h,w may be 1
        b, c, _, _ = S.shape
        s_gap = F.adaptive_avg_pool2d(S, (1, 1)).view(b, c)
        f_gap = F.adaptive_avg_pool2d(F, (1, 1)).view(b, c)
        # cross mapping
        alphaS = self.sig(self.mlp_S(f_gap))  # freq -> spatial weights
        alphaF = self.sig(self.mlp_F(s_gap))  # spat -> freq weights
        # reshape to (B,C,1,1)
        alphaS_map = alphaS.view(b, c, 1, 1)
        alphaF_map = alphaF.view(b, c, 1, 1)
        S_mod = S * alphaS_map
        F_mod = F * alphaF_map
        return S_mod, F_mod


class DualStreamSADMModel(nn.Module):
    def __init__(self, feature_dim: int = 256, num_classes: int = 1, pretrained_spatial: bool = False):
        super().__init__()
        # Spatial backbone (Xception). If not available, use a lightweight convnet.
        if _xception_loader is not None:
            # This call signature may differ depending on Xception implementation.
            self.spatial = _xception_loader(pretrained=pretrained_spatial)
            # remove final classifier, adapt to output feature map
            if hasattr(self.spatial, 'fc'):
                self.spatial.fc = nn.Identity()
        else:
            # fallback: small conv feature extractor
            self.spatial = nn.Sequential(
                nn.Conv2d(3, 64, 3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, feature_dim, 3, stride=2, padding=1),
                nn.BatchNorm2d(feature_dim),
                nn.ReLU(inplace=True)
            )

        # Frequency net
        from sadm_frequency_stream import FrequencyNet
        self.freqnet = FrequencyNet(feature_dim=feature_dim)

        # Bidirectional attention: channel dimension must match
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
        # x_rgb: (B,3,H,W)
        # spatial forward: get a feature map of shape (B,C,Hs,Ws)
        S = self._spatial_forward(x_rgb)
        b, c, hs, ws = S.shape
        # frequency forward: accept either a 317-dim vector (raw) or tensor outputs from freqnet
        if freq_feat_vec is not None and freq_feat_vec.dim() == 2 and freq_feat_vec.shape[1] == 317:
            freq_out = self.freqnet(freq_feat_vec)
            F_map = freq_out['map']
            # broadcast F_map to match spatial size
            F_map = F_map.expand(-1, -1, hs, ws)
        elif freq_feat_vec is not None and freq_feat_vec.dim() == 2 and freq_feat_vec.shape[1] == c:
            # if user passed a vec already of size c
            F_map = freq_feat_vec.view(b, c, 1, 1).expand(-1, -1, hs, ws)
        else:
            # If no frequency vector provided, initialize neutral map
            F_map = torch.ones((b, c, hs, ws), device=S.device, dtype=S.dtype) * 0.5

        # Align channel numbers if necessary
        if F_map.shape[1] != S.shape[1]:
            # project to S channels
            proj = nn.Conv2d(F_map.shape[1], S.shape[1], kernel_size=1).to(S.device)
            F_map = proj(F_map)

        # apply bidirectional attention
        S_mod, F_mod = self.attention(S, F_map)

        # concatenate and fuse
        fused = torch.cat([S_mod, F_mod], dim=1)
        fused = self.fusion_conv(fused)

        logits = self.classifier(fused)
        return logits

    def _spatial_forward(self, x: torch.Tensor) -> torch.Tensor:
        # attempt to use spatial backbone to produce a feature map
        try:
            out = self.spatial(x)
            # if backbone returned vector, reshape to (B,C,1,1)
            if out.dim() == 2:
                b, dim = out.shape
                out = out.view(b, dim, 1, 1)
            return out
        except Exception:
            # fallback: simple forwarding through fallback conv
            return self.spatial(x)


