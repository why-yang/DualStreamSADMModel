# Dual-Stream Attention for Deepfake Detection

This repository contains the core implementation of the paper:

> Authors: [Boyao Wei]  
> Conference: ICASSP 2026 

---


We propose a Dual-Stream Attention Deepfake Detection Model (SADM) that integrates:
1. Frequency Stream: Hybrid wavelet-based multi-level feature extraction .
2. Spatial Stream: Xception backbone for spatial feature extraction.
3. Dual-Stream Attention Fusion: A novel bidirectional attention mechanism that adaptively fuses spatial and frequency-domain features.

Minimal PyTorch implementation for deepfake detection. It couples a spatial stream (Xception or a lightweight CNN) with a frequency stream (Coiflet+Haar hybrid wavelets, region-adaptive weights, three-level features). The two streams are fused via bi-directional channel attention and trained with the SADM loss (classification + sensitivity contrast + channel regularization).

Key points

Works with standard face images; outputs two logits/probabilities .

Optional facial landmark predictor (shape_predictor_68_face_landmarks.dat) improves masks; if absent, a stable grid fallback is used.

Two modes available: Strict (Kovesi phase congruency + LBP-10) and Legacy (Gabor approximation + LBP-59).

If TIMM’s Xception is unavailable, the spatial stream automatically falls back to a light CNN.

Notes

Environment/setup instructions will be added in “Reminders”.

Cite Kovesi (Phase Congruency), Ojala et al. (LBP), and Xception when using this work.


---


