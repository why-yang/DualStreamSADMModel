# Dual-Stream Attention for Deepfake Detection

This repository contains the core implementation of the paper:

> Authors: [Boyao Wei]  
> Conference: ICASSP 2026 

---


We propose a Dual-Stream Attention Deepfake Detection Model (SADM) that integrates:
1. Frequency Stream: Hybrid wavelet-based multi-level feature extraction (317-d vector).
2. Spatial Stream: Xception backbone for spatial feature extraction.
3. Dual-Stream Attention Fusion: A novel bidirectional attention mechanism that adaptively fuses spatial and frequency-domain features.

This repo provides the **core code modules for reproducibility of the algorithmic design.
The full training pipeline (data loading, optimization, checkpoints) will be released in the camera-ready version.

---


