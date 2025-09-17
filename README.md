# Dual-Stream Attention for Deepfake Detection

This repository contains the core implementation of the paper:

> Authors: [Boyao Wei]  
> Conference: ICASSP 2026 

---


We propose a Dual-Stream Attention Deepfake Detection Model (SADM) that integrates:
1. Frequency Stream: Hybrid wavelet-based multi-level feature extraction (317-d vector).
2. Spatial Stream: Xception backbone for spatial feature extraction.
3. Dual-Stream Attention Fusion: A novel bidirectional attention mechanism that adaptively fuses spatial and frequency-domain features.

This repo provides the core code modules for reproducibility of the algorithmic design.
The full training pipeline (data loading, optimization, checkpoints) will be released in the camera-ready version.

Model implementation notes

DualStreamSADMModel.py
This file contains the core dual-stream model implementation. The architecture implements two independent branches: a spatial stream that processes RGB images and a frequency stream that processes grayscale images. The spatial stream is based on an Xception backbone for RGB input; the frequency stream uses a hybrid wavelet transform to extract frequency features from the grayscale image (you must explicitly separate RGB and grayscale inputs — see the implementation notes below).

DualStreamSADMModel_integrated.py
An integrated, end-to-end demo pipeline. Starting from an RGB image, this script automatically derives a grayscale image and extracts the domain (frequency) features, demonstrating the full data flow and model behaviour. It is intended for validating the overall logic and is not a production deployment script.

Note: DualStreamSADMModel_integrated.py is provided for demonstration purposes only. For formal experiments and training runs, prefer using DualStreamSADMModel.py
---


