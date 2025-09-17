import torch
import numpy as np
from DualStreamSADMModel import DualStreamSADMModel

# Initialize model
model = DualStreamSADMModel(feature_dim=256, num_classes=1)

# Random input
x = torch.randn(2, 3, 299, 299)

# Forward pass test
with torch.no_grad():
    output = model(x)
    print("Output shape:", output.shape)
    print("Forward pass successful!")
