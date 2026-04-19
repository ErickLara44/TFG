import torch
import sys
import os

sys.path.append(os.path.dirname(__file__))

model_path = "models/best_ignition_model.pth"
state_dict = torch.load(model_path, map_location="cpu", weights_only=True)

# Find the first layer's shape
first_key = list(state_dict.keys())[0]
print(f"First layer ({first_key}) shape:", state_dict[first_key].shape)

for k, v in state_dict.items():
    if "conv" in k or "weight" in k:
        print(f"{k}: {v.shape}")
        break  # Usually the first convolution
