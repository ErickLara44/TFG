import torch
import sys

model_path = "best_robust_ignition_model.pth"
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    state_dict = checkpoint['model_state_dict']
else:
    state_dict = checkpoint

# Find the first layer's shape
for k, v in state_dict.items():
    if "conv" in k or "weight" in k:
        try:
            print(f"{k}: {v.shape}")
            break
        except:
            pass
