import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel

STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
ds = PatchDataset("data/processed/patches/spread_224/test")
in_channels = len(CHANNELS) + 1
model = RobustFireSpreadModel(input_channels=in_channels, hidden_dims=[64, 128]).to(device)
model.load_state_dict(torch.load("models/best_spread_model.pth", map_location=device))
model.eval()

# Let's test on sample 120 and 122
for idx in [120, 122, 123]:
    x, y = ds[idx]
    x_tensor = x.unsqueeze(0).to(device)
    x_tensor = normalize_batch(x_tensor)
    
    with torch.no_grad():
        outputs = model(x_tensor)
        pred = outputs['spread_probability'] if isinstance(outputs, dict) else outputs
        
    p = pred.squeeze().cpu().numpy()
    target_sum = y.sum().item()
    print(f"Sample {idx}: Target fire pixels: {target_sum}")
    print(f"  Max prob: {p.max():.4f}, Min: {p.min():.4f}, Mean: {p.mean():.4f}")
    print(f"  Pixels > 0.1: {(p > 0.1).sum()}")
    print(f"  Pixels > 0.3: {(p > 0.3).sum()}")
    print(f"  Pixels > 0.5: {(p > 0.5).sum()}")
    print("-" * 30)

