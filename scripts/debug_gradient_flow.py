import torch
from pathlib import Path
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.prop_swinv2 import SwinV2_3D_FirePrediction

def test_gradients():
    model = SwinV2_3D_FirePrediction(
        in_chans=12,
        embed_dim=48, 
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        patch_size=(2, 4, 4), 
        window_size=(4, 4, 4)
    )
    
    # Create fake input tensor
    x = torch.randn(2, 24, 12, 32, 32, requires_grad=True)
    
    # Forward pass
    out = model(x)
    loss = out.sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    grad_env = x.grad[:, :, :-1, :, :].abs().mean().item()
    grad_fire = x.grad[:, :, -1:, :, :].abs().mean().item()
    
    print(f"Gradient Env Flow: {grad_env:.8f}")
    print(f"Gradient Fire Flow: {grad_fire:.8f}")

if __name__ == '__main__':
    test_gradients()
