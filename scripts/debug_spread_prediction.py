
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from src.models.prop import RobustFireSpreadModel

def debug_prediction():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🕵️‍♂️ Debugging on {device}")

    # 1. Load Dataset
    val_dir = "data/processed/patches/spread_224/val"
    try:
        val_ds = PatchDataset(val_dir)
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return

    loader = DataLoader(val_ds, batch_size=4, shuffle=True)
    
    # 2. Load Model
    # Determine input channels from the dataset (first sample)
    x, y = val_ds[0]
    in_channels = x.shape[1] # (T, C, H, W) -> C channels per timestep?
    # Wait, the model expects (B, T, C, H, W) or (B, C_total, H, W)?
    # Let's check model definition. RobustFireSpreadModel expects (B, T, C, H, W) in forward?
    # No, let's check train_spread.py. 
    # train_spread.py: in_channels = len(CHANNELS) + 1. 
    # The dataset returns (T, C, H, W).
    # The model input_channels arg usually refers to C.
    
    print(f"📊 Dataset input shape: {x.shape}")
    print(f"   Inferred channels per timestep: {in_channels}")
    
    model = RobustFireSpreadModel(input_channels=in_channels).to(device)
    
    # Load checkpoint if exists
    checkpoint_path = "models/best_spread_model.pth"
    if os.path.exists(checkpoint_path):
        print(f"💾 Loading checkpoint: {checkpoint_path}")
        try:
            state = torch.load(checkpoint_path, map_location=device)
            # Handle different saving formats (full dict or state_dict)
            if 'model_state_dict' in state:
                model.load_state_dict(state['model_state_dict'])
            else:
                model.load_state_dict(state)
            print("✅ Checkpoint loaded.")
        except Exception as e:
            print(f"⚠️ Could not load checkpoint: {e}")
    else:
        print("⚠️ No checkpoint found. Using random weights.")
    
    model.eval()
    
    # 3. Inference on one batch
    with torch.no_grad():
        x_batch, y_batch = next(iter(loader))
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        
        print(f"🚀 Running inference on batch of shape {x_batch.shape}...")
        outputs = model(x_batch)
        
        if isinstance(outputs, dict):
            pred = outputs['spread_probability']
        else:
            pred = outputs
            
        print(f"📊 Prediction shape: {pred.shape}")
        
        # 4. Analyze Statistics
        print("\n🧐 STATISTICS:")
        print(f"   Input Range: [{x_batch.min():.3f}, {x_batch.max():.3f}]")
        print(f"   Target Fire Pixels: {y_batch.sum().item()} / {y_batch.numel()} ({y_batch.sum().item()/y_batch.numel()*100:.4f}%)")
        print(f"   Predicted Range: [{pred.min():.4f}, {pred.max():.4f}]")
        print(f"   Predicted Mean:  {pred.mean():.4f}")
        
        # Check if prediction is "dead" (all zeros or all same value)
        if pred.std() < 1e-4:
            print("⚠️ WARNING: Prediction has almost zero variance! (Model collapsed?)")
        
        # 5. Visualize
        # Take the first sample
        input_fire_t0 = x_batch[0, -1, -1, :, :].cpu().numpy() # Last timestep, last channel (fire mask)
        target_fire_t1 = y_batch[0, 0, :, :].cpu().numpy()
        pred_fire_t1 = pred[0, 0, :, :].cpu().numpy()
        
        plt.figure(figsize=(15, 5))
        
        plt.subplot(1, 3, 1)
        plt.title("Input Fire (t)")
        plt.imshow(input_fire_t0, cmap='inferno')
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.title("Target Fire (t+1)")
        plt.imshow(target_fire_t1, cmap='inferno')
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.title(f"Predicted Spread (Min:{pred_fire_t1.min():.3f} Max:{pred_fire_t1.max():.3f})")
        plt.imshow(pred_fire_t1, cmap='inferno', vmin=0, vmax=1)
        plt.colorbar()
        
        os.makedirs("outputs/debug", exist_ok=True)
        save_path = "outputs/debug/spread_debug.png"
        plt.savefig(save_path)
        print(f"\n✅ Visualization saved to {save_path}")

if __name__ == "__main__":
    debug_prediction()
