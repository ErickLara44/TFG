import sys
import os
import torch
import numpy as np
import xarray as xr

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.preprocessing import compute_derived_features
from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS

def main():
    print("Loading Datacube...")
    datacube_path = "data/IberFire.nc"
    ds = xr.open_dataset(datacube_path)
    
    print("Computing derived features for 2023...")
    ds_year = ds.sel(time=ds.time.dt.year == 2023)
    ds_year = compute_derived_features(ds_year)
    
    print("Loading Model...")
    model = RobustFireIgnitionModel(num_input_channels=18, temporal_context=3, hidden_dims=[64, 128])
    checkpoint = torch.load("best_robust_ignition_model.pth", map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    print("Locating high and low elevation points without fire...")
    # Find spatial points
    elevation = ds_year['elevation_mean'].isel(time=0).values
    
    # We want valid mask
    valid_mask = ~np.isnan(elevation)
    high_y, high_x = np.where((elevation > 1000) & valid_mask)
    low_y, low_x = np.where((elevation < 500) & valid_mask)
    
    print(f"Found {len(high_y)} high elevation points and {len(low_y)} low elevation points.")

    # Calculate normalization stats manually based on 2023
    means = []
    stds = []
    for var in DEFAULT_FEATURE_VARS:
        da = ds_year[var]
        means.append(float(da.mean().values))
        stds.append(float(da.std().values))
    stats_mean = torch.tensor(means, dtype=torch.float32).view(1, -1, 1, 1)
    stats_std = torch.tensor(stds, dtype=torch.float32).view(1, -1, 1, 1)

    # Let's pick 50 random times for each to evaluate
    num_samples = 50
    temporal_context = 3
    
    def evaluate_points(ys, xs, desc):
        np.random.seed(42)
        logits = []
        buggy_probs = []
        true_probs = []
        
        for _ in range(num_samples):
            idx = np.random.randint(len(ys))
            y, x = ys[idx], xs[idx]
            # pick random valid time
            t = np.random.randint(temporal_context, len(ds_year.time) - 1)
            
            # Check if there's fire (we want NO fire)
            if ds_year['is_fire'].isel(time=t+1, y=y, x=x).values > 0:
                continue
                
            y_start = max(0, y - 32)
            y_end = min(ds_year.sizes['y'], y + 32)
            x_start = max(0, x - 32)
            x_end = min(ds_year.sizes['x'], x + 32)
            
            x_seq = []
            for var in DEFAULT_FEATURE_VARS:
                if "time" in ds_year[var].dims:
                    arr = ds_year[var].isel(time=slice(t-temporal_context+1, t+1), y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                else:
                    arr_static = ds_year[var].isel(y=slice(y_start, y_end), x=slice(x_start, x_end)).values
                    arr = np.repeat(arr_static[np.newaxis, ...], temporal_context, axis=0)
                x_seq.append(arr)
                
            x_seq = np.stack(x_seq, axis=1).astype(np.float32)
            
            # Padding if needed (simplified)
            if x_seq.shape[2] < 64 or x_seq.shape[3] < 64:
                continue # skip boundary
                
            x_seq = np.nan_to_num(x_seq, nan=0.0)
            tensor = torch.from_numpy(x_seq).unsqueeze(0) # (1, T, C, H, W)
            
            # Normalize
            tensor = (tensor - stats_mean) / (stats_std + 1e-6)
            
            with torch.no_grad():
                out = model(tensor)
                val = out['ignition'].item()
                
            logits.append(val)
            true_prob = float(torch.sigmoid(torch.tensor(val)).item())
            buggy_prob = float(torch.sigmoid(torch.tensor(val)).item() if val > 1.0 or val < 0.0 else val)
            
            true_probs.append(true_prob)
            buggy_probs.append(buggy_prob)
            
        print(f"\n--- {desc} (n={len(logits)}) ---")
        print(f"Mean Logit: {np.mean(logits):.4f}")
        print(f"Mean True Prob (Sigmoid): {np.mean(true_probs)*100:.1f}%")
        print(f"Mean UI Prob (Buggy):     {np.mean(buggy_probs)*100:.1f}%")
        
    evaluate_points(high_y, high_x, "High Elevation (>1000m)")
    evaluate_points(low_y, low_x, "Low Elevation (<500m)")

if __name__ == "__main__":
    main()
