import sys
from pathlib import Path
import time
import xarray as xr
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.data.data_ignition_improved import IgnitionDataset
from src.data.data_prop_improved import create_train_val_test_split

def main():
    print("🚀 Debugging Patch Generation...")
    
    # Load datacube
    datacube = xr.open_dataset(config.DATACUBE_PATH)
    splits = create_train_val_test_split(datacube, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Create dataset with patches (subset for speed)
    dataset = IgnitionDataset(
        datacube, 
        splits['train'][:100], 
        temporal_context=7, 
        patch_size=128,
        samples_per_epoch=10, 
        balance_ratio=1.0
    )
    
    print(f"\n⏱️ Testing access speed for 5 samples...")
    
    for i in range(min(5, len(dataset))):
        start = time.time()
        x, y = dataset[i]
        end = time.time()
        print(f"   Sample {i}: {end - start:.4f} seconds | X Shape: {x.shape} | Y: {y.item()}")
        
        # Verify shape
        if x.shape[-2:] != (128, 128):
            print(f"❌ Error: Expected 128x128, got {x.shape[-2:]}")

if __name__ == "__main__":
    main()
