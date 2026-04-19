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
    print("🚀 Debugging Dataset Access...")
    
    # Load datacube
    datacube = xr.open_dataset(config.DATACUBE_PATH)
    splits = create_train_val_test_split(datacube, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Create dataset (no balancing to be fast)
    dataset = IgnitionDataset(datacube, splits['train'], temporal_context=7, balance_classes=False)
    
    print(f"\n⏱️ Testing access speed for 5 samples...")
    
    for i in range(5):
        start = time.time()
        x, y = dataset[i]
        end = time.time()
        print(f"   Sample {i}: {end - start:.4f} seconds | Shape: {x.shape}")

if __name__ == "__main__":
    main()
