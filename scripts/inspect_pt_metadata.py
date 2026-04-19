
import torch
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset

def inspect_pt_metadata():
    print("🕵️‍♂️ Inspeccionando Metadatos de Archivo .pt (Muestra 21)...")
    
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    idx = 21
    
    file_path = ds.files[idx]
    print(f"📂 Archivo: {file_path}")
    
    data = torch.load(file_path)
    print(f"🔑 Keys encontradas: {data.keys()}")
    
    if 'metadata' in data:
        print(f"📄 Metadata: {data['metadata']}")
    elif 'time_idx' in data:
        print(f"🕒 Time Index: {data['time_idx']}")
        print(f"📍 Coords: y={data['y_idx']}, x={data['x_idx']}")

if __name__ == "__main__":
    inspect_pt_metadata()
