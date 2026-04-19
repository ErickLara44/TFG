
import torch
from pathlib import Path
from tqdm import tqdm
import os

def check_and_clean(data_dir):
    data_dir = Path(data_dir)
    files = sorted(list(data_dir.rglob("*.pt")))
    print(f"🔍 Checking {len(files)} files in {data_dir}...")
    
    corrupted_count = 0
    for f in tqdm(files):
        try:
            # Try to load
            torch.load(f, weights_only=False)
        except Exception as e:
            # If fails, delete
            print(f"🗑️ Deleting corrupted file: {f} ({e})")
            os.remove(f)
            corrupted_count += 1
            
    print(f"✅ Finished. Deleted {corrupted_count} corrupted files.")

if __name__ == "__main__":
    check_and_clean("data/processed/patches/spread_224")
