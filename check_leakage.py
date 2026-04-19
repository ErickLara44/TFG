
import torch
import numpy as np
from pathlib import Path
from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS

def check_leakage_patches():
    patches_dir = Path("data/processed/patches/spread_224/train") 
    patch_files = list(patches_dir.glob("patch_*.pt"))
    
    if not patch_files:
        print("❌ No patches found.")
        return

    print(f"🔍 Checking {len(patch_files)} patches for leakage...")
    
    # Indice de is_near_fire
    try:
        idx_near_fire = DEFAULT_FEATURE_VARS.index('is_near_fire')
    except ValueError:
        print("❌ 'is_near_fire' not in DEFAULT_FEATURE_VARS")
        return

    leakage_count = 0
    checked_count = 0
    
    # Check a subset
    for p_file in patch_files[:100]:
        try:
            data = torch.load(p_file)
            x = data['x'] # (T, C, H, W)
            y = data['y'] # (1,) -> 1.0 or 0.0
            
            # The label y determines if there is ignition at the CENTER pixel at T+1
            # logical prediction: input [0..T] -> predict T+1
            # We want to check if the INPUT at T (the last frame) contains info about T+1
            
            # Get the last frame of the sequence
            last_frame = x[-1] # (C, H, W)
            
            # Get is_near_fire channel
            near_fire_map = last_frame[idx_near_fire] # (H, W)
            
            # Center coordinates
            H, W = near_fire_map.shape
            cy, cx = H // 2, W // 2
            
            # Value at the center
            center_val = near_fire_map[cy, cx].item()
            
            # If label is 1 (Fire), and center_val is 1, it implies "near fire" at T is true.
            # "is_near_fire" usually means "is there fire in the neighborhood?". 
            # If it's 1 RIGHT AT THE CENTER, it means there is fire at the center at time T.
            # If the task is IGNITION, we assume NO FIRE at T, and FIRE at T+1.
            # So if there is fire at T (center_val=1), then it's not ignition, it's just burning.
            
            if y.item() == 1.0:
                checked_count += 1
                # Check center value
                if center_val > 0.5: # Assuming binary 0/1, using threshold just in case
                     # If is_near_fire is 1 at center at time T, and we predict Fire at T+1...
                     # This might mean the fire was ALREADY there.
                     # "Ignition" typically implies 0 -> 1 transition.
                     # If 1 -> 1, it's fire maintenance/spread, not ignition.
                     print(f"⚠️ Patch {p_file.name}: Label=1, is_near_fire(center, T)={center_val}")
                     leakage_count += 1
                else:
                     pass # Clean ignition: 0 -> 1
                     
        except Exception as e:
            print(f"Error reading {p_file}: {e}")
            
    print("-" * 30)
    print(f"Total positive samples checked: {checked_count}")
    print(f"Samples with is_near_fire=1 at center (Potential 'Already On Fire'): {leakage_count}")
    
    if checked_count > 0:
        ratio = leakage_count / checked_count
        print(f"Ratio: {ratio:.2%}")
        if ratio > 0.1:
            print("🚨 HIGH RISK: Many positive samples already have fire at T.")
        else:
            print("✅ LOW RISK: Most positive samples are clean ignitions.")
    else:
        print("⚠️ No positive samples found in the subset.")

if __name__ == "__main__":
    check_leakage_patches()
