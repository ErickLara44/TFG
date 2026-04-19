
import sys
from pathlib import Path
import xarray as xr
import torch
import numpy as np

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.data.data_prop_improved import generate_temporal_splits
from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS

def check_feature_ranges(patch_path):
    """
    Verifica rangos de features para sanidad física.
    """
    data = torch.load(patch_path, weights_only=False)
    # ignition dataset returns stacked features: (T*C, H, W)
    # We want to check values in the LAST timestep (most relevant)
    num_vars = len(DEFAULT_FEATURE_VARS)
    
    if isinstance(data, dict):
        x = data.get('x', data.get('features', data))
    else:
        x = data
    
    # Check shape
    if x.shape[0] % num_vars != 0:
        print(f"⚠️ Warning: Channel count {x.shape[0]} not multiple of vars {num_vars}")
        
    # Extract last timestep features
    x_last = x[-num_vars:, :, :]
    
    # Check FWI (Index of FWI in DEFAULT_FEATURE_VARS)
    try:
        fwi_idx = DEFAULT_FEATURE_VARS.index('FWI')
        fwi_map = x_last[fwi_idx]
        if (fwi_map < 0).any():
             print(f"❌ FWI < 0 detectado en {patch_path.name}")
             return False
    except ValueError:
        pass
        
    # Check RH (Index of RH_min)
    try:
        rh_idx = DEFAULT_FEATURE_VARS.index('RH_min')
        rh_map = x_last[rh_idx]
        if (rh_map > 100).any():
             print(f"❌ RH > 100 detectado en {patch_path.name}")
             return False
    except ValueError:
        pass
        
    return True

def main():
    print("🔍 Verificando calidad de datos generados...")
    
    patches_dir = config.DATA_DIR / "processed" / "patches_temporal_strict_FULL"
    
    if not patches_dir.exists():
        print(f"⚠️ Directorio {patches_dir} no existe. Esperando generación...")
        return

    splits = ['train', 'val', 'test']
    
    for split in splits:
        split_dir = patches_dir / split
        if not split_dir.exists():
            continue
            
        files = list(split_dir.glob("*.pt"))
        print(f"📁 Split {split}: {len(files)} parches")
        
        if len(files) > 0:
            # Check random sample
            sample_file = files[np.random.randint(len(files))]
            if check_feature_ranges(sample_file):
                print(f"   ✅ Sample check {sample_file.name} passed physics check.")

if __name__ == "__main__":
    main()
