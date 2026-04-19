import sys
from pathlib import Path
import torch
import pickle
import numpy as np
from tqdm import tqdm

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import config
from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS

def compute_stats():
    print("📊 Calculando estadísticas de normalización desde parches...")
    
    patches_dir = config.DATA_DIR / "processed" / "patches" / "train"
    patch_files = list(patches_dir.glob("patch_*.pt"))
    
    if not patch_files:
        print("❌ No se encontraron parches de entrenamiento.")
        return
    
    # Usar un subset para velocidad (ej. 500 parches)
    n_samples = min(len(patch_files), 500)
    indices = np.random.choice(len(patch_files), n_samples, replace=False)
    selected_files = [patch_files[i] for i in indices]
    
    print(f"   Usando {n_samples} parches aleatorios...")
    
    # Acumuladores
    # Shape esperado: (T, C, H, W) -> Queremos stats por canal C
    n_channels = len(DEFAULT_FEATURE_VARS)
    
    # Welford's algorithm for online mean/std or simple accumulation
    # Simple accumulation is fine for 500 patches
    all_pixels = []
    
    for p_file in tqdm(selected_files):
        try:
            data = torch.load(p_file)
            x = data['x'] # (T, C, H, W)
            
            # Subsample pixels to save memory (e.g. stride 4)
            # Flatten: (T*H*W, C)
            x_flat = x.permute(0, 2, 3, 1).reshape(-1, n_channels)
            
            # Take a random subset of pixels (e.g. 1000 per patch)
            if x_flat.shape[0] > 1000:
                idx = torch.randperm(x_flat.shape[0])[:1000]
                x_flat = x_flat[idx]
                
            all_pixels.append(x_flat)
            
        except Exception as e:
            print(f"⚠️ Error leyendo {p_file}: {e}")
            
    # Concatenate all
    all_pixels = torch.cat(all_pixels, dim=0) # (N_total, C)
    
    print(f"   Calculando mean/std sobre {all_pixels.shape[0]} píxeles...")
    
    means = torch.nanmean(all_pixels, dim=0)
    stds = torch.from_numpy(np.nanstd(all_pixels.numpy(), axis=0))
    
    # Evitar división por cero
    stds[stds < 1e-6] = 1.0
    
    stats = {
        'mean': means.numpy(),
        'std': stds.numpy(),
        'vars': DEFAULT_FEATURE_VARS
    }
    
    # Guardar
    save_path = config.DATA_DIR / 'processed' / 'iberfire_normalization_stats.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(stats, f)
        
    print(f"✅ Estadísticas guardadas en {save_path}")
    print("   Means:", means)
    print("   Stds: ", stds)

if __name__ == "__main__":
    compute_stats()
