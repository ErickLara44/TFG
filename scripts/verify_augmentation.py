import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_augmentation(sample_id_base, data_dir):
    suffixes = [
        'orig', 'rot90', 'rot180', 'rot270', 
        'flipH', 'flipV', 'flipH_rot90', 'flipV_rot90'
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    data_dir = Path(data_dir)
    
    # Índices (Asumiendo CHANNELS standard en prepare_spread_patches.py)
    # 0: elev, 1: slope, 2: wind_u, 3: wind_v
    idx_u = 2
    idx_v = 3
    
    for i, suffix in enumerate(suffixes):
        filename = f"{sample_id_base}_{suffix}.pt"
        filepath = data_dir / "train" / filename
        
        if not filepath.exists():
            print(f"⚠️ Missing: {filepath}")
            continue
            
        data = torch.load(filepath, weights_only=False)
        x = data['x']  # (T, C, H, W)
        y = data['y']  # (1, H, W)
        
        # Usamos el último tiempo T-1
        x_last = x[-1]
        
        # Extraer viento
        u = x_last[idx_u].numpy()
        v = x_last[idx_v].numpy()
        
        # Extraer máscara de fuego (target y)
        mask = y[0].numpy()
        
        ax = axes[i]
        
        # Plot máscara
        im = ax.imshow(mask, cmap='hot', alpha=0.6, origin='upper')
        
        # Plot vectores de viento (subsampling para claridad)
        step = 4
        H, W = mask.shape
        Y, X = np.mgrid[0:H:step, 0:W:step]
        U = u[::step, ::step]
        V = v[::step, ::step]
        
        # Quiver
        # Angles 'xy' means coords. Pivot 'mid' centers arrow.
        ax.quiver(X, Y, U, -V, color='cyan', scale=50, width=0.005) 
        # Note: matplotlib quiver Y is generally increasing downwards for images if origin='upper'
        # but standard math logic is Y up.
        # If origin='upper', Y increases down. So +V (North) should point UP (negative pixel index).
        # Hence -V.
        
        ax.set_title(f"{suffix}\nMean Wind: ({u.mean():.2f}, {v.mean():.2f})")
        ax.axis('off')
        
    plt.tight_layout()
    output_path = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/augmentation_check.png"
    plt.savefig(output_path)
    print(f"✅ Augmentation check saved to {output_path}")

if __name__ == "__main__":
    plot_augmentation("sample_000000", "data/processed/patches/spread_224")
