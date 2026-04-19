import os
import sys
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm
from scipy.ndimage import binary_dilation

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_ignition_improved import (
    IgnitionDataset,
    DEFAULT_FEATURE_VARS,
    load_default_stats,
    build_normalization_tensors,
)
from src.data.data_prop_improved import generate_temporal_splits
from src.models.ignition import RobustFireIgnitionModel

STATS = load_default_stats()


def normalize_ignition_tensor(x, channels):
    mean_t, std_t = build_normalization_tensors(
        channels, STATS, include_fire_state=False, device=x.device
    )
    return (x - mean_t) / std_t

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Procesando en {device}")

    print("🌍 Cargando Datacube para obtener la cuadrícula base de España...")
    ds_raw = xr.open_dataset('data/IberFire.nc')
    H_full, W_full = ds_raw.sizes['y'], ds_raw.sizes['x']
    spain_mask = ds_raw['is_spain'].values

    print("🗺️ Generando el Test Set completo (2022-2024)...")
    splits = generate_temporal_splits(ds_raw, strict=True)
    test_indices = splits['test']
    
    # Balance 5.0 significa que por cada fuego real, sacará 5 muestras negativas al azar en el mapa
    # para comprobar si el modelo da falsas alarmas. No ponemos límite por época.
    test_ds = IgnitionDataset(
        ds_raw, 
        test_indices, 
        temporal_context=3, 
        patch_size=64,
        feature_vars=DEFAULT_FEATURE_VARS,
        samples_per_epoch=400,
        balance_ratio=1.0,
        spatial_mask=spain_mask,
        max_fires_per_day=None
    )

    print("📦 Cargando Modelo de Ignición...")
    model = RobustFireIgnitionModel(num_input_channels=len(DEFAULT_FEATURE_VARS), temporal_context=3, hidden_dims=[64, 128]).to(device)
    model.load_state_dict(torch.load("models/best_ignition_model.pth", map_location=device))
    model.eval()

    # Acumuladores del mapa
    grid_TP = np.zeros((H_full, W_full), dtype=np.float32)
    grid_FN = np.zeros((H_full, W_full), dtype=np.float32)
    grid_FP = np.zeros((H_full, W_full), dtype=np.float32)
    grid_T0 = np.zeros((H_full, W_full), dtype=np.float32) # Fuego real
    
    # Umbral
    thr = 0.40

    dataloader = torch.utils.data.DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=4)

    print("🔍 Proyectando predicciones de Ignición en la cuadrícula de España...")
    
    with torch.no_grad():
        for batch_idx, (x_batch, y_batch) in enumerate(tqdm(dataloader)):
            # x_batch: (B, T, C, H, W)
            # y_batch: (B, 1) -> tensor normal
            
            x_t = normalize_ignition_tensor(x_batch.to(device), DEFAULT_FEATURE_VARS)
            outputs = model(x_t)
            
            if isinstance(outputs, tuple):
                ignition_prob = outputs[0]
            elif isinstance(outputs, dict):
                ignition_prob = outputs['ignition']
            else:
                ignition_prob = outputs
                
            probs = ignition_prob.cpu().numpy().flatten()
            labels = y_batch.numpy().flatten()
            
            # Recuperar las coordenadas Y, X del batch para pintarlas en el mapa global
            # El dataloader nos mezcla, así que las sacamos haciendo un indexing al array de samples del Dataset
            start_i = batch_idx * 16
            
            for i in range(len(probs)):
                sample_info = test_ds.samples[start_i + i]
                cy = sample_info["y"]
                cx = sample_info["x"]
                
                y_true = labels[i]
                y_pred = probs[i]
                
                if y_true > 0:
                    grid_T0[cy, cx] += 1
                
                if y_true > 0 and y_pred >= thr:
                    grid_TP[cy, cx] += 1 # True Positive
                elif y_true > 0 and y_pred < thr:
                    grid_FN[cy, cx] += 1 # False Negative
                elif y_true == 0 and y_pred >= thr:
                    grid_FP[cy, cx] += 1 # False Positive

    print("📊 Estadísticas de la cuadrícula global de Ignición:")
    print(f"   Sum T0 (Fuegos evaluados): {grid_T0.sum()}")
    print(f"   Sum TP (Aciertos): {grid_TP.sum()}")
    print(f"   Sum FN (Omisiones): {grid_FN.sum()}")
    print(f"   Sum FP (Falsas alrms): {grid_FP.sum()}")

    print("🎨 Dilatando píxeles para visibilidad y renderizando Mapa...")
    
    # 5 km radius to easily see the blimps
    struct = np.ones((5, 5))
    m_tp = binary_dilation(grid_TP > 0, structure=struct)
    m_fn = binary_dilation(grid_FN > 0, structure=struct)
    m_fp = binary_dilation(grid_FP > 0, structure=struct)
    m_t0 = binary_dilation(grid_T0 > 0, structure=struct)
    
    plt.figure(figsize=(15, 12), facecolor='white')
    ax = plt.gca()
    
    ax.imshow(spain_mask, cmap='gray', alpha=0.3)
    
    # False Positives (Cian)
    ax.imshow(np.ma.masked_where(~m_fp, m_fp), cmap=ListedColormap(['cyan']), alpha=0.6, interpolation='none')
    
    # Fuego Base (Morado)
    ax.imshow(np.ma.masked_where(~m_t0, m_t0), cmap=ListedColormap(['purple']), alpha=0.3, interpolation='none')
    
    # False Negatives (Rojo)
    ax.imshow(np.ma.masked_where(~m_fn, m_fn), cmap=ListedColormap(['red']), alpha=0.9, interpolation='none')
    
    # True Positives (Verde Lima)
    ax.imshow(np.ma.masked_where(~m_tp, m_tp), cmap=ListedColormap(['lime']), alpha=1.0, interpolation='none')
    
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='purple', markersize=10, alpha=0.5, label='Fuego Base (Suelo Real)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lime', markersize=10, label=f'TP: Acierto (Prob > {thr*100}%)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='FN: Omisión (No predijo fuego)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markersize=10, alpha=0.6, label=f'FP: Falsa Alarma (Prob > {thr*100}%)')
    ]
    ax.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.35, 1), fontsize=10)
    plt.title(f"Mapa Global de Errores - Modelo de Ignición (Test Set 2022-2024)", fontsize=14)
    plt.axis('off')
    
    os.makedirs('outputs_spread', exist_ok=True)
    out_path = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/global_ignition_test_errors.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Mapa guardado en: {out_path}")

if __name__ == "__main__":
    main()
