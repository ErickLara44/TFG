import os
import sys
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import (
    DEFAULT_FEATURE_VARS,
    load_default_stats,
    build_normalization_tensors,
)

STATS = load_default_stats()


def normalize_ignition_tensor(x, channels):
    mean_t, std_t = build_normalization_tensors(
        channels, STATS, include_fire_state=False, device=x.device
    )
    return (x - mean_t) / std_t

def visualize_ignition_heatmap(date_str="2022-08-01", stride=8):
    """
    Escanea la cuadrícula de España usando una ventana de 64x64 y guarda un mapa de Riesgo de Ignición.
    stride: Cada cuántos kilómetros (píxeles) evaluamos el modelo. 
    1 = pixel a pixel (tarda HORAS). 
    8 = salta de 8 en 8 km y luego interpola (rápido para tener una idea global).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Procesando en {device}")

    try:
        ds_raw = xr.open_dataset('data/IberFire.nc')
    except Exception as e:
        print(f"Error cargando IberFire.nc: {e}")
        return

    H_full, W_full = ds_raw.sizes['y'], ds_raw.sizes['x']
    spain_mask = ds_raw['is_spain'].values
    
    # 1. Buscar el time_index exacto del date_str solicitado
    time_array = ds_raw.time.values
    time_idx = np.where(time_array == np.datetime64(date_str))[0]
    
    if len(time_idx) == 0:
        # Fallback al primer día de agosto (época fuerte) si falla la fecha exacta
        time_idx = [np.where(time_array == np.datetime64("2022-08-01"))[0][0]]
        print(f"⚠️ Fecha {date_str} no encontrada. Usando 2022-08-01 por defecto.")
        
    t0 = time_idx[0]
    temporal_context = 3
    crop_size = 64
    half_crop = crop_size // 2

    print("📦 Cargando Modelo de Ignición...")
    channels = DEFAULT_FEATURE_VARS
    model = RobustFireIgnitionModel(num_input_channels=len(channels), temporal_context=temporal_context, hidden_dims=[64, 128]).to(device)
    model.load_state_dict(torch.load("models/best_ignition_model.pth", map_location=device))
    model.eval()

    # Mapa de salida (usaremos una resolución un pelín más baja si el stride > 1 para no iterar 1 millón de veces)
    risk_map = np.zeros((H_full, W_full), dtype=np.float32)
    # T0 Fuego empírico ese día
    fire_map = ds_raw["is_fire"].isel(time=t0).values
    
    print(f"🔍 Escaneando la cuadrícula Peninsular el {ds_raw.time.values[t0]} (Stride={stride}km)...")
    
    # Pre-cargar las variables del cubo en RAM para no ahogar al disco duro en el bucle
    print("📥 Cargando matrices temporales a RAM para acelerar...")
    cube_ram = {}
    for var in channels:
        if var in ds_raw:
            if 'time' in ds_raw[var].dims:
                cube_ram[var] = ds_raw[var].isel(time=slice(max(0, t0 - temporal_context + 1), t0 + 1)).values
            else:
                cube_ram[var] = ds_raw[var].values
                
    with torch.no_grad():
        # Iterar la península saltando según el stride
        # Rellenaremos bloques de (stride x stride) con el valor del riesgo central
        for cy in tqdm(range(half_crop, H_full - half_crop, stride)):
            # Optimización hiper-rápida: Si toda esta fila (de alto 'stride') no tiene suelo español, la saltamos enterita
            if not spain_mask[cy:cy+stride, :].any():
                continue
                
            for cx in range(half_crop, W_full - half_crop, stride):
                if not spain_mask[cy, cx]:
                    continue  # Es mar, Francia o Portugal
                    
                y1, y2 = cy - half_crop, cy + half_crop
                x1, x2 = cx - half_crop, cx + half_crop
                
                # Leer secuencia
                x_seq = []
                for dt in range(temporal_context):
                    t_ram = dt
                    ch_slices = []
                    for var in channels:
                        if 'time' in ds_raw[var].dims:
                            ch_slices.append(cube_ram[var][t_ram, y1:y2, x1:x2])
                        else:
                            ch_slices.append(cube_ram[var][y1:y2, x1:x2])
                    x_seq.append(np.stack(ch_slices, axis=0))
                    
                x_arr = np.stack(x_seq, axis=0) # (T, C, H, W)
                x_t = torch.FloatTensor(x_arr).unsqueeze(0).to(device)
                
                x_t = normalize_ignition_tensor(x_t, channels)
                
                outputs = model(x_t)
                if isinstance(outputs, tuple): # Modelo Ignition Robust devuelve varias cosas
                    ignition_prob = outputs[0]
                elif isinstance(outputs, dict):
                    ignition_prob = outputs['ignition']
                else:
                    ignition_prob = outputs
                    
                prob = torch.sigmoid(ignition_prob).item()
                
                # Rellenar el bloque entero del stride para que la imagen se vea rellena
                risk_map[cy - stride//2 : cy + stride//2, cx - stride//2 : cx + stride//2] = prob

    # Máscara estricta de España
    risk_map = np.ma.masked_where(~spain_mask, risk_map)
    fire_map = np.ma.masked_where(~spain_mask, fire_map > 0)

    print("🎨 Renderizando Mapa Global de Riesgo de Ignición...")
    fig, ax = plt.subplots(figsize=(24, 18), facecolor='white')
    
    # 1. Fondo (Terreno base)
    ax.imshow(spain_mask, cmap='gray', alpha=0.3)
    
    # 2. Mapa de Calor del Riesgo
    cmap = plt.cm.get_cmap('YlOrRd')
    im = ax.imshow(risk_map, cmap=cmap, alpha=0.8, interpolation='bilinear', vmin=0.0, vmax=1.0)
    
    # 3. Fuego Reales de ese día (Aciertos / Fallos del terreno empírico)
    fire_dilated = fire_map
    ax.imshow(np.ma.masked_where(~fire_map, fire_map), cmap=ListedColormap(['#00FF00']), alpha=1.0)
    
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probabilidad de Ignición', fontsize=16)

    plt.title(f"Riesgo de Ignición en España el {ds_raw.time.values[t0]}\n(Evaluación exhaustiva iterativa del terreno)", fontsize=24)
    plt.axis('off')
    
    out_path = f"/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread/global_ignition_risk_map.png"
    plt.savefig(out_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✅ Mapa de Riesgo guardado en:\n{out_path}")

if __name__ == "__main__":
    visualize_ignition_heatmap(date_str="2022-08-01", stride=8)
