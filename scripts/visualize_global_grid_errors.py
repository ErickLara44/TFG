import os
import sys
import torch
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    SpreadDataset,
    generate_temporal_splits,
    load_default_stats,
    build_normalization_tensors,
)
from src.models.prop import RobustFireSpreadModel

STATS = load_default_stats()

def normalize_batch(x, channels):
    mean_t, std_t = build_normalization_tensors(
        channels, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / std_t

def create_global_error_grid():
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🚀 Procesando en {device}")

    print("🌍 Cargando Datacube para obtener la cuadrícula base de España...")
    try:
        ds_raw = xr.open_dataset('data/IberFire.nc')
    except Exception as e:
        print(f"Error cargando IberFire.nc: {e}")
        return

    H_full, W_full = ds_raw.sizes['y'], ds_raw.sizes['x']
    
    # Extraer máscara de España para el fondo
    try:
        spain_mask = ds_raw['is_spain'].values
    except:
        spain_mask = np.ones((H_full, W_full)) # Fallback si no existe
        
    print(f"📏 Cuadrícula global: {H_full}x{W_full} píxeles (1 pixel = 1km²)")

    print("🗺️ Re-generando el índice original del Test Set...")
    splits = generate_temporal_splits(ds_raw, strict=True)
    test_indices_raw = splits['test']
    
    CHANNELS = ['elevation_mean', 'slope_mean', 'wind_u', 'wind_v', 'hydric_stress', 'solar_risk', 'CLC_current_forest_proportion', 'CLC_current_scrub_proportion', 'FWI', 'NDVI', 'dist_to_roads_mean']
    spread_ds = SpreadDataset(
        ds_raw, 
        test_indices_raw, 
        temporal_context=3, 
        filter_fire_samples=True, 
        preload_ram=False, 
        crop_size=32, 
        feature_vars=CHANNELS
    )
    original_indices = spread_ds.indices

    print("📦 Cargando PatchDataset y Modelo...")
    patch_ds = PatchDataset("data/processed/patches/spread_224/test")
    
    in_channels = len(CHANNELS) + 1 
    model = RobustFireSpreadModel(input_channels=in_channels, hidden_dims=[64, 128]).to(device)
    model.load_state_dict(torch.load("models/best_spread_model.pth", map_location=device))
    model.eval()

    # ACUMULADORES GLOBALES (920 x 1188)
    # Acumularemos aquí dónde acierta y dónde falla el modelo en todo el dataset de test.
    # Al sumar, creamos un "heatmap" si un fuego ocurre múltiples veces en el mismo sitio.
    grid_TP = np.zeros((H_full, W_full), dtype=np.float32)
    grid_FN = np.zeros((H_full, W_full), dtype=np.float32)
    grid_FP = np.zeros((H_full, W_full), dtype=np.float32)
    grid_T0 = np.zeros((H_full, W_full), dtype=np.float32) # Fuego inicial

    crop_size = 32
    half_crop = crop_size // 2

    print("🔍 Proyectando predicciones en la cuadrícula de España...")
    with torch.no_grad():
        for idx in tqdm(range(len(patch_ds))):
            try:
                # 1. Recuperar índice original para saber CÓMO se sacó el crop
                filename = patch_ds.files[idx]
                basename = os.path.basename(filename)
                orig_idx_str = basename.split('_')[1] # e.g. "000120"
                orig_idx = int(orig_idx_str)
                time_idx = original_indices[orig_idx]["time_index"]
                
                # 2. Re-calcular el centro del fuego para este time_idx (igual que hace SpreadDataset)
                fire_mask_full = ds_raw["is_fire"].isel(time=time_idx).values
                y_idxs, x_idxs = np.where(fire_mask_full > 0)
                if len(y_idxs) > 0:
                    cy, cx = int(np.mean(y_idxs)), int(np.mean(x_idxs))
                else:
                    cy, cx = H_full // 2, W_full // 2
                
                # Calcular límites en la cuadrícula global
                y1 = cy - half_crop
                y2 = cy + half_crop
                x1 = cx - half_crop
                x2 = cx + half_crop
                
                # Límites seguros en la matriz global
                y1_safe, y2_safe = max(0, y1), min(H_full, y2)
                x1_safe, x2_safe = max(0, x1), min(W_full, x2)
                
                # Límites del crop (por si se sale de España y hubo padding)
                py1 = max(0, -y1)         # Si y1 < 0, ignoramos los primeros 'py1' pixeles del parche
                py2 = crop_size - max(0, y2 - H_full)
                px1 = max(0, -x1)
                px2 = crop_size - max(0, x2 - W_full)
                
                # 3. Leer datos y predecir
                x, y_true_var = patch_ds[idx]
                
                # Si el true mask está vacío, ignoramos (esto pasa a veces por el padding extremo o si no hay fuego)
                if y_true_var.sum() < 1:
                    continue
                    
                x_b = x.unsqueeze(0).to(device)
                x_b = normalize_batch(x_b, CHANNELS)
                
                outputs = model(x_b)
                if isinstance(outputs, dict):
                    outputs = outputs['spread_probability']
                
                y_pred = torch.sigmoid(outputs).squeeze().cpu().numpy()
                y_true = y_true_var.squeeze().numpy()
                fire_t0 = x[-1, -1].numpy()
                
                # 4. Calcular Matrices de Confusión a nivel píxel
                # Umbrales
                thr_pred = 0.5
                thr_fp = 0.5 # Nubes cian = "Riesgo/Falsa alarma" > 50%
                
                tp_mask = (y_true > 0.5) & (y_pred >= thr_pred)
                fn_mask = (y_true > 0.5) & (y_pred < thr_pred)
                fp_mask = (y_true < 0.5) & (y_pred > thr_fp)
                
                # Extraer solo la parte del crop que cae dentro de España
                tp_crop = tp_mask[py1:py2, px1:px2]
                fn_crop = fn_mask[py1:py2, px1:px2]
                fp_crop = fp_mask[py1:py2, px1:px2]
                t0_crop = (fire_t0 > 0.5)[py1:py2, px1:px2]
                
                # 5. Sumar a los acumuladores globales
                grid_TP[y1_safe:y2_safe, x1_safe:x2_safe] += tp_crop
                grid_FN[y1_safe:y2_safe, x1_safe:x2_safe] += fn_crop
                grid_FP[y1_safe:y2_safe, x1_safe:x2_safe] += fp_crop * y_pred[py1:py2, px1:px2] # Sumar la probabilidad para ver intensidad
                grid_T0[y1_safe:y2_safe, x1_safe:x2_safe] += t0_crop

            except Exception as e:
                print(f"Error procesando {idx}: {e}")
                
    # --- DIBUJAR MAPA GLOBAL ---
    from scipy.ndimage import binary_dilation
    
    print("📊 Estadísticas de la cuadrícula global:")
    print(f"   Sum T0 (Fuego hist): {grid_T0.sum()}")
    print(f"   Sum TP (Aciertos): {grid_TP.sum()}")
    print(f"   Sum FN (Omisiones): {grid_FN.sum()}")
    print(f"   Sum FP (Falsas alrms): {grid_FP.sum()}")

    print("💾 Guardando matrices en .npz para visor interactivo rápido...")
    os.makedirs('outputs_spread', exist_ok=True)
    np.savez_compressed("outputs_spread/global_grid_data.npz",
                        grid_T0=grid_T0, grid_TP=grid_TP, 
                        grid_FN=grid_FN, grid_FP=grid_FP, 
                        spain_mask=spain_mask)

    print("🎨 Renderizando Mapa Global para exportar...")
    # Tamaño mucho más grande para ver las celdas, y usaremos un DPI altísimo
    fig, ax = plt.subplots(figsize=(60, 45), facecolor='white')
    
    # 1. Fondo de España
    ax.imshow(spain_mask, cmap='gray', alpha=0.3)
    
    # NO DILATAMOS: Queremos ver el pixel exacto de 1km x 1km
    
    # 2. Fuego T0 (Morado)
    m_t0 = np.ma.masked_where(grid_T0 == 0, grid_T0)
    ax.imshow(m_t0, cmap=ListedColormap(['purple']), alpha=0.5, interpolation='none')
    
    # 3. False Positives (Cian/Heatmap) -> Aquí usamos el umbral de thr_fp = 0.1 (10%)
    grid_FP_norm = grid_FP / (np.max(grid_FP) + 1e-9)
    m_fp = np.ma.masked_where(grid_FP == 0, grid_FP_norm)
    ax.imshow(m_fp, cmap=ListedColormap(['cyan']), alpha=0.6, interpolation='none')
    
    # 4. False Negatives (Rojo)
    m_fn = np.ma.masked_where(grid_FN == 0, grid_FN)
    ax.imshow(m_fn, cmap=ListedColormap(['red']), alpha=0.9, interpolation='none')
    
    # 5. True Positives (Verde Lima)
    m_tp = np.ma.masked_where(grid_TP == 0, grid_TP)
    ax.imshow(m_tp, cmap=ListedColormap(['lime']), alpha=1.0, interpolation='none')
    
    # DIBUJAR LA CUADRÍCULA (GRID) DE 1KM X 1KM
    print("📏 Dibujando las líneas negras de la cuadrícula...")
    # Configurar los ticks menores para que caigan exactamente entre los píxeles
    ax.set_xticks(np.arange(-.5, W_full, 1), minor=True)
    ax.set_yticks(np.arange(-.5, H_full, 1), minor=True)
    
    # Línea extremadamente fina (0.02) para que no se "coma" la imagen al hacer zoom out
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.05, alpha=0.8)
    
    # Ocultar los ticks (los numeritos)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Leyenda
    from matplotlib.lines import Line2D
    import matplotlib.patches as mpatches
    custom_lines = [
        Line2D([0], [0], color='purple', lw=8, alpha=0.5, label='Fuego t=0 (Histórico)'),
        Line2D([0], [0], color='lime', lw=8, label='TP: Acierto (Fuego predicho correctamente > 50%)'),
        Line2D([0], [0], color='red', lw=8, label='FN: Omisión (Fuego IMPREVISTO por el modelo)'),
        mpatches.Patch(color='cyan', alpha=0.6, label='FP: Falsa Alarma (Riesgo asignado > 10%)')
    ]
    plt.legend(handles=custom_lines, loc='upper right', bbox_to_anchor=(1.25, 1), fontsize=40)
    
    plt.title("Rendimiento del Modelo de Propagación (Test Set 2022-2024)\nCuadrícula Real de España (1km x 1km)", fontsize=60)
    plt.axis('off')
    
    # Guardar en PDF para VECTORES (Zoom infinito) y PNG altísimo DPI
    out_path_png = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread/global_pixel_grid_errors_highres.png"
    out_path_pdf = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread/global_pixel_grid_errors_vector.pdf"
    
    print("💾 Guardando en alta resolución (puede tardar un minuto)...")
    plt.tight_layout()
    plt.savefig(out_path_png, dpi=400, bbox_inches='tight', facecolor='white')
    print(f"✅ Mapa PNG guardado en:\n{out_path_png}")
    
    try:
        plt.savefig(out_path_pdf, format='pdf', bbox_inches='tight', facecolor='white')
        print(f"✅ Mapa PDF vectorial guardado en:\n{out_path_pdf}")
    except Exception as e:
        print(f"No se pudo guardar PDF: {e}")
        
    plt.close()

if __name__ == "__main__":
    create_global_error_grid()
