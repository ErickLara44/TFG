import sys
import os
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from tqdm import tqdm

# Agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import IgnitionDataset, DEFAULT_FEATURE_VARS
from src.data.preprocessing import compute_derived_features
from scripts.visualize_map import CoordinateIgnitionDataset

def main():
    parser = argparse.ArgumentParser(description="Visualizar evento de fuego en alta resolución")
    parser.add_argument("--datacube", type=str, default="data/IberFire.nc", help="Path al datacube")
    parser.add_argument("--model_path", type=str, default="best_robust_ignition_model.pth", help="Path al modelo")
    parser.add_argument("--year", type=int, default=2023, help="Año a visualizar")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo")
    parser.add_argument("--roi_size", type=int, default=128, help="Tamaño de la región de interés (ROI)")
    
    args = parser.parse_args()
    
    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
        
    print(f"🖥️ Usando dispositivo: {device}")
    
    # 1. Cargar Datacube
    print(f"📂 Cargando Datacube {args.year}...")
    ds = xr.open_dataset(args.datacube)
    ds_year = ds.sel(time=ds.time.dt.year == args.year)

    if 'is_waterbody' in ds_year:
        print("🗺️ Aplicando máscara tierra...")
        wb = ds_year['is_waterbody']
        if 'time' in wb.dims:
            land_mask = (wb.isel(time=0).values < 0.5)
        else:
            land_mask = (wb.values < 0.5)

    # 2. Encontrar un incendio GRANDE para visualizar
    print("🔍 Buscando el incendio más grande del año...")
    fire_mask = ds_year['is_fire'].values
    
    # Sumar en tiempo para ver dónde hubo más fuego acumulado
    total_fire = fire_mask.sum(axis=0)
    
    # Encontrar coordenadas del pico de fuego
    y_peak, x_peak = np.unravel_index(np.argmax(total_fire), total_fire.shape)
    print(f"🔥 Pico de fuego encontrado en (y={y_peak}, x={x_peak})")
    
    # Definir ROI
    half_roi = args.roi_size // 2
    y_start = max(0, y_peak - half_roi)
    y_end = min(ds_year.sizes['y'], y_peak + half_roi)
    x_start = max(0, x_peak - half_roi)
    x_end = min(ds_year.sizes['x'], x_peak + half_roi)
    
    print(f"📐 ROI definida: Y[{y_start}:{y_end}], X[{x_start}:{x_end}] ({y_end-y_start}x{x_end-x_start})")
    
    # Recortar dataset al ROI (+ padding para contexto)
    pad = 32 # Contexto para el modelo
    ds_roi = ds_year.isel(
        y=slice(max(0, y_start-pad), min(ds_year.sizes['y'], y_end+pad)),
        x=slice(max(0, x_start-pad), min(ds_year.sizes['x'], x_end+pad))
    )
    
    # Preprocesar
    print("🛠️ Computando features en ROI...")
    ds_roi = compute_derived_features(ds_roi)
    
    # Stats (Dummys o aproximados, idealmente usar los de training)
    # Usaremos los de visualization_map hardcoded si es posible, o calculados
    # Calculamos local stats para normalizar (batch norm style for inference)
    means = []
    stds = []
    for var in DEFAULT_FEATURE_VARS:
        da = ds_roi[var]
        means.append(float(da.mean().values))
        stds.append(float(da.std().values))
    stats = {
        'mean': torch.tensor(means, dtype=torch.float32), 
        'std': torch.tensor(stds, dtype=torch.float32)
    }

    # 3. Crear Dataset DENSO (cada píxel en el ROI central)
    # Queremos predecir para cada píxel en el rango original [y_start, y_end]
    # El dataset necesita coordenadas relativas al ds_roi
    
    # Encontrar indice temporal donde hubo fuego en el pico
    # (argmax en tiempo para ese pixel)
    t_peak_idx = np.argmax(ds_year['is_fire'].isel(y=y_peak, x=x_peak).values)
    t_peak_val = ds_year.time.values[t_peak_idx]
    print(f"⏰ Momento del incendio: {t_peak_val} (Index {t_peak_idx})")
    
    # Crear indices para predicción espacial en ese momento
    # Scaneamos TODOS los pixeles del ROI
    # CoordinateIgnitionDataset espera una lista de "muestras". 
    # Hack: Crearemos un dataset manual o usamos la clase existente pero forzando las coordenadas
    
    # Vamos a iterar y predecir row by row para no explotar memoria
    # Mejor: Usar el modelo en modo "Fully Convolutional" si se pudiera, pero ConvLSTM espera secuencia.
    # Usaremos sliding window en batches.
    
    # Generar lista de coordenadas RELATIVAS al ds_roi que corresponden al ROI central efectivo
    # El ds_roi tiene padding. El centro del ds_roi es nuestro target.
    # target_y en ds_roi va de pad a pad + roi_h
    
    roi_h = y_end - y_start
    roi_w = x_end - x_start
    
    prediction_indices = []
    # Predecimos para el dia t_peak_idx - 1 (para predecir t_peak)
    # O predecimos para una ventana alrededor del evento
    target_time = t_peak_idx
    
    # Crear grid de puntos a predecir
    # OJO: Esto puede ser lento (128*128 = 16k parches).
    # Con batch 128 -> 128 steps. Rápido.
    
    print(f"📍 Generando {roi_h * roi_w} parches para inferencia densa...")
    
    # Ajustar indices para que coincidan con la clase Dataset
    # La clase dataset usa indices globales del ds pasado.
    # Pasamos ds_roi.
    # Queremos samples centrados en (y, x) de ds_roi.
    
    # CoordinateIgnitionDataset genera samples buscando fuego o random. 
    # NO nos sirve. Necesitamos forzar coordenadas.
    
    class DenseGridDataset(IgnitionDataset):
        def __init__(self, *args, stats=None, **kwargs):
            # Guardar stats y no pasarlo a super()
            self.stats = stats
            super().__init__(*args, **kwargs)

        def _generate_spatial_samples(self):
            # Generar grid completo
            samples = []
            # Rango en ds_roi que corresponde al ROI real (quitando padding)
            # pad es 32.
            # Queremos y desde pad hasta pad+roi_h
            for y in range(pad, pad + roi_h):
                for x in range(pad, pad + roi_w):
                    samples.append({
                        "time_index": target_time,
                        "y": y,
                        "x": x,
                        "label": 0.0 # Dummy
                    })
            return samples
            
        def __getitem__(self, idx):
            # Copiar lógica de normalización manual aquí para no depender de la otra clase
            x_tensor, y_label = super().__getitem__(idx)
            if self.stats is not None:
                 mean = self.stats['mean'].to(x_tensor.device).view(1, -1, 1, 1)
                 std = self.stats['std'].to(x_tensor.device).view(1, -1, 1, 1)
                 x_tensor = (x_tensor - mean) / (std + 1e-6)
            
            sample = self.samples[idx]
            return x_tensor, sample['y'], sample['x'] # Sin label

    dense_ds = DenseGridDataset(
        ds_roi, 
        [{'time_index': target_time}], 
        temporal_context=7, 
        stats=stats
    )
    # Sobrescribir stats en la instancia (hack porque __init__ original no lo guarda en self.stats si no se pasa, ah no, mi DenseGrid sí)
    dense_ds.stats = stats
    
    loader = DataLoader(dense_ds, batch_size=64, num_workers=0) # Workers 0 para evitar overhead en loops cortos
    
    # Cargar modelo
    x0, _, _ = dense_ds[0]
    T, C, H, W = x0.shape
    model = RobustFireIgnitionModel(C, T, hidden_dims=[64, 128]).to(device)
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Inferencia
    print("🚀 Ejecutando inferencia densa...")
    pred_map = np.zeros((roi_h, roi_w))
    
    with torch.no_grad():
        for x_batch, ys, xs in tqdm(loader):
            x_batch = x_batch.to(device)
            out = model(x_batch)
            probs = torch.sigmoid(out['ignition']).cpu().numpy().flatten()
            
            # Mapear de vuelta al grid local (quitando el offset del pad)
            # ys son coordenadas en ds_roi. Restamos pad para tener 0..roi_h
            local_ys = ys - pad
            local_xs = xs - pad
            
            # Vectorizado
            # Asegurar indices validos
            valid = (local_ys >= 0) & (local_ys < roi_h) & (local_xs >= 0) & (local_xs < roi_w)
            
            if valid.any():
                pred_map[local_ys[valid], local_xs[valid]] = probs[valid]

    # 4. Visualización Comparativa
    print("🎨 Generando visualización...")
    
    # Ground Truth en el ROI (acumulado o snapshot?)
    # Snapshot del momento target_time + 1 (lo que predecimos)
    gt_map = ds_roi['is_fire'].isel(time=target_time+1).values
    # Recortar GT al ROI (quitar padding)
    gt_map = gt_map[pad:pad+roi_h, pad:pad+roi_w]
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # Mapa GT
    ax1 = axes[0]
    im1 = ax1.imshow(gt_map, cmap='hot', vmin=0, vmax=1)
    ax1.set_title(f"Fuego Real (Ground Truth)\n{t_peak_val}")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Mapa Predicción
    ax2 = axes[1]
    im2 = ax2.imshow(pred_map, cmap='inferno', vmin=0, vmax=1)
    ax2.set_title(f"Probabilidad de Ignición (Modelo)\nRiesgo estimado")
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    out_path = f"fire_event_analysis_{args.year}.png"
    plt.savefig(out_path, dpi=150)
    print(f"✅ Análisis guardado en: {out_path}")

if __name__ == "__main__":
    main()
