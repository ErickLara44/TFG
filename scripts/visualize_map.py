import sys
import os
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

# Agregar raíz del proyecto al path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import IgnitionDataset, DEFAULT_FEATURE_VARS
from src.data.preprocessing import compute_derived_features

class CoordinateIgnitionDataset(IgnitionDataset):
    """
    Extensión de IgnitionDataset que devuelve también las coordenadas (y, x) 
    y aplica normalización.
    """
    def __init__(self, *args, stats=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = stats

    def __getitem__(self, idx):
        x_tensor, y_label = super().__getitem__(idx)
        
        # Normalizar si hay stats
        if self.stats is not None:
             mean = self.stats['mean'].to(x_tensor.device).view(1, -1, 1, 1)
             std = self.stats['std'].to(x_tensor.device).view(1, -1, 1, 1)
             x_tensor = (x_tensor - mean) / (std + 1e-6)

        sample = self.samples[idx]
        return x_tensor, y_label, sample['y'], sample['x']
        return x_tensor, y_label, sample['y'], sample['x']

def main():
    parser = argparse.ArgumentParser(description="Visualizar predicciones en el mapa")
    parser.add_argument("--datacube", type=str, default="data/IberFire.nc", help="Path al datacube")
    parser.add_argument("--model_path", type=str, default="best_robust_ignition_model.pth", help="Path al modelo")
    parser.add_argument("--year", type=int, default=2023, help="Año a visualizar")
    parser.add_argument("--samples", type=int, default=200, help="Número de muestras a visualizar")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo")
    
    args = parser.parse_args()
    
    # Dispositivo
    if args.device == 'auto':
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️ Usando dispositivo: {device}")
    
    # 1. Cargar Datacube (solo el año seleccionado)
    print(f"📂 Cargando Datacube para el año {args.year}...")
    ds = xr.open_dataset(args.datacube)
    ds_year = ds.sel(time=ds.time.dt.year == args.year)
    
    if len(ds_year.time) == 0:
        print(f"❌ Error: No hay datos para el año {args.year}")
        sys.exit(1)
        
    # Preprocesar características
    print("🛠️ Computando características derivadas...")
    ds_year = compute_derived_features(ds_year)
    
    # Calcular estadísticas GLOBALES del año para normalizar (aproximación a training stats)
    print("📊 Calculando estadísticas para normalización...")
    means = []
    stds = []
    for var in DEFAULT_FEATURE_VARS:
        # Calcular mean/std solo sobre datos válidos (evitar nans de bordes si los hay, aunque fillna ayuda)
        # Usar .compute() si es dask, pero aquí asumimos numpy o cargado
        # Optimización: calcular sobre una muestra si es muy lento, pero 2023 cabe en memoria.
        da = ds_year[var]
        means.append(float(da.mean().values))
        stds.append(float(da.std().values))
    
    stats = {
        'mean': torch.tensor(means, dtype=torch.float32),
        'std': torch.tensor(stds, dtype=torch.float32)
    }
    print(f"   Stats computed. Mean[0] (Elev): {means[0]:.2f}")

    # 2. Crear Dataset
    indices = [{'time_index': i} for i in range(len(ds_year.time))]
    
    # Máscara espacial (solo tierra)
    spatial_mask = None
    if 'is_waterbody' in ds_year:
        print("🗺️ Aplicando máscara espacial (excluyendo agua)...")
        # is_waterbody: 1=agua, 0=tierra. Queremos tierra (0).
        spatial_mask = (ds_year['is_waterbody'].isel(time=0).values < 0.5)
    elif 'elevation_mean' in ds_year:
        print("🗺️ Aplicando máscara espacial (basada en elevación > 0)...")
        # Asumir elevación 0 es mar (aprox)
        elev = ds_year['elevation_mean']
        if 'time' in elev.dims: elev = elev.isel(time=0)
        spatial_mask = (elev.values > 1.0)
    
    # Usar nuestra versión con coordenadas y stats
    dataset = CoordinateIgnitionDataset(
        ds_year, indices, 
        temporal_context=7, 
        samples_per_epoch=args.samples,
        balance_ratio=1.0, # 50% fuego / 50% no fuego
        stats=stats,
        spatial_mask=spatial_mask
    )
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # 3. Cargar Modelo
    print(f"🏗️ Cargando modelo desde {args.model_path}...")
    
    # Inferir dimensiones
    x0, _, _, _ = dataset[0]
    T, C, H, W = x0.shape
    
    model = RobustFireIgnitionModel(
        num_input_channels=C,
        temporal_context=T, 
        hidden_dims=[64, 128] # Asumiendo default
    )
    
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    best_thr = checkpoint.get('threshold', 0.5)
    print(f"✅ Modelo cargado (Threshold: {best_thr:.3f})")
    
    # 4. Inferencia
    print("🚀 Ejecutando inferencia...")
    results = []
    
    with torch.no_grad():
        for x, y, y_coord, x_coord in loader:
            x = x.to(device)
            outputs = model(x)
            probs = torch.sigmoid(outputs['ignition']).cpu().numpy()
            targets = y.numpy().flatten()
            
            for i in range(len(probs)):
                pred_class = 1 if probs[i] > best_thr else 0
                
                # Clasificar resultado
                if targets[i] == 1 and pred_class == 1:
                    status = 'TP' # True Positive (Rojo)
                elif targets[i] == 0 and pred_class == 1:
                    status = 'FP' # False Positive (Naranja)
                elif targets[i] == 1 and pred_class == 0:
                    status = 'FN' # False Negative (Azul/Negro)
                else:
                    status = 'TN' # True Negative (Verde)
                    
                results.append({
                    'y': y_coord[i].item(),
                    'x': x_coord[i].item(),
                    'status': status,
                    'prob': probs[i]
                })

    # 5. Visualización
    print("🗺️ Generando mapa...")
    
    # Obtener coordenadas reales (suponiendo que ds tiene coordenadas y, x)
    # Si son índices, usamos índices. Pero idealmente ds['y'] y ds['x'] tienen valores proyectados
    ys = ds_year.y.values
    xs = ds_year.x.values
    
    # Crear plot sobre el mapa base (usando una variable estática como fondo, ej. elevación o solo máscara)
    plt.figure(figsize=(12, 10))
    
    # Fondo: Elevación media (si existe) o máscara de tierra
    if 'elevation_mean' in ds_year:
        elev_var = ds_year['elevation_mean']
        if 'time' in elev_var.dims:
            bg_data = elev_var.isel(time=0).values
        else:
            bg_data = elev_var.values
        plt.imshow(bg_data, cmap='gray', alpha=0.3, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='upper')
    
    # Separar puntos por categoría
    colors = {'TP': 'red', 'FP': 'orange', 'FN': 'blue', 'TN': 'green'}
    labels = {
        'TP': 'Fuego Real Detectado',
        'FP': 'Falsa Alarma',
        'FN': 'Fuego No Detectado',
        'TN': 'Sin Fuego (Correcto)'
    }
    markers = {'TP': '*', 'FP': 'o', 'FN': 'x', 'TN': '.'}
    
    for status in ['TP', 'FP', 'FN', 'TN']:
        subset = [r for r in results if r['status'] == status]
        if not subset:
            continue
            
        y_pts = [ys[r['y']] for r in subset]
        x_pts = [xs[r['x']] for r in subset]
        
        plt.scatter(x_pts, y_pts, c=colors[status], label=f"{labels[status]} ({len(subset)})", 
                    marker=markers[status], s=50 if status != 'TN' else 10, alpha=0.7)
        
    plt.title(f"Predicciones de Incendios en {args.year} (Muestra de {len(results)} puntos)")
    plt.xlabel("Coordenada X (m)")
    plt.ylabel("Coordenada Y (m)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    output_file = f"map_predictions_{args.year}.png"
    plt.savefig(output_file, dpi=150)
    print(f"✅ Mapa guardado en: {output_file}")

    # --- NUEVO: Mapa de Probabilidades (Heatmap) ---
    print("🗺️ Generando mapa de probabilidades...")
    plt.figure(figsize=(12, 10))
    
    # Fondo
    plt.imshow(bg_data, cmap='gray', alpha=0.3, extent=[xs.min(), xs.max(), ys.min(), ys.max()], origin='upper')

    # Extraer arrays
    all_y = [ys[r['y']] for r in results]
    all_x = [xs[r['x']] for r in results]
    all_probs = [r['prob'] for r in results]
    all_targets = [r['status'] for r in results]

    # Plot scatter con color según probabilidad
    sc = plt.scatter(all_x, all_y, c=all_probs, cmap='RdYlGn_r', vmin=0, vmax=1, s=50, edgecolors='k', alpha=0.8)
    
    # Añadir barra de color
    cbar = plt.colorbar(sc)
    cbar.set_label('Probabilidad de Ignición (Riesgo)')
    
    plt.title(f"Mapa de Riesgo de Incendio {args.year} (0.0 - 1.0)")
    plt.xlabel("Coordenada X")
    plt.ylabel("Coordenada Y")
    plt.grid(True, alpha=0.3)
    
    prob_output_file = f"map_probability_{args.year}.png"
    plt.savefig(prob_output_file, dpi=150)
    print(f"✅ Mapa de Probabilidad guardado en: {prob_output_file}")

if __name__ == "__main__":
    main()
