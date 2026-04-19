import xarray as xr
import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.data_prop_improved import SpreadDataset, generate_temporal_splits
from src.data.dataset_patch import PatchDataset

from pyproj import Transformer

def generate_global_map():
    print("🌍 Cargando Datacube Original...")
    ds_raw = xr.open_dataset('data/IberFire.nc')
    
    print("🗺️ Re-generando el índice original del Test Set...")
    # This might take a minute to build the index_list, but we need it for mapping
    # 1. Generar splits temporales
    splits = generate_temporal_splits(ds_raw, strict=True)
    test_indices_raw = splits['test']
    
    # 2. Instanciar SpreadDataset para que filtre y nos dé la lista EXACTA de índices
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
    
    print("📦 Cargando PatchDataset de Test para identificar muestras activas...")
    patch_ds = PatchDataset("data/processed/patches/spread_224/test")
    
    # Track the active fires and their physical locations
    active_samples = []
    
    print("🔄 Configurando transformador de coordenadas (EPSG:3035 -> EPSG:4326)...")
    transformer = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    
    print("🔍 Analizando anomalías/patrones y mapeando...")
    for idx in range(len(patch_ds)):
        try:
            filename = patch_ds.files[idx]
            # Extract the original index from the filename (e.g. data/processed/patches/spread_224/test/sample_000120_orig.pt)
            basename = os.path.basename(filename)
            orig_idx_str = basename.split('_')[1] # "000120"
            orig_idx = int(orig_idx_str)
            
            x, y = patch_ds[idx]
            fire_t0 = x[-1, -1].numpy()
            fire_t1 = y.squeeze().numpy()
            
            # Filtramos solo muestras donde realmente haya fuego en t+1 (las que nos interesan)
            if fire_t1.sum() > 5:
                # Recuperar coords
                time_idx = original_indices[orig_idx]["time_index"]
                
                # Para saber x,y exactos necesitamos aplicar la logica del fire_center
                fire_mask_full = ds_raw["is_fire"].isel(time=time_idx).values
                y_idxs, x_idxs = np.where(fire_mask_full > 0)
                if len(y_idxs) > 0:
                    y_center, x_center = int(np.mean(y_idxs)), int(np.mean(x_idxs))
                else:
                    y_center, x_center = fire_mask_full.shape[0] // 2, fire_mask_full.shape[1] // 2
                
                dt = pd.to_datetime(ds_raw.time.values[time_idx])
                
                x_coord = float(ds_raw.x.values[x_center])
                y_coord = float(ds_raw.y.values[y_center])
                lon, lat = transformer.transform(x_coord, y_coord)
                
                # Clasificar
                sum_t0 = fire_t0.sum()
                if sum_t0 < 1:
                    category = "Espontaneo (T0=0)"
                    color = "red"
                else:
                    intersection = (fire_t0 > 0.5) & (fire_t1 > 0.5)
                    if intersection.sum() == 0:
                        category = "Spotting/Salto"
                        color = "orange"
                    else:
                        category = "Expansion Normal"
                        color = "green"
                        
                active_samples.append({
                    "idx": idx,
                    "date": dt.strftime('%Y-%m-%d %H:%M'),
                    "lat": lat,
                    "lon": lon,
                    "category": category,
                    "color": color
                })
        except Exception as e:
            print(f"Error mapping index {idx}: {e}")

    print(f"✅ Se han mapeado {len(active_samples)} fuegos de prueba.")
    
    # Dibujar Mapa Global de España
    plt.figure(figsize=(12, 10))
    ax = plt.axes(projection=ccrs.PlateCarree())
    
    # Añadir costa y bordes
    ax.add_feature(cfeature.COASTLINE, linewidth=1)
    ax.add_feature(cfeature.BORDERS, linewidth=1, linestyle=':')
    ax.add_feature(cfeature.LAND, facecolor='lightgray', alpha=0.5)
    
    # Set extent to Spain
    ax.set_extent([-10, 4.5, 35.5, 44.5], crs=ccrs.PlateCarree())
    
    # Dibujar puntos
    for s in active_samples:
        ax.plot(s['lon'], s['lat'], marker='o', color=s['color'], markersize=6, alpha=0.8,
                transform=ccrs.PlateCarree())
        
        # Opcional: Anotar algunos casos muy raros
        if s['category'] != "Expansion Normal":
            ax.text(s['lon'] + 0.1, s['lat'] + 0.1, f"#{s['idx']}", fontsize=8, color='black',
                    transform=ccrs.PlateCarree())

    # Custom legend
    from matplotlib.lines import Line2D
    custom_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Expansión Normal'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Espontáneo (Fallo 16x16)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Salto / Spotting')
    ]
    ax.legend(handles=custom_lines, loc='lower right')
    
    plt.title("Ubicación de Incendios en el Test Set (2022-2024)\nResaltando Anomalías Físicas", fontsize=14)
    out_path = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread/global_test_map.png"
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🗺️ Mapa guardado en {out_path}")
    
    # Guardar también como CSV por si el usuario quiere verlo en una tabla
    df = pd.DataFrame(active_samples)
    csv_path = "/Users/erickmollinedolara/.gemini/antigravity/brain/a777d26d-4695-4944-9752-d3eb977c01d1/outputs_spread/test_samples_coordinates.csv"
    df.to_csv(csv_path, index=False)
    print(f"📊 Coordenadas guardadas en {csv_path}")

if __name__ == "__main__":
    generate_global_map()
