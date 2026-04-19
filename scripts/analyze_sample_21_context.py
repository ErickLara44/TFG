
import torch
import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from scripts.visualize_spread import CHANNELS

def analyze_sample_21():
    print("🕵️‍♂️ Analizando Contexto Biofísico de Muestra 21...")
    
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    idx = 21
    
    x, y = ds[idx]
    
    # x shape: (T, C, H, W)
    # Channels: 
    # 0: elevation
    # 6: CLC_forest
    # 7: CLC_scrub
    # ...
    # mask: Last channel
    
    fire_t0 = x[-1, -1].numpy()  # Máscara t
    fire_t1 = y.squeeze().numpy() # Máscara t+1
    
    # Indices de los pixels
    coords_t0 = np.where(fire_t0 > 0.5)
    coords_t1 = np.where(fire_t1 > 0.5)
    
    print(f"\n🔥 ANÁLISIS FUEGO t (ROJO - {len(coords_t0[0])} px):")
    # Analizar qué hay debajo del fuego que se apagó
    if len(coords_t0[0]) > 0:
        analyze_area(x, coords_t0, "SITIO DONDE SE APAGÓ")
    
    print(f"\n🔥 ANÁLISIS FUEGO t+1 (VERDE/NUEVO - {len(coords_t1[0])} px):")
    if len(coords_t1[0]) > 0:
        analyze_area(x, coords_t1, "SITIO DONDE APARECIÓ")

def analyze_area(x, coords, label):
    # Extraer valores medios en esas coordenadas
    # CLC Channels en x[-1]
    # Buscar índices en CHANNELS
    try:
        idx_forest = CHANNELS.index('CLC_current_forest_proportion')
        idx_scrub = CHANNELS.index('CLC_current_scrub_proportion')
        idx_roads = CHANNELS.index('dist_to_roads_mean')
    except:
        print("Error mapeando canales.")
        return

    forest_vals = x[-1, idx_forest][coords].numpy()
    scrub_vals = x[-1, idx_scrub][coords].numpy()
    roads_vals = x[-1, idx_roads][coords].numpy()
    
    print(f"   [{label}]")
    print(f"   - Bosque Promedio: {forest_vals.mean():.2f}")
    print(f"   - Matorral Promedio: {scrub_vals.mean():.2f}")
    print(f"   - Combustible Total: {(forest_vals + scrub_vals).mean():.2f}")
    print(f"   - Distancia a Carreteras: {roads_vals.mean():.2f} (Normalizado)")
    
    # Interpretación
    fuel = (forest_vals + scrub_vals).mean()
    if fuel < 0.1:
        print("   💡 HIPÓTESIS: Zona con MUY POCO combustible (Urbano/Agua/Roca).")
        print("      Esto explicaría por qué se apagó o por qué no debería haber fuego.")
    elif fuel > 0.8:
        print("   💡 HIPÓTESIS: Zona con MUCHO combustible.")
    
    # Carreteras
    # Si dist_to_roads es bajo, puede ser extinción humana
    # Nota: dist_to_roads suele estar normalizado. Menor valor = más cerca.
    
if __name__ == "__main__":
    analyze_sample_21()
