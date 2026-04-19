
import torch
import sys
import os
from pathlib import Path
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset
from scripts.visualize_spread import normalize_batch

def inspect_sample_2():
    # 1. Load Data
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    
    # El usuario pregunta por la muestra 2
    idx = 2 
    
    x, y = ds[idx]
    
    # t=0 (Input Fire)
    fire_t0 = x[-1, -1].numpy()
    # t+1 (Target Fire)
    fire_t1 = y.squeeze().numpy()
    
    print(f"📊 ANÁLISIS MUESTRA {idx}:")
    
    # Conteo de píxeles
    pixels_t0 = (fire_t0 > 0.5).sum()
    pixels_t1 = (fire_t1 > 0.5).sum()
    
    print(f"   Píxeles Fuego Inicial (t): {pixels_t0}")
    print(f"   Píxeles Fuego Final (t+1): {pixels_t1}")
    
    if pixels_t0 == 0 and pixels_t1 > 0:
        print("\n⚠️ CASO DETECTADO: Aparición Espontánea (Ignición).")
        print("   No había fuego en t, pero hay en t+1.")
        print("   Esto es trabajo del modelo de Ignición, no del de Propagación.")
    elif pixels_t0 > 0:
        print("\n✅ CASO NORMAL: Había fuego previo.")
        print(f"   El fuego creció de {pixels_t0} a {pixels_t1} píxeles.")
        
        # Analizar distancia
        # Si los nuevos píxeles están muy lejos de los viejos -> Spotting o Error
        # Simplificación: Ver si hay solapamiento
        overlap = ((fire_t0 > 0.5) & (fire_t1 > 0.5)).sum()
        print(f"   Píxeles que se mantienen quemados: {overlap}")
        
        if overlap == 0:
             print("   ⚠️ PERO no se tocan. El fuego saltó o se movió completamente.")

if __name__ == "__main__":
    inspect_sample_2()
