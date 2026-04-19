import torch
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.dataset_patch import PatchDataset

def analyze_anomalies():
    print("Iniciando análisis de anomalías en el Test Set...")
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    
    anomalies = {
        "no_fire_t0": [],          # Fuego aparece de la nada en t=1
        "complete_extinction": [], # Fuego desaparece por completo en t=1
        "teleportation": [],       # Fuego en t=0 y t=1 no tienen solapamiento (salto)
        "normal_expansion": []     # Comportamiento esperado
    }
    
    total_valid = 0
    
    for idx in range(len(ds)):
        try:
            x, y = ds[idx]
            fire_t0 = x[-1, -1].numpy()
            fire_t1 = y.squeeze().numpy()
            
            # Solo iteramos sobre muestras donde pasa ALGO en t=1
            if fire_t1.sum() <= 5: 
                continue
                
            total_valid += 1
            sum_t0 = fire_t0.sum()
            sum_t1 = fire_t1.sum()
            
            if sum_t0 < 1:
                anomalies["no_fire_t0"].append((idx, sum_t1))
                continue
                
            intersection = (fire_t0 > 0.5) & (fire_t1 > 0.5)
            if sum_t1 > 0 and intersection.sum() == 0:
                anomalies["teleportation"].append((idx, sum_t0, sum_t1))
            elif sum_t1 < 1:
                pass # Already filtered by sum_t1 > 5, but for logic
            else:
                anomalies["normal_expansion"].append((idx, sum_t0, sum_t1))
                
        except Exception as e:
            pass

    # Complete extincion (fire in t0, but NOT in t1)
    for idx in range(len(ds)):
        try:
            x, y = ds[idx]
            fire_t0 = x[-1, -1].numpy()
            fire_t1 = y.squeeze().numpy()
            if fire_t0.sum() > 5 and fire_t1.sum() <= 5:
                anomalies["complete_extinction"].append(idx)
        except:
            pass

    print(f"\n--- REPORTE DE ANOMALÍAS EN TEST SET ---")
    print(f"Total de muestras analizadas con fuego en t+1: {total_valid}")
    print(f"1. Fuego Espontáneo (Aparecen de la nada, No fuego en t=0): {len(anomalies['no_fire_t0'])} casos")
    print(f"2. Fuego Teletransportado (Jump/Spotting, 0 Solapamiento): {len(anomalies['teleportation'])} casos")
    print(f"3. Expansión Normal (Crecimiento adyacente esperado): {len(anomalies['normal_expansion'])} casos")
    print(f"4. Extinción Completa (Fuego en t=0 desaparece en t=1): {len(anomalies['complete_extinction'])} casos")
    
    print("\nEjemplos de Fuego Espontáneo (idx, pixeles en t+1):", anomalies["no_fire_t0"][:5])
    print("Ejemplos de Teletransportación (idx, pixeles t0, pixeles t1):", anomalies["teleportation"][:5])

if __name__ == "__main__":
    analyze_anomalies()
