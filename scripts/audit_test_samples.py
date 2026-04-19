
import torch
import sys
import os
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.dataset_patch import PatchDataset

def audit_test_samples():
    print(f"🕵️‍♂️ Auditando Dataset de TEST para casos de 'Ignición Espontánea'...")
    
    test_dir = "data/processed/patches/spread_224/test"
    ds = PatchDataset(test_dir)
    
    ignition_cases = []
    spread_cases = []
    extinction_cases = []
    empty_cases = []
    
    print(f"📦 Total Muestras: {len(ds)}")
    
    for i in range(len(ds)):
        try:
            x, y = ds[i]
            
            # Fire t (Last channel of last timestep)
            fire_t = x[-1, -1].numpy()
            # Fire t+1 (Target)
            fire_t1 = y.squeeze().numpy()
            
            pixels_t = (fire_t > 0.5).sum()
            pixels_t1 = (fire_t1 > 0.5).sum()
            
            info = {
                'idx': i,
                'pixels_t': pixels_t,
                'pixels_t1': pixels_t1
            }
            
            # Overlap (Persistencia real)
            overlap = ((fire_t > 0.5) & (fire_t1 > 0.5)).sum()
            
            if pixels_t == 0 and pixels_t1 == 0:
                empty_cases.append(info)
            elif pixels_t < 10 and pixels_t1 > 50:
                ignition_cases.append(info)
            elif pixels_t > 0 and pixels_t1 == 0:
                extinction_cases.append(info)
            elif overlap == 0 and pixels_t > 0 and pixels_t1 > 0:
                # Fuego en t y en t+1, pero NO en el mismo sitio -> SALTO
                spread_cases.append({**info, 'type': 'JUMP'})
            else:
                spread_cases.append({**info, 'type': 'NORMAL'})
                
        except Exception as e:
            print(f"❌ Error en muestra {i}: {e}")

    jump_cases = [c for c in spread_cases if c.get('type') == 'JUMP']
    normal_cases = [c for c in spread_cases if c.get('type') == 'NORMAL']
    
    print("\n📊 RESULTADOS DE LA AUDITORÍA:")
    print(f"   🏠 Casos Vacíos: {len(empty_cases)}")
    print(f"   🔥 Casos Normales (Expansión contigua): {len(normal_cases)}")
    print(f"   🐇 Casos de SALTO (Spotting - Sin superposición): {len(jump_cases)}")
    print(f"   🧯 Casos de Extinción Total: {len(extinction_cases)}")
    print(f"   🧨 Casos de Ignición Espontánea: {len(ignition_cases)}")
    
    if jump_cases:
        print("\n🐇 DETALLE DE SALTOS (Spotting):")
        for c in jump_cases:
             print(f"   - Muestra {c['idx']}: {c['pixels_t']} px -> SALTO -> {c['pixels_t1']} px")
    
    if ignition_cases:
        print("\n⚠️ ALERTA: Se encontraron casos de Ignición (Fuego aparece de la nada):")
        for case in ignition_cases[:10]:
            print(f"   - Muestra {case['idx']}: t={case['pixels_t']} -> t+1={case['pixels_t1']} píxeles")
            
        print("\n💡 EXPLICACIÓN PARA EL USUARIO:")
        print("   Estos casos son imposibles de predecir para el modelo de propagación.")
        print("   El modelo busca 'expandir' lo que ve. Si no ve nada (0 píxeles), predice 0.")
        print("   Estos deberían ser manejados por el Modelo de Ignición.")

if __name__ == "__main__":
    audit_test_samples()
