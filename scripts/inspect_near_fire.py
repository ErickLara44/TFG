import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import xarray as xr
import numpy as np
import src.config as config

def main():
    print(f"📂 Cargando Datacube: {config.DATACUBE_PATH}")
    ds = xr.open_dataset(config.DATACUBE_PATH)
    
    if 'is_near_fire' not in ds:
        print("❌ 'is_near_fire' NO está en el datacube.")
        return

    print("✅ 'is_near_fire' encontrado.")
    
    print("� Buscando un momento con fuego...")
    # Iterate to find fire
    fire_da = ds['is_fire']
    t_idx = -1
    for t in range(0, len(ds.time), 100): # Jump to find faster
        if fire_da.isel(time=t).sum() > 10:
            t_idx = t
            break
            
    if t_idx == -1:
        print("❌ No se encontraron fuegos en el muestreo rápido.")
        return
        
    print(f"🔥 Fuego encontrado en índice {t_idx} ({ds.time[t_idx].values})")
    
    # Analyze around this time
    slice_t = slice(t_idx, t_idx + 5)
    
    fire_sample = ds['is_fire'].isel(time=slice_t).values
    near_sample = ds['is_near_fire'].isel(time=slice_t).values
    
    for i in range(5):
        f = fire_sample[i]
        n = near_sample[i]
        print(f"\n⏰ T+{i}:")
        print(f"   Fuego Pixeles: {np.sum(f > 0)}")
        print(f"   Near  Pixeles: {np.sum(n > 0)}")
        
        # Check Overlap
        overlap = (f > 0) & (n > 0)
        print(f"   Overlap: {np.sum(overlap)}")
        
        # Check if near predicts next fire (Leakage test)
        if i < 4:
            f_next = fire_sample[i+1]
            overlap_future = (n > 0) & (f_next > 0)
            print(f"   Overlap con Fuego T+1: {np.sum(overlap_future)}")

    description = ds['is_near_fire'].attrs.get('description', 'No description')
    print(f"\n📝 Descripción de variable: {description}")

if __name__ == "__main__":
    main()
