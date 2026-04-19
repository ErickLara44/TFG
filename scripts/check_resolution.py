
import xarray as xr
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def check_res():
    ds = xr.open_dataset(config.DATACUBE_PATH)
    
    # Check x and y diffs
    if 'x' in ds.coords and 'y' in ds.coords:
        x_diff = np.abs(np.diff(ds.x.values)).mean()
        y_diff = np.abs(np.diff(ds.y.values)).mean()
        
        print(f"📏 Resolución X promedio: {x_diff:.2f} metros (aprox)")
        print(f"📏 Resolución Y promedio: {y_diff:.2f} metros (aprox)")
        
        # Check units if available
        if 'units' in ds.x.attrs:
            print(f"   Unidades X: {ds.x.attrs['units']}")
            
        print(f"\n🧠 Interpretación para Crop 16x16:")
        print(f"   Ancho en Km: {(16 * x_diff) / 1000:.2f} km")
        print(f"   Radio desde el centro: {(8 * x_diff) / 1000:.2f} km")
        
    else:
        print("❌ No se encontraron coordenadas x/y estándar.")

if __name__ == "__main__":
    check_res()
