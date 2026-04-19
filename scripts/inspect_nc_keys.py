
import xarray as xr
import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config

def inspect_keys():
    path = config.DATACUBE_PATH
    print(f"📂 Abrir Datacube: {path}")
    ds = xr.open_dataset(path)
    print("🔑 Keys en el dataset:")
    print(list(ds.data_vars))

if __name__ == "__main__":
    inspect_keys()
