import sys
import os
import torch
import numpy as np
import pandas as pd
import xarray as xr
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import compute_derived_features

def main():
    parser = argparse.ArgumentParser(description="Audit Data Quality for Fire Ignition")
    parser.add_argument("--year", type=int, default=2023, help="Year to analyze")
    
    args = parser.parse_args()
    
    # 1. Cargar Datacube
    datacube_path = "data/IberFire.nc"
    if not os.path.exists(datacube_path):
        print(f"❌ Datacube not found: {datacube_path}")
        return

    print(f"📂 Loading Datacube ({args.year}) from {datacube_path}...")
    ds = xr.open_dataset(datacube_path)
    ds_year = ds.sel(time=ds.time.dt.year == args.year)
    
    # Load 10 random timesteps to get a representative sample without OOM
    import numpy as np
    total_times = len(ds_year.time)
    random_times = np.random.choice(total_times, size=20, replace=False) # 20 timesteps
    random_times.sort()
    
    print(f"📉 Optimization: Loading {len(random_times)} timesteps for audit...")
    ds_subset = ds_year.isel(time=random_times)
    
    # 2. Calcular Features (Subset)
    print("🛠️ Computing derived features...")
    ds_subset = compute_derived_features(ds_subset)
    
    # 3. Extraer Dataframes
    df_list = []
    
    # Variables de interés
    vars_to_check = ['FWI', 'wind_speed_mean', 'SWI_010', 'NDVI', 'is_fire']
    
    # Flatten y crear DataFrame
    print("📊 Extracting data...")
    data_dict = {}
    for var in vars_to_check:
        if var in ds_subset:
            data_dict[var] = ds_subset[var].values.flatten()
        else:
            print(f"⚠️ Warning: Variable {var} not found in dataset.")
            
    df = pd.DataFrame(data_dict)
    
    # Filtrar solo pixeles válidos (evitar NaNs si los hay)
    df = df.dropna()
    
    # Separar Fuego vs No Fuego
    df_fire = df[df['is_fire'] > 0.5]
    df_nofire = df[df['is_fire'] <= 0.5]
    
    print(f"\n🔥 Total Fire Pixels Analyzed: {len(df_fire)}")
    print(f"🌲 Total Non-Fire Pixels Analyzed: {len(df_nofire)}")
    
    # --- AUDIT 1: FWI aprox 0 en Incendios ---
    fwi_threshold = 1.0
    zero_fwi_fires = df_fire[df_fire['FWI'] < fwi_threshold]
    pct_zero_fwi = (len(zero_fwi_fires) / len(df_fire)) * 100 if len(df_fire) > 0 else 0
    
    print(f"\n🧐 AUDIT 1: FWI ~ 0 in Fires (Threshold < {fwi_threshold})")
    print(f"   - Count: {len(zero_fwi_fires)}")
    print(f"   - Percentage: {pct_zero_fwi:.2f}%")
    if pct_zero_fwi > 5:
        print("   ❌ CRITICAL: High percentage of fires with FWI ~ 0. Likely data artifact or filling.")
    else:
        print("   ✅ OK: Low percentage.")

    # --- AUDIT 2: Wind Speed Distribution ---
    print("\n🧐 AUDIT 2: Wind Speed Stats")
    print(f"   - Fire Mean Wind: {df_fire['wind_speed_mean'].mean():.2f} +/- {df_fire['wind_speed_mean'].std():.2f}")
    print(f"   - No-Fire Mean Wind: {df_nofire['wind_speed_mean'].mean():.2f} +/- {df_nofire['wind_speed_mean'].std():.2f}")
    
    # --- AUDIT 3: Wet Soil (SWI_010 > 60) ---
    wet_threshold = 60
    wet_fires = df_fire[df_fire['SWI_010'] > wet_threshold]
    pct_wet = (len(wet_fires) / len(df_fire)) * 100 if len(df_fire) > 0 else 0
    
    print(f"\n🧐 AUDIT 3: Fires on Wet Soil (SWI > {wet_threshold})")
    print(f"   - Count: {len(wet_fires)}")
    print(f"   - Percentage: {pct_wet:.2f}%")
    
    # --- AUDIT 4: High NDVI (Forests) ---
    ndvi_high = 0.8
    dense_forest_fires = df_fire[df_fire['NDVI'] > ndvi_high]
    pct_forest = (len(dense_forest_fires) / len(df_fire)) * 100 if len(df_fire) > 0 else 0
    
    print(f"\n🧐 AUDIT 4: Fires in Dense Vegetation (NDVI > {ndvi_high})")
    print(f"   - Count: {len(dense_forest_fires)}")
    print(f"   - Percentage: {pct_forest:.2f}%")
    
    # --- AUDIT 5: Artifacts (Exact Zeros) ---
    exact_zeros_fwi = df_fire[df_fire['FWI'] == 0.0]
    print(f"\n🧐 AUDIT 5: Exact 0.0 Artifacts in FWI (Fire samples)")
    print(f"   - Count: {len(exact_zeros_fwi)}")


if __name__ == "__main__":
    main()
