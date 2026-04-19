import sys
import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xarray as xr
import argparse
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.preprocessing import compute_derived_features
from src.data.data_ignition_improved import IgnitionDataset

def main():
    parser = argparse.ArgumentParser(description="Generate Scatterplot Matrix for Feature Analysis")
    parser.add_argument("--samples", type=int, default=2000, help="Number of points to sample (balanced)")
    parser.add_argument("--year", type=int, default=2023, help="Year to analyze")
    parser.add_argument("--output", type=str, default="feature_scatterplot_matrix.png", help="Output filename")
    
    args = parser.parse_args()
    
    # 1. Cargar Datacube
    datacube_path = "data/IberFire.nc"
    if not os.path.exists(datacube_path):
        print(f"❌ Datacube not found: {datacube_path}")
        return

    print(f"📂 Loading Datacube ({args.year}) from {datacube_path}...")
    ds = xr.open_dataset(datacube_path)
    
    # Select year but keep dask lazy
    ds_year = ds.sel(time=ds.time.dt.year == args.year)
    
    # Optimization: Select random time indices first to avoid loading full year in compute_derived_features
    # We need enough points. One timestep has 900x1200 ~ 1M points.
    # So 5-10 random timesteps should be enough to get 2000 diversified points.
    import numpy as np
    total_times = len(ds_year.time)
    random_times = np.random.choice(total_times, size=10, replace=False)
    random_times.sort()
    
    print(f"📉 Optimization: Loading only {len(random_times)} timesteps...")
    ds_subset = ds_year.isel(time=random_times)
    
    # 2. Calcular Features (Safe on small subset)
    print("🛠️ Computing derived features (Subset)...")
    ds_subset = compute_derived_features(ds_subset)
    
    # 3. Extraer Datos (Sampling)
    # Seleccionamos variables clave para el plot (no todas para evitar matriz gigante)
    # Top features from SHAP + Target
    features_to_plot = [
        'FWI', 
        'RH_min', 
        't2m_mean', 
        'wind_speed_mean', 
        'NDVI', 
        'SWI_010' # Soil Water Index
    ]
    
    target_var = 'is_fire'
    
    print(f"🎲 Sampling {args.samples} points (50/50 Fire/No-Fire)...")
    
    # Extraer arrays numpy para las variables seleccionadas + target
    data_arrays = {}
    for var in features_to_plot:
        data_arrays[var] = ds_year[var].values.flatten()
    
    target_flat = ds_year[target_var].values.flatten()
    
    # Indices con fuego y sin fuego
    # Usamos un umbral > 0 para fuego (o 0.5)
    fire_indices = np.where(target_flat > 0.5)[0]
    nofire_indices = np.where(target_flat <= 0.5)[0]
    
    # Sampling balanceado
    n_per_class = args.samples // 2
    
    if len(fire_indices) < n_per_class:
        print(f"⚠️ Warning: Only {len(fire_indices)} fire pixels found. Upsampling/Using all.")
        idx_fire = np.random.choice(fire_indices, n_per_class, replace=True)
    else:
        idx_fire = np.random.choice(fire_indices, n_per_class, replace=False)
        
    idx_nofire = np.random.choice(nofire_indices, n_per_class, replace=False)
    
    selected_indices = np.concatenate([idx_fire, idx_nofire])
    np.random.shuffle(selected_indices)
    
    # Construir DataFrame
    df_data = {}
    for var in features_to_plot:
        df_data[var] = data_arrays[var][selected_indices]
    
    # Target label strings para el plot
    df_data['Condition'] = ['Fire' if target_flat[i] > 0.5 else 'No Fire' for i in selected_indices]
    
    df = pd.DataFrame(df_data)
    
    # 4. Generar Plot
    print("📊 Generating Scatterplot Matrix with Seaborn...")
    sns.set_theme(style="ticks")
    
    # Custom colors: Blue for No Fire, Red for Fire
    palette = {"No Fire": "#3498db", "Fire": "#e74c3c"}
    
    g = sns.pairplot(
        df, 
        hue="Condition", 
        hue_order=["No Fire", "Fire"],
        palette=palette,
        diag_kind="kde", # Kernel density estimation on diagonal
        plot_kws={'alpha': 0.6, 's': 15},
        height=2.5
    )
    
    g.fig.suptitle("Feature Correlation Matrix: Fire vs No-Fire Conditions", y=1.02)
    
    print(f"💾 Saving to {args.output}...")
    plt.savefig(args.output, bbox_inches='tight', dpi=150)
    print("✅ Done!")

if __name__ == "__main__":
    main()
