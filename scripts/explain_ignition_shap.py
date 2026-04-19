import sys
import os
import torch
import shap
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.models.ignition import RobustFireIgnitionModel
from src.data.data_ignition_improved import (
    PrecomputedIgnitionDataset,
    DEFAULT_FEATURE_VARS,
    load_default_stats,
    build_channel_stats_arrays,
)

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for Ignition Model")
    parser.add_argument("--data_dir", type=str, default="data/processed/ignition_patches", help="Path a parches")
    parser.add_argument("--model_path", type=str, default="best_robust_ignition_model.pth", help="Path al modelo")
    parser.add_argument("--background_size", type=int, default=25, help="Tamaño del dataset de fondo para SHAP")
    parser.add_argument("--test_size", type=int, default=15, help="Muestras a explicar")
    parser.add_argument("--device", type=str, default="cpu", help="Dispositivo (CPU recomendado para SHAP si hay poca VRAM)")
    
    args = parser.parse_args()
    
    # Force CPU for SHAP to avoid VRAM OOM if using GradientExplainer on large inputs
    # Unless user specifically asks for cuda and knows what they are doing.
    # But usually SHAP operations + gradients + large tensors = OOM on GPU.
    device = torch.device(args.device)
    print(f"🖥️ Usando dispositivo para SHAP: {device}")

    import xarray as xr
    # 1. Cargar Datos desde Datacube (más robusto si no hay parches)
    datacube_path = "data/IberFire.nc"
    if not os.path.exists(datacube_path):
        print(f"❌ No se encuentra: {datacube_path}")
        return

    print(f"📂 Cargando Datacube desde {datacube_path}...")
    ds = xr.open_dataset(datacube_path)
    
    # Preprocesar
    from src.data.preprocessing import compute_derived_features
    from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS
    print("🛠️ Computando features derivados...")
    # Usar solo 2023 para la explicación (más representativo de lo actual)
    ds_year = ds.sel(time=ds.time.dt.year == 2023)
    ds_year = compute_derived_features(ds_year)

    # Cargar stats centralizadas (scripts/compute_ignition_stats.py)
    raw_stats = load_default_stats()
    if not raw_stats:
        print("⚠️ No se encontraron stats. Ejecuta: python scripts/compute_ignition_stats.py")
    stats = build_channel_stats_arrays(DEFAULT_FEATURE_VARS, raw_stats)

    # Definir dataset que normalice
    # Reusamos la logica de CoordinateIgnitionDataset pero sin coordenadas extra, 
    # o simplemente normalizamos manualmente aqui
    from src.data.data_ignition_improved import IgnitionDataset

    class NormalizedIgnitionDataset(IgnitionDataset):
        def __init__(self, *args, stats=None, **kwargs):
            self.stats = stats
            super().__init__(*args, **kwargs)
            
        def __getitem__(self, idx):
            x, y = super().__getitem__(idx)
            if self.stats:
                 mean = self.stats['mean'].to(x.device).view(1, -1, 1, 1)
                 std = self.stats['std'].to(x.device).view(1, -1, 1, 1)
                 x = (x - mean) / (std + 1e-6)
            return x, y

    # Indices temporales para 2023
    indices = [{'time_index': i} for i in range(len(ds_year.time))]
    
    # Dataset
    total_samples = args.background_size + args.test_size
    dataset = NormalizedIgnitionDataset(
        ds_year, indices, 
        temporal_context=7, 
        samples_per_epoch=total_samples,
        balance_ratio=1.0, 
        stats=stats
    )
    
    # 2. Cargar Modelo
    print(f"🏗️ Cargando modelo desde {args.model_path}...")
    x0, _ = dataset[0]
    T, C, H, W = x0.shape
    
    model = RobustFireIgnitionModel(C, T, hidden_dims=[64, 128])
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # SHAP DeepExplainer no soporta ReLU(inplace=True) con Backward Hooks
    print("🔧 Ajustando modelo para SHAP (RELU inplace=False)...")
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # 3. Preparar Background y Test Samples para SHAP
    print(f"🎲 Generando {total_samples} muestras frescas...")
    
    loader = DataLoader(dataset, batch_size=total_samples, shuffle=True)
    batch_x, batch_y = next(iter(loader))
    
    background = batch_x[:args.background_size].to(device)
    test_samples = batch_x[args.background_size:].to(device)
    
    print("🧠 Iniciando SHAP DeepExplainer (CPU)...")
    
    try:
        # Usar DeepExplainer (funciona bien con PyTorch models)
        # El modelo retorna {'ignition': logit}. SHAP espera tensor.
        
        # Wrapper para que el modelo devuelva solo el tensor de salida
        class ModelWrapper(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model
            def forward(self, x):
                out = self.model(x)
                return out['ignition'].view(-1, 1)
        
        wrapped_model = ModelWrapper(model)
        print("🧠 Iniciando SHAP DeepExplainer (CPU)...")
        explainer = shap.DeepExplainer(wrapped_model, background)
        
        print("💥 Calculando valores SHAP (Muestra a muestra para evitar OOM RAM)...")
        shap_values_list = []
        for i in range(test_samples.shape[0]):
            print(f"   Procesando muestra {i+1}/{test_samples.shape[0]}...")
            s_val = explainer.shap_values(test_samples[i:i+1], check_additivity=False)
            
            if isinstance(s_val, list):
                 s_val = s_val[0]
            if s_val.shape[-1] == 1:
                 s_val = s_val[..., 0]
                 
            shap_values_list.append(s_val)

        shap_values = np.concatenate(shap_values_list, axis=0)
        print(f"✅ SHAP Values shape final: {shap_values.shape}")
            
        # --- 1. BAR PLOT (Global Importance) ---
        print("📊 Generando Bar Plot...")
        # shap_values: (N, T, C, H, W)
        # Sumamos over (0, 1, 3, 4) -> (C)
        shap_values_bar = np.abs(shap_values).mean(axis=(0, 1, 3, 4))
        
        from src.data.data_ignition_improved import DEFAULT_FEATURE_VARS
        
        plt.figure()
        indices = np.argsort(shap_values_bar)
        
        # Bar Plot
        plt.barh(range(len(indices)), shap_values_bar[indices], color='#ff0051')
        plt.yticks(range(len(indices)), [DEFAULT_FEATURE_VARS[i] for i in indices])
        plt.xlabel("mean(|SHAP value|)")
        plt.title("Ignition Model Feature Importance")
        plt.tight_layout()
        plt.savefig("shap_importance_bar.png")
        print("✅ Guardado: shap_importance_bar.png")

        # --- 2. BEESWARM PLOT (Detailed pixel effects) ---
        print("📊 Generando Beeswarm Plot (Subsampling)...")
        # Flatten everything except Channels -> (TotalPixels, C)
        # Transpose (N, T, C, H, W) -> (N, T, H, W, C)
        shap_t = shap_values.transpose(0, 1, 3, 4, 2).reshape(-1, len(DEFAULT_FEATURE_VARS))
        feat_t = test_samples.permute(0, 1, 3, 4, 2).reshape(-1, len(DEFAULT_FEATURE_VARS)).cpu().numpy()
        
        # Subsample randomly 2000 pixels
        num_points = shap_t.shape[0]
        max_display = 2000
        if num_points > max_display:
            idx = np.random.choice(num_points, max_display, replace=False)
            shap_subset = shap_t[idx]
            feat_subset = feat_t[idx]
        else:
            shap_subset = shap_t
            feat_subset = feat_t
            
        plt.figure()
        shap.summary_plot(shap_subset, feat_subset, feature_names=DEFAULT_FEATURE_VARS, show=False)
        plt.tight_layout()
        plt.savefig("shap_beeswarm_detailed.png")
        print("✅ Guardado: shap_beeswarm_detailed.png")
        
    except Exception as e:
        print(f"❌ Error en SHAP: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
