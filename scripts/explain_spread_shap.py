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

from src.models.prop import RobustFireSpreadModel
from src.data.dataset_patch import PatchDataset
from src.data.data_prop_improved import (
    DEFAULT_FEATURE_VARS as CHANNELS,
    load_default_stats,
    build_normalization_tensors,
)

FEATURE_VARS = list(CHANNELS) + ['Fire_Mask_T0']
STATS = load_default_stats()

def normalize_batch(x):
    mean_t, std_t = build_normalization_tensors(
        CHANNELS, STATS, include_fire_state=True, device=x.device
    )
    return (x - mean_t) / (std_t + 1e-6)

def main():
    parser = argparse.ArgumentParser(description="SHAP Analysis for Fire Spread Model")
    parser.add_argument("--data_dir", type=str, default="data/processed/patches/spread_224/test")
    parser.add_argument("--model_path", type=str, default="models/best_spread_model.pth")
    parser.add_argument("--crop_size", type=int, default=32, help="Tamaño de recorte (Center Crop)")
    parser.add_argument("--background_size", type=int, default=25, help="Tamaño dataset fondo SHAP")
    parser.add_argument("--test_size", type=int, default=15, help="Muestras a explicar")
    parser.add_argument("--device", type=str, default="cpu")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(f"🖥️ Usando dispositivo para SHAP: {device}")

    # 1. Dataset
    print(f"📂 Cargando parches desde {args.data_dir}...")
    dataset = PatchDataset(args.data_dir)
    
    if len(dataset) == 0:
        print("❌ Dataset vacío. Revisa la ruta.")
        return

    loader = DataLoader(dataset, batch_size=args.background_size + args.test_size, shuffle=True)
    try:
        batch_x, batch_y = next(iter(loader))
    except StopIteration:
        print("❌ No hay suficientes datos.")
        return

    batch_x = normalize_batch(batch_x).to(device)
    
    # On the fly Center Crop to emulate training input
    import torchvision.transforms.functional as TF
    if args.crop_size < 224:
        B, T, C, H, W = batch_x.shape
        # CenterCrop applied to spatial dims. We unbind time
        x_crops = []
        for t in range(T):
            x_t = batch_x[:, t] # (B, C, H, W)
            x_t_cropped = TF.center_crop(x_t, output_size=args.crop_size)
            x_crops.append(x_t_cropped.unsqueeze(1))
        batch_x = torch.cat(x_crops, dim=1) # (B, T, C, crop_H, crop_W)

    # 2. Modelo
    T, C, H, W = batch_x.shape[1], batch_x.shape[2], batch_x.shape[3], batch_x.shape[4]
    print(f"🏗️ Inicializando modelo: T={T}, C={C}, H={H}, W={W}")
    
    model = RobustFireSpreadModel(input_channels=C, hidden_dims=[64, 128])
    if os.path.exists(args.model_path):
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        # Handle cases where full dict or just model state was saved
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        print("✅ Weights loaded.")
    else:
        print(f"⚠️ CUIDADO: Weights no encontrados en {args.model_path}")
    
    model.to(device)
    model.eval()

    # SHAP DeepExplainer no soporta ReLU(inplace=True) 
    print("🔧 Ajustando modelo para SHAP (RELU inplace=False)...")
    for module in model.modules():
        if isinstance(module, torch.nn.ReLU):
            module.inplace = False

    # 3. Preparar Background y Test
    background = batch_x[:args.background_size]
    test_samples = batch_x[args.background_size:]
    
    # Wrapper para SHAP (SHAP espera un tensor de salida, no dict)
    class SpreadModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            
        def forward(self, x):
            out = self.model(x)
            # Se reduce a un escalar para evitar 1024 backwards en CPU (ahorrando horas de ejecución)
            return out['spread_probability'].mean(dim=[1, 2, 3]).view(x.size(0), -1)

    wrapped_model = SpreadModelWrapper(model)
    
    print("🧠 Iniciando SHAP DeepExplainer (CPU)...")
    explainer = shap.DeepExplainer(wrapped_model, background)
    
    print("💥 Calculando valores SHAP iterativamente...")
    shap_values_list = []
    
    for i in range(test_samples.shape[0]):
        print(f"   Procesando muestra {i+1}/{test_samples.shape[0]}...")
        s_val = explainer.shap_values(test_samples[i:i+1], check_additivity=False)
        
        # s_val type depends on out shape. We flattened to (B, H*W).
        # It's a list if multiple outputs, or directly a numpy array.
        if isinstance(s_val, list):
             s_val = s_val[0]
             
        # Reshape to keep consistency with input: (B, H*W, T, C, H_in, W_in) -> sum over predicted pixels
        # For simplicity in global feature importance, we can just aggregate over the output spatial dimensions.
        # DeepExplainer on multiple outputs returns a list of arrays. Or an array of shape (num_outputs, batch_size, ...)
        
        # We need the values relative to the input shape: (1, T, C, H, W)
        # DeepExplainer sum over output dims if we just care about feature impact.
        # Actually SHAP gives list of length (H*W) with shape (1, T, C, H, W).
        # We will average over the output dimension list.
        if isinstance(s_val, list):
            s_val = np.mean(np.array(s_val), axis=0) # (1, T, C, H, W)
            
        shap_values_list.append(s_val)

    shap_values = np.concatenate(shap_values_list, axis=0)
    print(f"✅ SHAP Values shape final: {shap_values.shape}")
    
    if shap_values.shape[-1] == 1:
        shap_values = shap_values[..., 0]

    # --- 1. BAR PLOT ---
    print("📊 Generando Bar Plot...")
    shap_values_bar = np.abs(shap_values).mean(axis=(0, 1, 3, 4))
    
    plt.figure()
    indices = np.argsort(shap_values_bar)
    plt.barh(range(len(indices)), shap_values_bar[indices], color='#ff7b00')
    plt.yticks(range(len(indices)), [FEATURE_VARS[i] for i in indices])
    plt.xlabel("mean(|SHAP value|)")
    plt.title("Propagation Model Feature Importance")
    plt.tight_layout()
    plt.savefig("shap_spread_importance_bar.png")
    
    # --- 2. BEESWARM PLOT ---
    print("📊 Generando Beeswarm Plot...")
    shap_t = shap_values.transpose(0, 1, 3, 4, 2).reshape(-1, len(FEATURE_VARS))
    feat_t = test_samples.permute(0, 1, 3, 4, 2).reshape(-1, len(FEATURE_VARS)).cpu().numpy()
    
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
    shap.summary_plot(shap_subset, feat_subset, feature_names=FEATURE_VARS, show=False)
    plt.tight_layout()
    plt.savefig("shap_spread_beeswarm_detailed.png")
    print("✅ Todo completado.")

if __name__ == "__main__":
    main()
